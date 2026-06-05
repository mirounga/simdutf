#ifndef SIMDUTF_STDSIMD_SIMD_H
#define SIMDUTF_STDSIMD_SIMD_H

// std::simd wrapper presenting the EXACT public interface of haswell/simd.h.
//
// Storage is portable std::simd with EXPLICITLY PINNED widths so the ABI is
// independent of the command-line ISA. The width is parameterized by the macro
// SIMDUTF_STDSIMD_VEC_BYTES (16 = SSE, 32 = AVX2 default, 64 = AVX512); the vec
// aliases are derived from it in tier_ops.h:
//     simd8<T>   <-> ss::vec<uint8_t , BYTES>     (BYTES == 16/32/64-byte reg)
//     simd16<T>  <-> ss::vec<uint16_t, BYTES/2>
//     simd32<T>  <-> ss::vec<uint32_t, BYTES/4>
//     simd64<T>  <-> ss::vec<uint64_t, BYTES/8>
//     simd8x64 NUM_CHUNKS == 64 / BYTES
//
// Ordinary operations are portable std::simd. A handful of operations have no
// portable form; the wrapper routes them through the width-agnostic tier_*
// helper API in tier_ops.h (the per-tier impl in tier_ops_{sse,avx2,avx512}.h
// owns the actual intrinsics). The wrapper itself spells NO raw _mm* intrinsic.
// Operations using the escape hatch:
//     lookup_16   -> tier_shuffle      (per-128-lane pshufb)
//     prev<N>     -> tier_alignr<N>    (cross-lane concatenation byte shift)
//     to_bitmask  -> tier_movemask8    (byte high-bit movemask)
//     swap_bytes  -> tier_byteswap16/32
//     pack        -> tier_pack16       (saturating u16->u8)
//     saturating_sub -> tier_subs_epu8
//     sum_bytes / sum_8bytes -> tier_sum_bytes / tier_sad_8groups
//     store_ascii_as_utf16/32 -> tier_cvt8to16 / tier_cvt8to32
// Only the x86 arm is exercised now; arm/scalar arms are marked TODO(verify).

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
namespace simd {

// ---- pinned-width vector aliases + per-tier escape hatches ---------------
// tier_ops.h defines `ss`, the width-derived vec aliases (v8/m8/v16/m16/v32/
// m32/v64) off SIMDUTF_STDSIMD_VEC_BYTES, and the tier_* helper API the wrapper
// uses for the non-portable operations.
#include "simdutf/stdsimd/tier_ops.h"

// ---- portable load/store helpers (mirror haswell load/store) ------------
simdutf_really_inline v8 loadu8(const std::uint8_t *ptr) {
  return ss::unchecked_load<v8>(ptr, SIMDUTF_STDSIMD_VEC_BYTES, ss::flag_default);
}
simdutf_really_inline void storeu8(const v8 v, std::uint8_t *ptr) {
  ss::unchecked_store(v, ptr, SIMDUTF_STDSIMD_VEC_BYTES, ss::flag_default);
}

// Forward-declared so they can be used by splat and friends.
template <typename Child> struct base {
  v8 value;

  // Zero constructor
  simdutf_really_inline base() : value{} {}

  // Conversion from the portable vector
  simdutf_really_inline base(const v8 _value) : value(_value) {}

  simdutf_really_inline operator const v8 &() const { return this->value; }
  simdutf_really_inline operator v8 &() { return this->value; }

  template <endianness big_endian>
  simdutf_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    tier_cvt8to16(this->value, ptr, big_endian == endianness::BIG);
  }

  simdutf_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    tier_cvt8to32(this->value, ptr);
  }

  // Build a Child carrying the given raw byte storage, regardless of which
  // element-typed constructor the Child exposes.
  static simdutf_really_inline Child from_storage(const v8 raw) {
    Child c;
    c.value = raw;
    return c;
  }

  // Bit operations (portable)
  simdutf_really_inline Child operator|(const Child other) const {
    return from_storage(this->value | other.value);
  }
  simdutf_really_inline Child operator&(const Child other) const {
    return from_storage(this->value & other.value);
  }
  simdutf_really_inline Child operator^(const Child other) const {
    return from_storage(this->value ^ other.value);
  }
  simdutf_really_inline Child &operator|=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast | other;
    return *this_cast;
  }
};

// Forward-declared so they can be used by splat and friends.
template <typename T> struct simd8;

template <typename T, typename Mask = simd8<bool>>
struct base8 : base<simd8<T>> {
  simdutf_really_inline base8() : base<simd8<T>>() {}

  simdutf_really_inline base8(const v8 _value) : base<simd8<T>>(_value) {}

  friend simdutf_always_inline Mask operator==(const simd8<T> lhs,
                                               const simd8<T> rhs) {
    // Comparison on the pinned-width vec -> mask, bit-cast back into the
    // simd8<bool> register representation (all-ones / all-zeros lanes).
    const m8 m = (lhs.value == rhs.value);
    return Mask(ss::select(m, v8(std::uint8_t(0xFF)), v8(std::uint8_t(0))));
  }

  static const int SIZE = sizeof(v8);

  template <int N = 1>
  simdutf_really_inline simd8<T> prev(const simd8<T> prev_chunk) const {
    // Cross-lane concatenation byte shift: ESCAPE HATCH.
    return simd8<T>(tier_alignr<N>(this->value, prev_chunk.value));
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd8<bool> : base8<bool> {
  static simdutf_really_inline simd8<bool> splat(bool _value) {
    return v8(std::uint8_t(-(!!_value)));
  }

  simdutf_really_inline simd8() : base8<bool>() {}

  simdutf_really_inline simd8(const v8 _value) : base8<bool>(_value) {}

  simdutf_really_inline simd8(bool _value) : base8<bool>(splat(_value)) {}

  simdutf_really_inline uint64_t to_bitmask() const {
    // movemask: ESCAPE HATCH.
    return tier_movemask8(this->value);
  }
};

template <typename T> struct base8_numeric : base8<T> {
  static simdutf_really_inline simd8<T> splat(T _value) {
    return v8(std::uint8_t(_value));
  }
  static simdutf_really_inline simd8<T> zero() { return v8{}; }
  static simdutf_really_inline simd8<T>
  load(const T values[SIMDUTF_STDSIMD_VEC_BYTES]) {
    return loadu8(reinterpret_cast<const std::uint8_t *>(values));
  }
  // Repeat 16 values as many times as necessary to fill the pinned-width vector
  // (usually for per-128-lane pshufb lookup tables). The 16-byte pattern must be
  // replicated across EVERY 128-bit lane: once for SSE (16B), twice for AVX2
  // (32B), four times for AVX512 (64B) -- otherwise tier_shuffle reads a garbage
  // table in the upper lanes. tier_repeat16 broadcasts the 16-byte lane to all
  // lanes (folds to a constant + broadcast).
  static simdutf_really_inline simd8<T> repeat_16(T v0, T v1, T v2, T v3, T v4,
                                                  T v5, T v6, T v7, T v8, T v9,
                                                  T v10, T v11, T v12, T v13,
                                                  T v14, T v15) {
    const std::uint8_t lane[16] = {
        std::uint8_t(v0),  std::uint8_t(v1),  std::uint8_t(v2),
        std::uint8_t(v3),  std::uint8_t(v4),  std::uint8_t(v5),
        std::uint8_t(v6),  std::uint8_t(v7),  std::uint8_t(v8),
        std::uint8_t(v9),  std::uint8_t(v10), std::uint8_t(v11),
        std::uint8_t(v12), std::uint8_t(v13), std::uint8_t(v14),
        std::uint8_t(v15)};
    return simd8<T>(tier_repeat16(lane));
  }

  simdutf_really_inline base8_numeric() : base8<T>() {}
  simdutf_really_inline base8_numeric(const v8 _value) : base8<T>(_value) {}

  // Store to array
  simdutf_really_inline void store(T dst[SIMDUTF_STDSIMD_VEC_BYTES]) const {
    storeu8(this->value, reinterpret_cast<std::uint8_t *>(dst));
  }

  // Addition/subtraction are the same for signed and unsigned (wrapping)
  simdutf_really_inline simd8<T> operator-(const simd8<T> other) const {
    return v8(this->value - other.value);
  }
  simdutf_really_inline simd8<T> &operator-=(const simd8<T> other) {
    *this = *this - other;
    return *static_cast<simd8<T> *>(this);
  }

  // Override to distinguish from bool version
  simdutf_really_inline simd8<T> operator~() const { return *this ^ 0xFFu; }

  // Perform a lookup assuming the value is between 0 and 16 (undefined behavior
  // for out of range values). pshufb: ESCAPE HATCH.
  template <typename L>
  simdutf_really_inline simd8<L> lookup_16(simd8<L> lookup_table) const {
    // Per-128-lane pshufb: ESCAPE HATCH.
    return simd8<L>(tier_shuffle(lookup_table.value, this->value));
  }

  template <typename L>
  simdutf_really_inline simd8<L>
  lookup_16(L replace0, L replace1, L replace2, L replace3, L replace4,
            L replace5, L replace6, L replace7, L replace8, L replace9,
            L replace10, L replace11, L replace12, L replace13, L replace14,
            L replace15) const {
    return lookup_16(simd8<L>::repeat_16(
        replace0, replace1, replace2, replace3, replace4, replace5, replace6,
        replace7, replace8, replace9, replace10, replace11, replace12,
        replace13, replace14, replace15));
  }
};

// Signed bytes
template <> struct simd8<int8_t> : base8_numeric<int8_t> {
  simdutf_really_inline simd8() : base8_numeric<int8_t>() {}
  simdutf_really_inline simd8(const v8 _value) : base8_numeric<int8_t>(_value) {}

  // Splat constructor
  simdutf_really_inline simd8(int8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  simdutf_really_inline simd8(const int8_t values[SIMDUTF_STDSIMD_VEC_BYTES])
      : simd8(load(values)) {}
  simdutf_really_inline operator simd8<uint8_t>() const;

  // signed view of the pinned byte storage
  using sv8 = ss::vec<std::int8_t, SIMDUTF_STDSIMD_VEC_BYTES>;

  simdutf_really_inline bool is_ascii() const {
    // high bit clear in every lane
    const sv8 sv = std::bit_cast<sv8>(this->value);
    return ss::none_of(sv < sv8(0));
  }
  // Order-sensitive comparisons
  simdutf_really_inline simd8<bool> operator>(const simd8<int8_t> other) const {
    const sv8 a = std::bit_cast<sv8>(this->value);
    const sv8 b = std::bit_cast<sv8>(other.value);
    const m8 m = std::bit_cast<m8>(a > b);
    return simd8<bool>(ss::select(m, v8(std::uint8_t(0xFF)), v8(std::uint8_t(0))));
  }
  simdutf_really_inline simd8<bool> operator<(const simd8<int8_t> other) const {
    const sv8 a = std::bit_cast<sv8>(this->value);
    const sv8 b = std::bit_cast<sv8>(other.value);
    const m8 m = std::bit_cast<m8>(a < b);
    return simd8<bool>(ss::select(m, v8(std::uint8_t(0xFF)), v8(std::uint8_t(0))));
  }
};

// Unsigned bytes
template <> struct simd8<uint8_t> : base8_numeric<uint8_t> {
  simdutf_really_inline simd8() : base8_numeric<uint8_t>() {}
  simdutf_really_inline simd8(const v8 _value) : base8_numeric<uint8_t>(_value) {}
  // Splat constructor
  simdutf_really_inline simd8(uint8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  simdutf_really_inline simd8(const uint8_t values[SIMDUTF_STDSIMD_VEC_BYTES])
      : simd8(load(values)) {}
  // Member-by-member initialization
  simdutf_really_inline
  simd8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5,
        uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9, uint8_t v10,
        uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
        uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19, uint8_t v20,
        uint8_t v21, uint8_t v22, uint8_t v23, uint8_t v24, uint8_t v25,
        uint8_t v26, uint8_t v27, uint8_t v28, uint8_t v29, uint8_t v30,
        uint8_t v31)
      : simd8(init_bytes(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                         v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
                         v24, v25, v26, v27, v28, v29, v30, v31)) {}

  static simdutf_really_inline v8
  init_bytes(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4,
             uint8_t v5, uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9,
             uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14,
             uint8_t v15, uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
             uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23, uint8_t v24,
             uint8_t v25, uint8_t v26, uint8_t v27, uint8_t v28, uint8_t v29,
             uint8_t v30, uint8_t v31) {
    alignas(32) const std::uint8_t buf[32] = {
        v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
        v22, v23, v24, v25, v26, v27, v28, v29, v30, v31};
    return loadu8(buf);
  }

  // Saturated math: ESCAPE HATCH (no portable saturating sub).
  simdutf_really_inline simd8<uint8_t>
  saturating_sub(const simd8<uint8_t> other) const {
    return simd8<uint8_t>(tier_subs_epu8(this->value, other.value));
  }

  // Order-specific operations
  simdutf_really_inline simd8<uint8_t>
  min_val(const simd8<uint8_t> other) const {
    return v8(ss::min(other.value, this->value));
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  simdutf_really_inline simd8<uint8_t>
  gt_bits(const simd8<uint8_t> other) const {
    return this->saturating_sub(other);
  }
  simdutf_really_inline simd8<bool>
  operator>=(const simd8<uint8_t> other) const {
    return other.min_val(*this) == other;
  }

  // Bit-specific operations
  simdutf_really_inline bool is_ascii() const {
    using sv8 = ss::vec<std::int8_t, SIMDUTF_STDSIMD_VEC_BYTES>;
    const sv8 sv = std::bit_cast<sv8>(this->value);
    return ss::none_of(sv < sv8(0));
  }
  simdutf_really_inline bool bits_not_set_anywhere() const {
    return ss::none_of(this->value != v8{});
  }

  simdutf_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }

  template <int N> simdutf_really_inline simd8<uint8_t> shr() const {
    // Portable per-element shift, then mask the bits that crossed lanes
    // (mirrors haswell's _mm256_srli_epi16 + &(0xFF>>N) trick, but std::simd
    // shifts bytes element-wise so the mask is unnecessary; we keep it for
    // exact parity).
    return v8((this->value >> N) & v8(std::uint8_t(0xFFu >> N)));
  }

  simdutf_really_inline uint64_t sum_bytes() const {
    return tier_sum_bytes(this->value);
  }
};
simdutf_really_inline simd8<int8_t>::operator simd8<uint8_t>() const {
  return this->value;
}

template <typename T> struct simd8x64 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd8<T>);
  static_assert(NUM_CHUNKS == 64 / SIMDUTF_STDSIMD_VEC_BYTES,
                "simd8x64 must tile a 64-byte block with the pinned width.");
  // Bits contributed to a 64-bit bitmask by each chunk's byte-movemask.
  static constexpr int CHUNK_BITS = SIMDUTF_STDSIMD_VEC_BYTES;
  simd8<T> chunks[NUM_CHUNKS];

  // Low CHUNK_BITS bits set; masks each chunk's movemask before it is shifted
  // into place (the avx512 single-chunk case is the full 64 bits).
  static constexpr uint64_t CHUNK_MASK =
      (CHUNK_BITS >= 64) ? ~uint64_t(0) : ((uint64_t(1) << CHUNK_BITS) - 1);

  simd8x64(const simd8x64<T> &o) = delete; // no copy allowed
  simd8x64<T> &
  operator=(const simd8<T> other) = delete; // no assignment allowed
  simd8x64() = delete;                      // no default constructor allowed

  // Per-chunk constructor: exactly NUM_CHUNKS chunks (2 on AVX2, 4 on SSE,
  // 1 on AVX512). Variadic so a single definition serves every tier; constrained
  // to simd8<T> arguments so it never competes with the pointer constructor.
  template <typename... Chunks>
    requires(sizeof...(Chunks) == NUM_CHUNKS &&
             (std::is_same_v<Chunks, simd8<T>> && ...))
  simdutf_really_inline simd8x64(const Chunks... cs) : chunks{cs...} {}
  simdutf_really_inline simd8x64(const T *ptr) {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      chunks[i] = simd8<T>::load(ptr + i * sizeof(simd8<T>) / sizeof(T));
    }
  }

  simdutf_really_inline void store(T *ptr) const {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i].store(ptr + i * sizeof(simd8<T>) / sizeof(T));
    }
  }

  simdutf_really_inline uint64_t to_bitmask() const {
    uint64_t r = 0;
    for (int i = 0; i < NUM_CHUNKS; i++) {
      r |= (uint64_t(this->chunks[i].to_bitmask()) & CHUNK_MASK)
           << (CHUNK_BITS * i);
    }
    return r;
  }

  simdutf_really_inline simd8x64<T> &operator|=(const simd8x64<T> &other) {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i] |= other.chunks[i];
    }
    return *this;
  }

  simdutf_really_inline simd8<T> reduce_or() const {
    simd8<T> r = this->chunks[0];
    for (int i = 1; i < NUM_CHUNKS; i++) {
      r = r | this->chunks[i];
    }
    return r;
  }

  simdutf_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  template <endianness endian>
  simdutf_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i].template store_ascii_as_utf16<endian>(
          ptr + sizeof(simd8<T>) * i);
    }
  }

  simdutf_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * i);
    }
  }

  // Build a 64-bit bitmask by applying a per-chunk predicate that returns a
  // simd8<bool>; each chunk's byte-movemask is masked and shifted into place.
  template <typename Pred>
  simdutf_really_inline uint64_t bitmask_of(Pred pred) const {
    uint64_t r = 0;
    for (int i = 0; i < NUM_CHUNKS; i++) {
      r |= (uint64_t(pred(this->chunks[i]).to_bitmask()) & CHUNK_MASK)
           << (CHUNK_BITS * i);
    }
    return r;
  }

  simdutf_really_inline uint64_t in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);
    return bitmask_of([&](const simd8<T> c) {
      return (c <= mask_high) & (c >= mask_low);
    });
  }

  simdutf_really_inline uint64_t lt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return bitmask_of([&](const simd8<T> c) { return c < mask; });
  }

  simdutf_really_inline uint64_t gt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return bitmask_of([&](const simd8<T> c) { return c > mask; });
  }
  simdutf_really_inline uint64_t eq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return bitmask_of([&](const simd8<T> c) { return c == mask; });
  }
  simdutf_really_inline uint64_t gteq_unsigned(const uint8_t m) const {
    const simd8<uint8_t> mask = simd8<uint8_t>::splat(m);
    return bitmask_of([&](const simd8<T> c) {
      return simd8<uint8_t>(c.value) >= mask;
    });
  }
}; // struct simd8x64<T>

#include "simdutf/stdsimd/simd16-inl.h"
#include "simdutf/stdsimd/simd32-inl.h"
#include "simdutf/stdsimd/simd64-inl.h"

simdutf_really_inline simd64<uint64_t> sum_8bytes(const simd8<uint8_t> v) {
  return simd64<uint64_t>(tier_sad_8groups(v.value));
}

} // namespace simd

} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#endif // SIMDUTF_STDSIMD_SIMD_H
