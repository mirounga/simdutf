#ifndef SIMDUTF_STDSIMD_SIMD_H
#define SIMDUTF_STDSIMD_SIMD_H

// std::simd wrapper presenting the EXACT public interface of haswell/simd.h.
//
// Storage is portable std::simd with EXPLICITLY PINNED widths so the ABI is
// independent of the command-line ISA:
//     simd8<T>   <-> std::simd::vec<uint8_t , 32>  (32 bytes == __m256i)
//     simd16<T>  <-> std::simd::vec<uint16_t, 16>
//     simd32<T>  <-> std::simd::vec<uint32_t,  8>
//     simd64<T>  <-> std::simd::vec<uint64_t,  4>
//
// Ordinary operations are portable std::simd. A handful of operations have no
// portable form and use guarded intrinsic escape hatches, bridged through
// std::bit_cast<__m256i> <-> std::bit_cast<vec<uint8_t,32>>:
//     lookup_16      (pshufb            -> _mm256_shuffle_epi8)
//     prev<N>        (lane-cross alignr -> _mm256_alignr_epi8 + permute2x128)
//     to_bitmask     (movemask         -> _mm256_movemask_epi8)
//     swap_bytes     (BE byteswap      -> _mm256_shuffle_epi8)
//     pack           (lane shuffle     -> _mm256_packus_epi16)
// Only the x86 arm is exercised now; arm/scalar arms are marked TODO(verify).

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
namespace simd {

namespace ss = std::simd;

// ---- pinned-width portable vector storage types -------------------------
using v8 = ss::vec<std::uint8_t, 32>;  // 32 bytes
using m8 = ss::mask<std::uint8_t, 32>; // byte mask

// ---- escape-hatch bridge (x86) ------------------------------------------
#if SIMDUTF_IS_X86_64
simdutf_really_inline __m256i to_m256i(const v8 v) {
  return std::bit_cast<__m256i>(v);
}
simdutf_really_inline v8 from_m256i(const __m256i r) {
  return std::bit_cast<v8>(r);
}
#endif

// ---- portable load/store helpers (mirror haswell load/store) ------------
simdutf_really_inline v8 loadu8(const std::uint8_t *ptr) {
  return ss::unchecked_load<v8>(ptr, 32, ss::flag_default);
}
simdutf_really_inline void storeu8(const v8 v, std::uint8_t *ptr) {
  ss::unchecked_store(v, ptr, 32, ss::flag_default);
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
#if SIMDUTF_IS_X86_64
    const __m256i self = to_m256i(this->value);
    __m256i first = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(self));
    __m256i second = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(self, 1));
    if (big_endian == endianness::BIG) {
      const __m256i swap = _mm256_setr_epi8(
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
          21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
      first = _mm256_shuffle_epi8(first, swap);
      second = _mm256_shuffle_epi8(second, swap);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), first);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 16), second);
#else
    // TODO(verify): portable scalar fallback.
    alignas(32) std::uint8_t buf[32];
    storeu8(this->value, buf);
    for (int i = 0; i < 32; i++) {
      char16_t c = char16_t(buf[i]);
      if (big_endian == endianness::BIG) {
        c = char16_t((c >> 8) | (c << 8));
      }
      ptr[i] = c;
    }
#endif
  }

  simdutf_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
#if SIMDUTF_IS_X86_64
    const __m256i self = to_m256i(this->value);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr),
                        _mm256_cvtepu8_epi32(_mm256_castsi256_si128(self)));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(ptr + 8),
        _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_srli_si256(self, 8))));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 16),
                        _mm256_cvtepu8_epi32(_mm256_extractf128_si256(self, 1)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 24),
                        _mm256_cvtepu8_epi32(_mm_srli_si128(
                            _mm256_extractf128_si256(self, 1), 8)));
#else
    // TODO(verify): portable scalar fallback.
    alignas(32) std::uint8_t buf[32];
    storeu8(this->value, buf);
    for (int i = 0; i < 32; i++) {
      ptr[i] = char32_t(buf[i]);
    }
#endif
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
    // Lane-crossing alignr: ESCAPE HATCH.
#if SIMDUTF_IS_X86_64
    return from_m256i(_mm256_alignr_epi8(
        to_m256i(this->value),
        _mm256_permute2x128_si256(to_m256i(prev_chunk.value),
                                  to_m256i(this->value), 0x21),
        16 - N));
#else
    // TODO(verify): portable scalar fallback for lane-crossing alignr.
    alignas(32) std::uint8_t cur[32];
    alignas(32) std::uint8_t prv[32];
    alignas(32) std::uint8_t out[32];
    storeu8(this->value, cur);
    storeu8(prev_chunk.value, prv);
    for (int i = 0; i < 32; i++) {
      int idx = i - N;
      out[i] = (idx < 0) ? prv[idx + 32] : cur[idx];
    }
    return simd8<T>(loadu8(out));
#endif
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

  simdutf_really_inline uint32_t to_bitmask() const {
    // movemask: ESCAPE HATCH.
#if SIMDUTF_IS_X86_64
    return uint32_t(_mm256_movemask_epi8(to_m256i(this->value)));
#else
    // TODO(verify): portable scalar fallback for movemask.
    alignas(32) std::uint8_t buf[32];
    storeu8(this->value, buf);
    uint32_t r = 0;
    for (int i = 0; i < 32; i++) {
      r |= (uint32_t(buf[i] >> 7) & 1u) << i;
    }
    return r;
#endif
  }
};

template <typename T> struct base8_numeric : base8<T> {
  static simdutf_really_inline simd8<T> splat(T _value) {
    return v8(std::uint8_t(_value));
  }
  static simdutf_really_inline simd8<T> zero() { return v8{}; }
  static simdutf_really_inline simd8<T> load(const T values[32]) {
    return loadu8(reinterpret_cast<const std::uint8_t *>(values));
  }
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  static simdutf_really_inline simd8<T> repeat_16(T v0, T v1, T v2, T v3, T v4,
                                                  T v5, T v6, T v7, T v8, T v9,
                                                  T v10, T v11, T v12, T v13,
                                                  T v14, T v15) {
    return simd8<T>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                    v14, v15, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                    v12, v13, v14, v15);
  }

  simdutf_really_inline base8_numeric() : base8<T>() {}
  simdutf_really_inline base8_numeric(const v8 _value) : base8<T>(_value) {}

  // Store to array
  simdutf_really_inline void store(T dst[32]) const {
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
#if SIMDUTF_IS_X86_64
    return simd8<L>(from_m256i(_mm256_shuffle_epi8(
        to_m256i(lookup_table.value), to_m256i(this->value))));
#else
    // TODO(verify): portable scalar fallback for per-128-lane pshufb.
    alignas(32) std::uint8_t tbl[32];
    alignas(32) std::uint8_t idx[32];
    alignas(32) std::uint8_t out[32];
    storeu8(lookup_table.value, tbl);
    storeu8(this->value, idx);
    for (int lane = 0; lane < 2; lane++) {
      const int base_off = lane * 16;
      for (int i = 0; i < 16; i++) {
        std::uint8_t b = idx[base_off + i];
        out[base_off + i] = (b & 0x80) ? 0 : tbl[base_off + (b & 0x0F)];
      }
    }
    return simd8<L>(loadu8(out));
#endif
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
  simdutf_really_inline simd8(const int8_t values[32]) : simd8(load(values)) {}
  simdutf_really_inline operator simd8<uint8_t>() const;

  simdutf_really_inline bool is_ascii() const {
    // high bit clear in every lane
    const ss::vec<std::int8_t, 32> sv = std::bit_cast<ss::vec<std::int8_t, 32>>(
        this->value);
    return ss::none_of(sv < ss::vec<std::int8_t, 32>(0));
  }
  // Order-sensitive comparisons
  simdutf_really_inline simd8<bool> operator>(const simd8<int8_t> other) const {
    const ss::vec<std::int8_t, 32> a =
        std::bit_cast<ss::vec<std::int8_t, 32>>(this->value);
    const ss::vec<std::int8_t, 32> b =
        std::bit_cast<ss::vec<std::int8_t, 32>>(other.value);
    const m8 m = std::bit_cast<m8>(a > b);
    return simd8<bool>(ss::select(m, v8(std::uint8_t(0xFF)), v8(std::uint8_t(0))));
  }
  simdutf_really_inline simd8<bool> operator<(const simd8<int8_t> other) const {
    const ss::vec<std::int8_t, 32> a =
        std::bit_cast<ss::vec<std::int8_t, 32>>(this->value);
    const ss::vec<std::int8_t, 32> b =
        std::bit_cast<ss::vec<std::int8_t, 32>>(other.value);
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
  simdutf_really_inline simd8(const uint8_t values[32]) : simd8(load(values)) {}
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
#if SIMDUTF_IS_X86_64
    return from_m256i(
        _mm256_subs_epu8(to_m256i(this->value), to_m256i(other.value)));
#else
    // TODO(verify): portable saturating subtract.
    const m8 ge = (this->value >= other.value);
    return v8(ss::select(ge, this->value - other.value, v8(std::uint8_t(0))));
#endif
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
    const ss::vec<std::int8_t, 32> sv =
        std::bit_cast<ss::vec<std::int8_t, 32>>(this->value);
    return ss::none_of(sv < ss::vec<std::int8_t, 32>(0));
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
#if SIMDUTF_IS_X86_64
    const __m256i tmp =
        _mm256_sad_epu8(to_m256i(this->value), _mm256_setzero_si256());
    return _mm256_extract_epi64(tmp, 0) + _mm256_extract_epi64(tmp, 1) +
           _mm256_extract_epi64(tmp, 2) + _mm256_extract_epi64(tmp, 3);
#else
    // TODO(verify): portable horizontal byte sum.
    ss::vec<std::uint64_t, 4> acc{};
    alignas(32) std::uint8_t buf[32];
    storeu8(this->value, buf);
    std::uint64_t s = 0;
    for (int i = 0; i < 32; i++)
      s += buf[i];
    return s;
#endif
  }
};
simdutf_really_inline simd8<int8_t>::operator simd8<uint8_t>() const {
  return this->value;
}

template <typename T> struct simd8x64 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd8<T>);
  static_assert(NUM_CHUNKS == 2,
                "stdsimd kernel should use two registers per 64-byte block.");
  simd8<T> chunks[NUM_CHUNKS];

  simd8x64(const simd8x64<T> &o) = delete; // no copy allowed
  simd8x64<T> &
  operator=(const simd8<T> other) = delete; // no assignment allowed
  simd8x64() = delete;                      // no default constructor allowed

  simdutf_really_inline simd8x64(const simd8<T> chunk0, const simd8<T> chunk1)
      : chunks{chunk0, chunk1} {}
  simdutf_really_inline simd8x64(const T *ptr)
      : chunks{simd8<T>::load(ptr),
               simd8<T>::load(ptr + sizeof(simd8<T>) / sizeof(T))} {}

  simdutf_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd8<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd8<T>) * 1 / sizeof(T));
  }

  simdutf_really_inline uint64_t to_bitmask() const {
    uint64_t r_lo = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r_hi = this->chunks[1].to_bitmask();
    return r_lo | (r_hi << 32);
  }

  simdutf_really_inline simd8x64<T> &operator|=(const simd8x64<T> &other) {
    this->chunks[0] |= other.chunks[0];
    this->chunks[1] |= other.chunks[1];
    return *this;
  }

  simdutf_really_inline simd8<T> reduce_or() const {
    return this->chunks[0] | this->chunks[1];
  }

  simdutf_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  template <endianness endian>
  simdutf_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 0);
    this->chunks[1].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 1);
  }

  simdutf_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    this->chunks[0].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 0);
    this->chunks[1].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 1);
  }

  simdutf_really_inline uint64_t in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);

    return simd8x64<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low))
        .to_bitmask();
  }

  simdutf_really_inline uint64_t lt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] < mask, this->chunks[1] < mask)
        .to_bitmask();
  }

  simdutf_really_inline uint64_t gt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] > mask, this->chunks[1] > mask)
        .to_bitmask();
  }
  simdutf_really_inline uint64_t eq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] == mask, this->chunks[1] == mask)
        .to_bitmask();
  }
  simdutf_really_inline uint64_t gteq_unsigned(const uint8_t m) const {
    const simd8<uint8_t> mask = simd8<uint8_t>::splat(m);
    return simd8x64<bool>((simd8<uint8_t>(this->chunks[0].value) >= mask),
                          (simd8<uint8_t>(this->chunks[1].value) >= mask))
        .to_bitmask();
  }
}; // struct simd8x64<T>

#include "simdutf/stdsimd/simd16-inl.h"
#include "simdutf/stdsimd/simd32-inl.h"
#include "simdutf/stdsimd/simd64-inl.h"

simdutf_really_inline simd64<uint64_t> sum_8bytes(const simd8<uint8_t> v) {
#if SIMDUTF_IS_X86_64
  return simd64<uint64_t>(std::bit_cast<v64>(
      _mm256_sad_epu8(to_m256i(v.value), _mm256_setzero_si256())));
#else
  // TODO(verify): portable 8-byte horizontal sums.
  alignas(32) std::uint8_t buf[32];
  storeu8(v.value, buf);
  alignas(32) std::uint64_t out[4] = {0, 0, 0, 0};
  for (int g = 0; g < 4; g++) {
    std::uint64_t s = 0;
    for (int i = 0; i < 8; i++)
      s += buf[g * 8 + i];
    out[g] = s;
  }
  return simd64<uint64_t>(out);
#endif
}

} // namespace simd

} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#endif // SIMDUTF_STDSIMD_SIMD_H
