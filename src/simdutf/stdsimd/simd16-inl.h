// part of simd.h (included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::
// (anonymous)::simd). std::simd wrapper presenting haswell/simd16-inl.h's
// interface over pinned-width vec<uint16_t,16>.

// v16 / m16 (= vec<uint16_t, BYTES/2>) are defined in tier_ops.h. Number of
// u16 lanes the vector holds:
static constexpr int V16_LANES = SIMDUTF_STDSIMD_VEC_BYTES / 2;

simdutf_really_inline v16 loadu16(const std::uint16_t *ptr) {
  return ss::unchecked_load<v16>(ptr, V16_LANES, ss::flag_default);
}
simdutf_really_inline void storeu16(const v16 v, std::uint16_t *ptr) {
  ss::unchecked_store(v, ptr, V16_LANES, ss::flag_default);
}

template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>>
struct base16 : base<simd16<T>> {
  // A single simd16<bool> spans SIMDUTF_STDSIMD_VEC_BYTES bytes, so its
  // byte-movemask has that many bits: 16 (SSE) / 32 (AVX2) fit in uint32_t, but
  // the AVX512 tier (64 bytes) needs all 64 bits. Using uint64_t for every tier
  // avoids truncating the AVX512 single-chunk movemask (which made
  // simd16x32::to_bitmask drop half its bits).
  using bitmask_type = uint64_t;

  simdutf_really_inline base16() : base<simd16<T>>() {}
  // construct from the 8-bit-pinned storage of base<> via re-bitcast
  simdutf_really_inline base16(const v16 _value)
      : base<simd16<T>>(std::bit_cast<v8>(_value)) {}
  template <typename Pointer>
  simdutf_really_inline base16(const Pointer *ptr)
      : base16(loadu16(reinterpret_cast<const std::uint16_t *>(ptr))) {}

  // the underlying 16-bit view of the stored bytes
  simdutf_really_inline v16 v() const {
    return std::bit_cast<v16>(this->value);
  }

  friend simdutf_always_inline Mask operator==(const simd16<T> lhs,
                                               const simd16<T> rhs) {
    const m16 m = (lhs.v() == rhs.v());
    return Mask(ss::select(m, v16(std::uint16_t(0xFFFF)), v16(std::uint16_t(0))));
  }

  /// the size of vector in bytes
  static const int SIZE = sizeof(v8);

  /// the number of elements of type T a vector can hold
  static const int ELEMENTS = SIZE / sizeof(T);
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd16<bool> : base16<bool> {
  static simdutf_really_inline simd16<bool> splat(bool _value) {
    return v16(std::uint16_t(-(!!_value)));
  }

  simdutf_really_inline simd16() : base16() {}

  simdutf_really_inline simd16(const v16 _value) : base16<bool>(_value) {}

  simdutf_really_inline simd16(bool _value) : base16<bool>(splat(_value)) {}

  simdutf_really_inline bitmask_type to_bitmask() const {
    // movemask over bytes: ESCAPE HATCH.
    return bitmask_type(tier_movemask8(this->value));
  }

  simdutf_really_inline simd16<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base16_numeric : base16<T> {
  static simdutf_really_inline simd16<T> splat(T _value) {
    return v16(std::uint16_t(_value));
  }

  static simdutf_really_inline simd16<T> zero() { return v16{}; }

  static simdutf_really_inline simd16<T> load(const T values[V16_LANES]) {
    return loadu16(reinterpret_cast<const std::uint16_t *>(values));
  }

  simdutf_really_inline base16_numeric() : base16<T>() {}

  simdutf_really_inline base16_numeric(const v16 _value) : base16<T>(_value) {}

  // Store to array
  simdutf_really_inline void store(T dst[V16_LANES]) const {
    storeu16(this->v(), reinterpret_cast<std::uint16_t *>(dst));
  }

  // Override to distinguish from bool version
  simdutf_really_inline simd16<T> operator~() const { return *this ^ 0xFFFFu; }

  // Addition/subtraction are the same for signed and unsigned (wrapping)
  simdutf_really_inline simd16<T> operator+(const simd16<T> other) const {
    return v16(this->v() + other.v());
  }
  simdutf_really_inline simd16<T> &operator+=(const simd16<T> other) {
    *this = *this + other;
    return *static_cast<simd16<T> *>(this);
  }
};

// Unsigned code units
template <> struct simd16<uint16_t> : base16_numeric<uint16_t> {
  simdutf_really_inline simd16() : base16_numeric<uint16_t>() {}
  simdutf_really_inline simd16(const v16 _value)
      : base16_numeric<uint16_t>(_value) {}
  // From a boolean mask: per-lane 0xFFFF/0x0000 (mirrors haswell where
  // simd16<uint16_t> accepts the __m256i mask). Needed by
  // generic/utf16/utf8_length_from_utf16_bytemask.h.
  simdutf_really_inline simd16(const simd16<bool> other)
      : base16_numeric<uint16_t>(other.v()) {}

  // Splat constructor
  simdutf_really_inline simd16(uint16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  simdutf_really_inline simd16(const uint16_t *values) : simd16(load(values)) {}
  simdutf_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const uint16_t *>(values))) {}

  // Order-specific operations
  simdutf_really_inline simd16<uint16_t>
  max_val(const simd16<uint16_t> other) const {
    return v16(ss::max(this->v(), other.v()));
  }
  simdutf_really_inline simd16<uint16_t>
  min_val(const simd16<uint16_t> other) const {
    return v16(ss::min(this->v(), other.v()));
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  simdutf_really_inline simd16<bool>
  operator<=(const simd16<uint16_t> other) const {
    return other.max_val(*this) == other;
  }
  simdutf_really_inline simd16<bool>
  operator>=(const simd16<uint16_t> other) const {
    return other.min_val(*this) == other;
  }
  simdutf_really_inline simd16<bool>
  operator>(const simd16<uint16_t> other) const {
    const m16 m = (this->v() > other.v());
    return simd16<bool>(
        ss::select(m, v16(std::uint16_t(0xFFFF)), v16(std::uint16_t(0))));
  }

  // Bit-specific operations
  simdutf_really_inline simd16<bool> bits_not_set() const {
    return *this == uint16_t(0);
  }

  simdutf_really_inline simd16<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }

  template <int N> simdutf_really_inline simd16<uint16_t> shr() const {
    return v16(this->v() >> N);
  }

  // Change the endianness: ESCAPE HATCH (byte shuffle).
  simdutf_really_inline simd16<uint16_t> swap_bytes() const {
    return simd16<uint16_t>(tier_byteswap16(this->v()));
  }

  // Pack with the unsigned saturation of two uint16_t code units into single
  // uint8_t vector: ESCAPE HATCH (lane shuffle + packus).
  static simdutf_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
    return simd8<uint8_t>(tier_pack16(v0.v(), v1.v()));
  }

  simdutf_really_inline uint64_t sum() const {
    std::uint16_t buf[V16_LANES];
    storeu16(this->v(), buf);
    std::uint64_t s = 0;
    for (int i = 0; i < V16_LANES; i++)
      s += buf[i];
    return s;
  }
};

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(NUM_CHUNKS == 64 / SIMDUTF_STDSIMD_VEC_BYTES,
                "simd16x32 must tile a 64-byte block with the pinned width.");
  // Each chunk contributes a byte-movemask of its underlying byte storage.
  static constexpr int CHUNK_BITS = SIMDUTF_STDSIMD_VEC_BYTES;
  static constexpr uint64_t CHUNK_MASK =
      (CHUNK_BITS >= 64) ? ~uint64_t(0) : ((uint64_t(1) << CHUNK_BITS) - 1);
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  // Per-chunk constructor: exactly NUM_CHUNKS chunks (2 on AVX2, 4 on SSE,
  // 1 on AVX512). Constrained to simd16<T> args so it never competes with the
  // pointer constructor.
  template <typename... Chunks>
    requires(sizeof...(Chunks) == NUM_CHUNKS &&
             (std::is_same_v<Chunks, simd16<T>> && ...))
  simdutf_really_inline simd16x32(const Chunks... cs) : chunks{cs...} {}
  simdutf_really_inline simd16x32(const T *ptr) {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      chunks[i] = simd16<T>::load(ptr + i * sizeof(simd16<T>) / sizeof(T));
    }
  }

  simdutf_really_inline void store(T *ptr) const {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i].store(ptr + i * sizeof(simd16<T>) / sizeof(T));
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

  simdutf_really_inline simd16<T> reduce_or() const {
    simd16<T> r = this->chunks[0];
    for (int i = 1; i < NUM_CHUNKS; i++) {
      r = r | this->chunks[i];
    }
    return r;
  }

  simdutf_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  simdutf_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * i);
    }
  }

  simdutf_really_inline void swap_bytes() {
    for (int i = 0; i < NUM_CHUNKS; i++) {
      this->chunks[i] = this->chunks[i].swap_bytes();
    }
  }

  // Build a 64-bit bitmask by applying a per-chunk predicate returning a
  // simd16<bool>; each chunk contributes CHUNK_BITS bits of byte-movemask.
  template <typename Pred>
  simdutf_really_inline uint64_t bitmask_of(Pred pred) const {
    uint64_t r = 0;
    for (int i = 0; i < NUM_CHUNKS; i++) {
      r |= (uint64_t(pred(this->chunks[i]).to_bitmask()) & CHUNK_MASK)
           << (CHUNK_BITS * i);
    }
    return r;
  }

  simdutf_really_inline uint64_t gt(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return bitmask_of([&](const simd16<T> c) { return c > mask; });
  }

  simdutf_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return bitmask_of([&](const simd16<T> c) { return c <= mask; });
  }
  simdutf_really_inline uint64_t eq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return bitmask_of([&](const simd16<T> c) { return c == mask; });
  }
  simdutf_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(static_cast<T>(low - 1));
    const simd16<T> mask_high = simd16<T>::splat(static_cast<T>(high + 1));
    return bitmask_of([&](const simd16<T> c) {
      return (c >= mask_high) | (c <= mask_low);
    });
  }
}; // struct simd16x32<T>

simd16<uint16_t> min(const simd16<uint16_t> a, simd16<uint16_t> b) {
  return v16(ss::min(a.v(), b.v()));
}
