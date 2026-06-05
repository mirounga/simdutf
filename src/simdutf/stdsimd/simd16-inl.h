// part of simd.h (included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::
// (anonymous)::simd). std::simd wrapper presenting haswell/simd16-inl.h's
// interface over pinned-width vec<uint16_t,16>.

using v16 = ss::vec<std::uint16_t, 16>; // 16 elements == 32 bytes
using m16 = ss::mask<std::uint16_t, 16>;

#if SIMDUTF_IS_X86_64
simdutf_really_inline __m256i to_m256i(const v16 v) {
  return std::bit_cast<__m256i>(v);
}
simdutf_really_inline v16 v16_from_m256i(const __m256i r) {
  return std::bit_cast<v16>(r);
}
#endif

simdutf_really_inline v16 loadu16(const std::uint16_t *ptr) {
  return ss::unchecked_load<v16>(ptr, 16, ss::flag_default);
}
simdutf_really_inline void storeu16(const v16 v, std::uint16_t *ptr) {
  ss::unchecked_store(v, ptr, 16, ss::flag_default);
}

template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>>
struct base16 : base<simd16<T>> {
  using bitmask_type = uint32_t;

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
#if SIMDUTF_IS_X86_64
    return _mm256_movemask_epi8(to_m256i(this->v()));
#else
    // TODO(verify): portable scalar fallback.
    alignas(32) std::uint8_t buf[32];
    storeu8(this->value, buf);
    uint32_t r = 0;
    for (int i = 0; i < 32; i++) {
      r |= (uint32_t(buf[i] >> 7) & 1u) << i;
    }
    return r;
#endif
  }

  simdutf_really_inline simd16<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base16_numeric : base16<T> {
  static simdutf_really_inline simd16<T> splat(T _value) {
    return v16(std::uint16_t(_value));
  }

  static simdutf_really_inline simd16<T> zero() { return v16{}; }

  static simdutf_really_inline simd16<T> load(const T values[8]) {
    return loadu16(reinterpret_cast<const std::uint16_t *>(values));
  }

  simdutf_really_inline base16_numeric() : base16<T>() {}

  simdutf_really_inline base16_numeric(const v16 _value) : base16<T>(_value) {}

  // Store to array
  simdutf_really_inline void store(T dst[8]) const {
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
#if SIMDUTF_IS_X86_64
    const __m256i swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
        21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
    return v16_from_m256i(_mm256_shuffle_epi8(to_m256i(this->v()), swap));
#else
    // TODO(verify): portable byteswap.
    return v16((this->v() << 8) | (this->v() >> 8));
#endif
  }

  // Pack with the unsigned saturation of two uint16_t code units into single
  // uint8_t vector: ESCAPE HATCH (lane shuffle + packus).
  static simdutf_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
#if SIMDUTF_IS_X86_64
    const __m128i lo_0 = _mm256_extracti128_si256(to_m256i(v0.v()), 0);
    const __m128i lo_1 = _mm256_extracti128_si256(to_m256i(v1.v()), 0);
    const __m128i hi_0 = _mm256_extracti128_si256(to_m256i(v0.v()), 1);
    const __m128i hi_1 = _mm256_extracti128_si256(to_m256i(v1.v()), 1);
    const __m256i t0 =
        _mm256_permute2f128_si256(_mm256_castsi128_si256(lo_0),
                                  _mm256_castsi128_si256(lo_1), 0x20);
    const __m256i t1 =
        _mm256_permute2f128_si256(_mm256_castsi128_si256(hi_0),
                                  _mm256_castsi128_si256(hi_1), 0x20);
    return simd8<uint8_t>(from_m256i(_mm256_packus_epi16(t0, t1)));
#else
    // TODO(verify): portable saturating pack to bytes.
    alignas(32) std::uint16_t a[16];
    alignas(32) std::uint16_t b[16];
    alignas(32) std::uint8_t out[32];
    storeu16(v0.v(), a);
    storeu16(v1.v(), b);
    for (int i = 0; i < 16; i++)
      out[i] = std::uint8_t(a[i] > 0xFF ? 0xFF : a[i]);
    for (int i = 0; i < 16; i++)
      out[16 + i] = std::uint8_t(b[i] > 0xFF ? 0xFF : b[i]);
    return simd8<uint8_t>(loadu8(out));
#endif
  }

  simdutf_really_inline uint64_t sum() const {
    ss::vec<std::uint32_t, 8> acc{};
    // widen-and-sum portably
    alignas(32) std::uint16_t buf[16];
    storeu16(this->v(), buf);
    std::uint64_t s = 0;
    for (int i = 0; i < 16; i++)
      s += buf[i];
    return s;
  }
};

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(NUM_CHUNKS == 2,
                "stdsimd kernel should use two registers per 64-byte block.");
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  simdutf_really_inline simd16x32(const simd16<T> chunk0,
                                  const simd16<T> chunk1)
      : chunks{chunk0, chunk1} {}
  simdutf_really_inline simd16x32(const T *ptr)
      : chunks{simd16<T>::load(ptr),
               simd16<T>::load(ptr + sizeof(simd16<T>) / sizeof(T))} {}

  simdutf_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd16<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd16<T>) * 1 / sizeof(T));
  }

  simdutf_really_inline uint64_t to_bitmask() const {
    uint64_t r_lo = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r_hi = this->chunks[1].to_bitmask();
    return r_lo | (r_hi << 32);
  }

  simdutf_really_inline simd16<T> reduce_or() const {
    return this->chunks[0] | this->chunks[1];
  }

  simdutf_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  simdutf_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 0);
    this->chunks[1].store_ascii_as_utf16(ptr + sizeof(simd16<T>));
  }

  simdutf_really_inline void swap_bytes() {
    this->chunks[0] = this->chunks[0].swap_bytes();
    this->chunks[1] = this->chunks[1].swap_bytes();
  }
  simdutf_really_inline uint64_t gt(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] > mask, this->chunks[1] > mask)
        .to_bitmask();
  }

  simdutf_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask)
        .to_bitmask();
  }
  simdutf_really_inline uint64_t eq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] == mask, this->chunks[1] == mask)
        .to_bitmask();
  }
  simdutf_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(static_cast<T>(low - 1));
    const simd16<T> mask_high = simd16<T>::splat(static_cast<T>(high + 1));
    return simd16x32<bool>(
               (this->chunks[0] >= mask_high) | (this->chunks[0] <= mask_low),
               (this->chunks[1] >= mask_high) | (this->chunks[1] <= mask_low))
        .to_bitmask();
  }
}; // struct simd16x32<T>

simd16<uint16_t> min(const simd16<uint16_t> a, simd16<uint16_t> b) {
  return v16(ss::min(a.v(), b.v()));
}
