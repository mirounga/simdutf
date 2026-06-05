// part of simd.h. std::simd wrapper presenting haswell/simd32-inl.h's interface
// over pinned-width vec<uint32_t,8>.

using v32 = ss::vec<std::uint32_t, 8>; // 8 elements == 32 bytes
using m32 = ss::mask<std::uint32_t, 8>;

simdutf_really_inline v32 loadu32(const std::uint32_t *ptr) {
  return ss::unchecked_load<v32>(ptr, 8, ss::flag_default);
}

template <typename T> struct simd32;

template <> struct simd32<uint32_t> {
  static const size_t SIZE = sizeof(v8);
  static const size_t ELEMENTS = SIZE / sizeof(uint32_t);

  v32 value;

  simdutf_really_inline simd32(const v32 v) : value(v) {}

  template <typename Pointer>
  simdutf_really_inline simd32(const Pointer *ptr)
      : value(loadu32(reinterpret_cast<const std::uint32_t *>(ptr))) {}

  simdutf_really_inline uint64_t sum() const {
    alignas(32) std::uint32_t buf[8];
    ss::unchecked_store(value, buf, 8, ss::flag_default);
    std::uint64_t s = 0;
    for (int i = 0; i < 8; i++)
      s += buf[i];
    return s;
  }

  // Change the endianness: ESCAPE HATCH (byte shuffle).
  simdutf_really_inline simd32<uint32_t> swap_bytes() const {
#if SIMDUTF_IS_X86_64
    const __m256i shuffle =
        _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12,
                         3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12);
    return std::bit_cast<v32>(
        _mm256_shuffle_epi8(std::bit_cast<__m256i>(value), shuffle));
#else
    // TODO(verify): portable 32-bit byteswap.
    return v32(((value & v32(0x000000FFu)) << 24) |
               ((value & v32(0x0000FF00u)) << 8) |
               ((value & v32(0x00FF0000u)) >> 8) |
               ((value & v32(0xFF000000u)) >> 24));
#endif
  }

  // operators
  simdutf_really_inline simd32 &operator+=(const simd32 other) {
    value = value + other.value;
    return *this;
  }

  // static members
  simdutf_really_inline static simd32<uint32_t> zero() { return v32{}; }

  simdutf_really_inline static simd32<uint32_t> splat(uint32_t v) {
    return v32(v);
  }
};

//----------------------------------------------------------------------

template <> struct simd32<bool> {
  v32 value;

  simdutf_really_inline simd32(const v32 v) : value(v) {}

  simdutf_really_inline bool any() const { return ss::any_of(value != v32{}); }
};

//----------------------------------------------------------------------

template <typename T>
simdutf_really_inline simd32<T> operator|(const simd32<T> a,
                                          const simd32<T> b) {
  return v32(a.value | b.value);
}

simdutf_really_inline simd32<uint32_t> min(const simd32<uint32_t> b,
                                           const simd32<uint32_t> a) {
  return v32(ss::min(a.value, b.value));
}

simdutf_really_inline simd32<uint32_t> max(const simd32<uint32_t> a,
                                           const simd32<uint32_t> b) {
  return v32(ss::max(a.value, b.value));
}

simdutf_really_inline simd32<uint32_t> operator&(const simd32<uint32_t> b,
                                                 const simd32<uint32_t> a) {
  return v32(a.value & b.value);
}

simdutf_really_inline simd32<uint32_t> operator+(const simd32<uint32_t> a,
                                                 const simd32<uint32_t> b) {
  return v32(a.value + b.value);
}

simdutf_really_inline simd32<bool> operator==(const simd32<uint32_t> a,
                                              const simd32<uint32_t> b) {
  const m32 m = (a.value == b.value);
  return simd32<bool>(ss::select(m, v32(0xFFFFFFFFu), v32(0u)));
}

simdutf_really_inline simd32<bool> operator>=(const simd32<uint32_t> a,
                                              const simd32<uint32_t> b) {
  const m32 m = (a.value >= b.value);
  return simd32<bool>(ss::select(m, v32(0xFFFFFFFFu), v32(0u)));
}

simdutf_really_inline simd32<bool> operator!(const simd32<bool> v) {
  return v32(~v.value);
}

simdutf_really_inline simd32<bool> operator>(const simd32<uint32_t> a,
                                             const simd32<uint32_t> b) {
  return !(b >= a);
}
