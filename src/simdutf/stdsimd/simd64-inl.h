// part of simd.h. std::simd wrapper presenting haswell/simd64-inl.h's interface
// over pinned-width vec<uint64_t,4>.

// v64 (= vec<uint64_t, BYTES/8>) is defined in tier_ops.h.
static constexpr int V64_LANES = SIMDUTF_STDSIMD_VEC_BYTES / 8;

simdutf_really_inline v64 loadu64(const std::uint64_t *ptr) {
  return ss::unchecked_load<v64>(ptr, V64_LANES, ss::flag_default);
}

template <typename T> struct simd64;

template <> struct simd64<uint64_t> {
  v64 value;

  simdutf_really_inline simd64(const v64 v) : value(v) {}

  template <typename Pointer>
  simdutf_really_inline simd64(const Pointer *ptr)
      : value(loadu64(reinterpret_cast<const std::uint64_t *>(ptr))) {}

  simdutf_really_inline uint64_t sum() const { return ss::reduce(value); }

  // operators
  simdutf_really_inline simd64 &operator+=(const simd64 other) {
    value = value + other.value;
    return *this;
  }

  // static members
  simdutf_really_inline static simd64<uint64_t> zero() { return v64{}; }

  simdutf_really_inline static simd64<uint64_t> splat(uint64_t v) {
    return v64(v);
  }
};
