// ==== utf16fix family (REAL PORT) ====
//
// This partial holds the to_well_formed_utf16 methods for the stdsimd backend.
// It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp, which includes
// haswell/avx2_utf16fix.cpp and calls utf16fix_avx<>): the AVX2 kernel operates
// entirely on raw __m256i and never uses the haswell simd:: types, so -- like
// stdsimd/convert_utf16_to_utf8.cpp -- it is reused as an x86 escape-hatch
// kernel with NO wrapper dependency. The adapted kernel lives in
// stdsimd/utf16fix.cpp; it is emitted in the backend's anonymous namespace
// (exactly like haswell emits avx2_utf16fix.cpp) so the implementation methods
// below can call utf16fix_avx() by unqualified name. The #if SIMDUTF_FEATURE_*
// guards mirror the originals.

#if SIMDUTF_FEATURE_UTF16

namespace {
// utf16fix bulk kernel (adapted from haswell, x86 intrinsic escape hatches kept;
// portable scalar fallback for non-x86).
  #include "stdsimd/utf16fix.cpp"
} // unnamed namespace

void implementation::to_well_formed_utf16le(const char16_t *input, size_t len,
                                            char16_t *output) const noexcept {
  return utf16fix_avx<endianness::LITTLE>(input, len, output);
}

void implementation::to_well_formed_utf16be(const char16_t *input, size_t len,
                                            char16_t *output) const noexcept {
  return utf16fix_avx<endianness::BIG>(input, len, output);
}
#endif // SIMDUTF_FEATURE_UTF16
