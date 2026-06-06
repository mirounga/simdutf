// ==== stdsimd utf16fix (to_well_formed_utf16) -- tier-selected arch kernel ====
//
// utf16fix is a width-specific intrinsic kernel (cross-lane byte shifts,
// movemask / testz, blendv) with no portable std::simd form. There is a generic
// header (src/generic/utf16/to_well_formed.h) but it relies on simd16<> wrapper
// members the stdsimd wrapper does not expose (byte_right_shift<>, first(),
// is_zero(), as_vector_u16()). Rather than carry a copy, we reuse the existing
// arch kernels per tier and alias them to a uniform name:
//   * SSE tier (128-bit):        westmere/sse_utf16fix.cpp  (utf16fix_sse)
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_utf16fix.cpp  (utf16fix_avx)
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too.)
//
// Both arch kernels are self-contained (raw __m128i/__m256i, no internal::
// helpers), so no westmere loader include is required.
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf16fix.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_utf16fix.cpp"
  #define stdsimd_utf16fix utf16fix_avx
#else
  #include "westmere/sse_utf16fix.cpp"
  #define stdsimd_utf16fix utf16fix_sse
#endif
