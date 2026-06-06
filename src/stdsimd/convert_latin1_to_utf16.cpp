// ==== stdsimd latin1 -> utf16 conversion (tier-selected arch kernel) ====
//
// The latin1->utf16 bulk transcoder is a width-specific intrinsic kernel
// (cvtepu8_epi16 widening, lane-crossing byte swap) with no portable std::simd
// form and no backend-agnostic generic header. Rather than carry a copy, we
// reuse the existing arch kernels per tier and alias them to a uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_latin1_to_utf16.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_latin1_to_utf16.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too.)
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_latin1.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_latin1_to_utf16.cpp"
  #define stdsimd_convert_latin1_to_utf16 avx2_convert_latin1_to_utf16
#else
  #include "westmere/sse_convert_latin1_to_utf16.cpp"
  #define stdsimd_convert_latin1_to_utf16 sse_convert_latin1_to_utf16
#endif
