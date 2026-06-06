// ==== stdsimd utf16 -> latin1 conversion (tier-selected arch kernel) ====
//
// The utf16->latin1 bulk transcoder is a width-specific intrinsic kernel
// (testz, packus narrowing, lane-crossing permute and byte swap) with no
// portable std::simd form and no backend-agnostic generic header. Rather than
// carry a copy, we reuse the existing arch kernels per tier and alias them to a
// uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_utf16_to_latin1.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_utf16_to_latin1.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too.)
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_latin1.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf16_to_latin1.cpp"
  #define stdsimd_convert_utf16_to_latin1 avx2_convert_utf16_to_latin1
  #define stdsimd_convert_utf16_to_latin1_with_errors                           \
    avx2_convert_utf16_to_latin1_with_errors
#else
  #include "westmere/sse_convert_utf16_to_latin1.cpp"
  #define stdsimd_convert_utf16_to_latin1 sse_convert_utf16_to_latin1
  #define stdsimd_convert_utf16_to_latin1_with_errors                           \
    sse_convert_utf16_to_latin1_with_errors
#endif
