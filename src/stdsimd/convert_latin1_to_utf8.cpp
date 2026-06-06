// ==== stdsimd latin1 -> utf8 conversion (tier-selected arch kernel) ====
//
// The latin1->utf8 bulk transcoder is a width-specific intrinsic kernel
// (cvtepu8 widening / movemask / table-driven byte compression) with no
// portable std::simd form and no backend-agnostic generic header. Rather than
// carry a copy, we reuse the existing arch kernels per tier and alias them to a
// uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_latin1_to_utf8.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_latin1_to_utf8.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too.)
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_latin1.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_latin1_to_utf8.cpp"
  #define stdsimd_convert_latin1_to_utf8 avx2_convert_latin1_to_utf8
#else
  // westmere's sse kernel calls internal::westmere::write_v_u16_11bits_to_utf8;
  // loader.cpp self-wraps it in namespace internal::westmere. Guard so multiple
  // westmere-using kernels in this TU don't redefine it.
  #ifndef SIMDUTF_STDSIMD_WESTMERE_INTERNAL_INCLUDED
    #define SIMDUTF_STDSIMD_WESTMERE_INTERNAL_INCLUDED 1
    #include "westmere/internal/loader.cpp"
  #endif
  #include "westmere/sse_convert_latin1_to_utf8.cpp"
  #define stdsimd_convert_latin1_to_utf8 sse_convert_latin1_to_utf8
#endif
