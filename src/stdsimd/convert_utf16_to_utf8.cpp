// ==== stdsimd utf16 -> utf8 conversion (tier-selected arch kernel) ====
//
// The utf16->utf8 bulk transcoder is a width-specific intrinsic kernel (pshufb /
// movemask / table-driven byte compression) with no portable std::simd form and
// no backend-agnostic generic header. Rather than carry a copy, we reuse the
// existing arch kernels per tier and alias them to a uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_utf16_to_utf8.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_utf16_to_utf8.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too; a native
//      512-bit icelake-style kernel would be a further optimization.)
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf8_utf16.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf16_to_utf8.cpp"
  #define stdsimd_convert_utf16_to_utf8 avx2_convert_utf16_to_utf8
  #define stdsimd_convert_utf16_to_utf8_with_errors                            \
    avx2_convert_utf16_to_utf8_with_errors
#else
  // westmere's sse kernel calls internal::westmere::write_v_u16_11bits_to_utf8;
  // loader.cpp self-wraps it in namespace internal::westmere. Guard so multiple
  // westmere-using kernels in this TU don't redefine it.
  #ifndef SIMDUTF_STDSIMD_WESTMERE_INTERNAL_INCLUDED
    #define SIMDUTF_STDSIMD_WESTMERE_INTERNAL_INCLUDED 1
    #include "westmere/internal/loader.cpp"
  #endif
  #include "westmere/sse_convert_utf16_to_utf8.cpp"
  #define stdsimd_convert_utf16_to_utf8 sse_convert_utf16_to_utf8
  #define stdsimd_convert_utf16_to_utf8_with_errors                            \
    sse_convert_utf16_to_utf8_with_errors
#endif
