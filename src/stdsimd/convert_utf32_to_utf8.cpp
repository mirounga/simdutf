// ==== stdsimd utf32 -> utf8 conversion (tier-selected arch kernel) ====
//
// The utf32->utf8 bulk transcoder is a width-specific intrinsic kernel (packs
// UTF-32 -> UTF-16 with unsigned saturation, then runs the table-driven
// pshufb / movemask / maddubs byte-compression path) with no portable std::simd
// form and no backend-agnostic generic header. Rather than carry a copy, we
// reuse the existing arch kernels per tier and alias them to a uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_utf32_to_utf8.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_utf32_to_utf8.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too; a native
//      512-bit icelake-style kernel would be a further optimization.)
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf8_utf32.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf32_to_utf8.cpp"
  #define stdsimd_convert_utf32_to_utf8 avx2_convert_utf32_to_utf8
  #define stdsimd_convert_utf32_to_utf8_with_errors                              \
    avx2_convert_utf32_to_utf8_with_errors
#else
  // westmere's sse utf32->utf8 kernel is self-contained (only pshufb/SSE
  // intrinsics + simdutf::tables::utf16_to_utf8); it references no
  // internal::westmere helper, so no loader.cpp include is needed here.
  #include "westmere/sse_convert_utf32_to_utf8.cpp"
  #define stdsimd_convert_utf32_to_utf8 sse_convert_utf32_to_utf8
  #define stdsimd_convert_utf32_to_utf8_with_errors                              \
    sse_convert_utf32_to_utf8_with_errors
#endif
