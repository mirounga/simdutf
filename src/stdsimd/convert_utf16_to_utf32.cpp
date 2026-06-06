// ==== stdsimd utf16 -> utf32 conversion (tier-selected arch kernel) ====
//
// The utf16->utf32 bulk transcoder is a width-specific intrinsic kernel
// (cmpeq / movemask surrogate detection + zero-extending 16->32 widening) with
// no portable std::simd form and no backend-agnostic generic header. Rather
// than carry a copy, we reuse the existing arch kernels per tier and alias them
// to a uniform name:
//   * SSE tier (128-bit):        westmere/sse_convert_utf16_to_utf32.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_utf16_to_utf32.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too; a native
//      512-bit icelake-style kernel would be a further optimization.)
//
// Both arch kernels are pure x86 intrinsics + scalar::utf16::swap_if_needed /
// result / error_code -- self-contained, no internal:: or simd-library helper
// dependencies -- so they run on every tier.
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf16_utf32.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf16_to_utf32.cpp"
  #define stdsimd_convert_utf16_to_utf32 avx2_convert_utf16_to_utf32
  #define stdsimd_convert_utf16_to_utf32_with_errors                           \
    avx2_convert_utf16_to_utf32_with_errors
#else
  #include "westmere/sse_convert_utf16_to_utf32.cpp"
  #define stdsimd_convert_utf16_to_utf32 sse_convert_utf16_to_utf32
  #define stdsimd_convert_utf16_to_utf32_with_errors                           \
    sse_convert_utf16_to_utf32_with_errors
#endif
