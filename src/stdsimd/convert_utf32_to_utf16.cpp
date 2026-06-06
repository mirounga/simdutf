// ==== stdsimd utf32 -> utf16 conversion (tier-selected arch kernel) ====
//
// The utf32->utf16 bulk transcoder is a width-specific intrinsic kernel
// (testz / packus_epi32 32->16 narrowing + surrogate-pair expansion) with no
// portable std::simd form and no backend-agnostic generic header. Rather than
// carry a copy, we reuse the existing arch kernels per tier and alias them to a
// uniform name:
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_convert_utf32_to_utf16.cpp
//     (AVX-512 implies AVX2; pure x86 intrinsics, self-contained.)
//   * SSE tier (128-bit):        NOT tier-selected here. westmere's
//     sse_convert_utf32_to_utf16.cpp drives its surrogate-pair expansion
//     through the FULL westmere simd:: library (simd8/simd32 with
//     as_vector_u8 / lookup_16 / to_4bit_bitmask / select), which the stdsimd
//     backend's simd:: namespace does not provide. Pulling in westmere's
//     simd.h would collide with stdsimd's own simd:: in the same namespace, so
//     the SSE tier keeps utf32->utf16 on the scalar path (see the impl
//     partial). The other three directions in this family (utf16->utf32, both
//     endiannesses) DO run as real SIMD on the SSE tier.
//
//     A native 128-bit pure-intrinsic kernel (inlining the shuffle/pack tables
//     instead of the westmere simd:: wrappers) would be a further optimization.
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf16_utf32.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf32_to_utf16.cpp"
  #define stdsimd_convert_utf32_to_utf16 avx2_convert_utf32_to_utf16
  #define stdsimd_convert_utf32_to_utf16_with_errors                           \
    avx2_convert_utf32_to_utf16_with_errors
#endif
