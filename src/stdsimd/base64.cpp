// ==== base64 (stdsimd backend) -- tier-selected arch kernel ====
//
// The base64 SIMD encode/decode family is a width-specific intrinsic kernel
// dominated by operations that have NO portable std::simd form: dynamic byte
// shuffle (pshufb / *_shuffle_epi8), saturating subtract (*_subs_epu8),
// fixed-point multishift (*_mulhi_epu16 / *_mullo_epi16 / *_maddubs_epi16 /
// *_madd_epi16), movemask, lane-crossing alignr/permute and the thintable-driven
// byte compress. Rather than carry a copy, we reuse the existing per-arch
// kernels per tier and alias their entry points to uniform stdsimd_ names:
//   * SSE tier (128-bit):        westmere/sse_base64.cpp
//   * AVX2/AVX512 tiers (256b):  haswell/avx2_base64.cpp
//     (AVX-512 implies AVX2, so the 256-bit kernel runs there too.)
//
// Both arch kernels define a `block64` class consumed VERBATIM by the
// backend-agnostic decode driver generic/base64.h (compress_decode_base64), an
// `encode_base64<bool>` and an encode-with-lines impl, plus (haswell only) a
// free `avx2_binary_length_from_base64`. They reference only tables::base64::*
// and scalar::base64::*, both visible at the inclusion scope -- no internal::
// helpers, so (unlike utf16->utf8) no westmere loader.cpp is required here.
//
// This file is #included inside namespace
//   simdutf::SIMDUTF_IMPLEMENTATION::{anon}
// by stdsimd/impl_base64.inc.cpp (same placement haswell uses), and inside the
// per-tier target region applied by stdsimd/begin.h.
//
// References and further reading:
//   Wojciech Muła, Daniel Lemire, "Base64 encoding and decoding at almost the
//   speed of a memory copy", Software: Practice and Experience 50 (2), 2020,
//   https://arxiv.org/abs/1910.05109
//   Wojciech Muła, Daniel Lemire, "Faster Base64 Encoding and Decoding using
//   AVX2 Instructions", ACM Trans. Web 12 (3), 2018,
//   https://arxiv.org/abs/1704.00605

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_base64.cpp"
  // encode_base64<bool> and the block64 class already carry uniform names.
  #define stdsimd_encode_base64_impl avx2_encode_base64_impl
  #define stdsimd_binary_length_from_base64 avx2_binary_length_from_base64
#else
  #include "westmere/sse_base64.cpp"
  // westmere names the with-lines encoder encode_base64_impl (no avx2_ prefix);
  // westmere has no free binary_length helper -- the impl partial pulls in
  // generic/base64lengths.h at file scope and routes there on this tier.
  #define stdsimd_encode_base64_impl encode_base64_impl
#endif
