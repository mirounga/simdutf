// ==== stdsimd utf32 -> utf8 conversion (adapted from haswell) ====
//
// There is no backend-agnostic generic header for the AVX2 utf32->utf8 path,
// so this file is adapted from src/haswell/avx2_convert_utf32_to_utf8.cpp.
//
// Per the stdsimd escape-hatch policy, the operations used here have no
// portable std::simd form: the algorithm packs UTF-32 -> UTF-16 with unsigned
// saturation (packus), then runs the UTF-16 -> UTF-8 byte-compression path,
// which is driven by pshufb with per-128-lane shuffles selected from
// tables::utf16_to_utf8::pack_1_2{,_3}_utf8_bytes, plus movemask / maddubs /
// testz. None of these map cleanly onto std::simd primitives, and the
// algorithm never leaves raw __m256i, so -- exactly like the slice-1
// convert_utf16_to_utf8.cpp kernel -- we PRAGMATICALLY keep the x86 AVX2
// intrinsics. They compile and run under the AVX2 target region applied by
// stdsimd/begin.h and the -mavx2 command-line baseline (verified equivalent to
// haswell's). The vec<->__m256i bridge is not required because the algorithm
// operates only on raw __m256i.
//
// TODO(verify): an ARM (vqtbl / vsh) and a scalar fallback arm could be added
// for non-x86 targets; only the x86 arm is exercised for the AVX2 tier.
//
// This file is #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION by
// stdsimd/impl_utf8_utf32.inc.cpp.

#if SIMDUTF_IS_X86_64

// The kernel below is byte-for-byte the haswell AVX2 implementation: it depends
// only on <immintrin.h> intrinsics and simdutf::tables::utf16_to_utf8 (no
// haswell simd:: wrapper types), so it is backend agnostic and reused VERBATIM
// as the escape hatch, mirroring how the slice-1 utf8->utf16/utf32 masked
// transcoders are reused.
  #include "haswell/avx2_convert_utf32_to_utf8.cpp"

#else
  #error "stdsimd convert_utf32_to_utf8: only the x86_64 AVX2 arm is implemented"
#endif // SIMDUTF_IS_X86_64
