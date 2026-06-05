// ==== stdsimd latin1 -> utf32 conversion (adapted from haswell) ====
//
// There is no backend-agnostic generic header for the AVX2 latin1->utf32 path,
// so this file is adapted from src/haswell/avx2_convert_latin1_to_utf32.cpp.
//
// Per the stdsimd escape-hatch policy, the operation used here has no portable
// std::simd form (cvtepu8_epi32 8->32-bit widening), so we PRAGMATICALLY keep
// the x86 AVX2 intrinsics. The haswell kernel depends only on <immintrin.h>
// (no haswell simd:: types), so it is backend agnostic and is reused VERBATIM
// here. The vec<->__m256i bridge is not required because the algorithm never
// leaves raw __m256i.
//
// This file is #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION by
// stdsimd/impl_latin1.inc.cpp.

#include "haswell/avx2_convert_latin1_to_utf32.cpp"
