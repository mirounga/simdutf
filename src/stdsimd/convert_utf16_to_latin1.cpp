// ==== stdsimd utf16 -> latin1 conversion (adapted from haswell) ====
//
// There is no backend-agnostic generic header for the AVX2 utf16->latin1 path,
// so this file is adapted from src/haswell/avx2_convert_utf16_to_latin1.cpp.
//
// Per the stdsimd escape-hatch policy, the operations used here have no
// portable std::simd form (testz, packus narrowing, lane-crossing permute and
// byte swap), so we PRAGMATICALLY keep the x86 AVX2 intrinsics. The haswell
// kernel depends only on <immintrin.h> and scalar:: helpers (no haswell simd::
// types), so it is backend agnostic and is reused VERBATIM here. The
// vec<->__m256i bridge is not required because the algorithm never leaves raw
// __m256i.
//
// This file is #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION by
// stdsimd/impl_latin1.inc.cpp.

#include "haswell/avx2_convert_utf16_to_latin1.cpp"
