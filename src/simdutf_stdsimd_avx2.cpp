// Dedicated translation unit for the AVX2 tier of the C++26 std::simd backend
// (namespace simdutf::stdsimd_avx2, name "stdsimd_avx2").
//
// Re-includes the amalgamation with SIMDUTF_STDSIMD_ONLY=1, emitting ONLY the
// stdsimd implementation for THIS tier. SIMDUTF_STDSIMD_VEC_BYTES selects the
// std::simd vec width (32 => 256-bit) and, in stdsimd.h, the namespace, name,
// target region and ISA gate. CMake compiles this TU with an
// -mavx2 -mbmi -mbmi2 -mlzcnt command-line baseline so std::simd's
// vec<uint8_t,32> uses the YMM register ABI rather than the no-AVX memory ABI
// the target-pragma alone leaves in place (which made validate_utf8 ~16x
// slower). See [[stdsimd-validate-perf]].
#define SIMDUTF_STDSIMD_ONLY 1
#define SIMDUTF_STDSIMD_VEC_BYTES 32
#include "simdutf.cpp"
