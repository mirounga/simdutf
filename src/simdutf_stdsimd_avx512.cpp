// Dedicated translation unit for the AVX512 tier of the C++26 std::simd backend
// (namespace simdutf::stdsimd_avx512, name "stdsimd_avx512").
//
// Re-includes the amalgamation with SIMDUTF_STDSIMD_ONLY=1, emitting ONLY the
// stdsimd implementation for THIS tier. SIMDUTF_STDSIMD_VEC_BYTES selects the
// std::simd vec width (64 => 512-bit) and, in stdsimd.h, the namespace, name,
// target region and ISA gate. CMake compiles this TU with an
// -mavx512f,bw,cd,dq,vl,vbmi -mbmi -mbmi2 command-line baseline so std::simd's
// vec<uint8_t,64> uses the ZMM register ABI rather than the no-AVX memory ABI
// the target-pragma alone leaves in place. See [[stdsimd-validate-perf]].
#define SIMDUTF_STDSIMD_ONLY 1
#define SIMDUTF_STDSIMD_VEC_BYTES 64
#include "simdutf.cpp"
