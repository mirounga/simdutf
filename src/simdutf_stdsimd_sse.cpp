// Dedicated translation unit for the SSE tier of the C++26 std::simd backend
// (namespace simdutf::stdsimd_sse, name "stdsimd_sse").
//
// Re-includes the amalgamation with SIMDUTF_STDSIMD_ONLY=1, emitting ONLY the
// stdsimd implementation for THIS tier. SIMDUTF_STDSIMD_VEC_BYTES selects the
// std::simd vec width (16 => 128-bit) and, in stdsimd.h, the namespace, name,
// target region and ISA gate. CMake compiles this TU with an -msse4.2
// command-line baseline so std::simd's vec<uint8_t,16> uses the XMM register
// ABI rather than the no-SIMD memory ABI the target-pragma alone leaves in
// place. See [[stdsimd-validate-perf]].
#define SIMDUTF_STDSIMD_ONLY 1
#define SIMDUTF_STDSIMD_VEC_BYTES 16
#include "simdutf.cpp"
