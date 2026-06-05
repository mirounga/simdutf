// Dedicated translation unit for the C++26 std::simd backend.
//
// This re-includes the amalgamation with SIMDUTF_STDSIMD_ONLY=1, which emits
// ONLY the stdsimd implementation (the dispatcher, every other backend and the
// C API stay in the primary simdutf.cpp TU). The CMake build compiles THIS file
// with an AVX2 *command-line* baseline (-mavx2 -mbmi -mbmi2 -mlzcnt) so that
// std::simd's vec<uint8_t,32> uses the YMM register calling convention. Without
// it, the per-function target("avx2") pragma enables AVX inside function bodies
// but leaves the vector type's ABI at the no-AVX baseline, forcing every vec
// across a boundary (and under register pressure) through memory -- which made
// validate_utf8 ~16x slower than haswell. See memory: stdsimd-validate-perf.
#define SIMDUTF_STDSIMD_ONLY 1
#include "simdutf.cpp"
