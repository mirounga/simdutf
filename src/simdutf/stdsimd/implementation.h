#ifndef SIMDUTF_STDSIMD_IMPLEMENTATION_H
#define SIMDUTF_STDSIMD_IMPLEMENTATION_H

#include "simdutf/implementation.h"

// The C++26 std::simd backend ships as THREE runtime-selectable tiers, each in
// its own namespace and translation unit:
//   - stdsimd_sse    : 128-bit std::simd, SSE4.2     ISA gate
//   - stdsimd_avx2   : 256-bit std::simd, AVX2|BMI1|BMI2
//   - stdsimd_avx512 : 512-bit std::simd, AVX512 F/BW/CD/DQ/VL/VBMI(/VBMI2)
//
// The implementation class declaration is identical for every tier except for
// the namespace, the name() string and the ISA bitmask passed to the base
// constructor. We therefore declare ALL THREE classes here by re-including the
// parameterized body (implementation_body.inc.h) once per tier with the right
// macros, so that whichever TU pulls this header sees every tier's declaration
// (the registration in implementation.cpp needs all three). The constructor may
// run on any host, so this header must stay OUTSIDE any SIMDUTF_TARGET_REGION.

// ---- SSE tier (stdsimd_sse) ------------------------------------------------
#define SIMDUTF_STDSIMD_BODY_NS stdsimd_sse
#define SIMDUTF_STDSIMD_BODY_NAME "stdsimd_sse"
#define SIMDUTF_STDSIMD_BODY_ISA internal::instruction_set::SSE42
#include "simdutf/stdsimd/implementation_body.inc.h"
#undef SIMDUTF_STDSIMD_BODY_NS
#undef SIMDUTF_STDSIMD_BODY_NAME
#undef SIMDUTF_STDSIMD_BODY_ISA

// ---- AVX2 tier (stdsimd_avx2) ----------------------------------------------
#define SIMDUTF_STDSIMD_BODY_NS stdsimd_avx2
#define SIMDUTF_STDSIMD_BODY_NAME "stdsimd_avx2"
#define SIMDUTF_STDSIMD_BODY_ISA                                               \
  (internal::instruction_set::AVX2 | internal::instruction_set::BMI1 |         \
   internal::instruction_set::BMI2)
#include "simdutf/stdsimd/implementation_body.inc.h"
#undef SIMDUTF_STDSIMD_BODY_NS
#undef SIMDUTF_STDSIMD_BODY_NAME
#undef SIMDUTF_STDSIMD_BODY_ISA

// ---- AVX512 tier (stdsimd_avx512) ------------------------------------------
#define SIMDUTF_STDSIMD_BODY_NS stdsimd_avx512
#define SIMDUTF_STDSIMD_BODY_NAME "stdsimd_avx512"
// Mirror the icelake gate (same AVX-512 instruction family: vbmi/vbmi2
// permute/multishift/compress). The enum has no separate AVX512VBMI bit, so we
// gate on VBMI2 (which on every shipping uarch implies VBMI) plus the icelake
// set. AVX512F is implied by BW/CD/DQ/VL but listed for clarity where present.
#define SIMDUTF_STDSIMD_BODY_ISA                                               \
  (internal::instruction_set::AVX2 | internal::instruction_set::BMI1 |         \
   internal::instruction_set::BMI2 | internal::instruction_set::AVX512F |      \
   internal::instruction_set::AVX512BW | internal::instruction_set::AVX512CD | \
   internal::instruction_set::AVX512DQ | internal::instruction_set::AVX512VL | \
   internal::instruction_set::AVX512VBMI2 |                                    \
   internal::instruction_set::AVX512VPOPCNTDQ)
#include "simdutf/stdsimd/implementation_body.inc.h"
#undef SIMDUTF_STDSIMD_BODY_NS
#undef SIMDUTF_STDSIMD_BODY_NAME
#undef SIMDUTF_STDSIMD_BODY_ISA

#endif // SIMDUTF_STDSIMD_IMPLEMENTATION_H
