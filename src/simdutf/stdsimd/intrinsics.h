#ifndef SIMDUTF_STDSIMD_INTRINSICS_H
#define SIMDUTF_STDSIMD_INTRINSICS_H

#include "simdutf.h"

// The portable C++26 std::simd backend. We compile at the AVX2 tier (see
// begin.h) and present the EXACT same wrapper interface as the haswell backend
// so that src/generic/* algorithm headers can be reused verbatim.
//
// Default policy: portable std::simd. A small number of operations have no
// portable form and use guarded intrinsic "escape hatches" bridged through
// std::bit_cast (see simd.h). Those escape hatches require the platform
// intrinsics headers below.

#include <simd>

#include <bit>     // std::bit_cast
#include <cstddef> // std::size_t
#include <cstdint>

#if SIMDUTF_IS_X86_64
  #ifdef SIMDUTF_VISUAL_STUDIO
    #include <intrin.h> // visual studio or clang
  #else

    #if SIMDUTF_GCC11ORMORE
// We should not get warnings while including <x86intrin.h> yet we do
// under some versions of GCC.
SIMDUTF_DISABLE_GCC_WARNING(-Wuninitialized)
    #endif

    #include <x86intrin.h> // for the AVX2 escape hatches

    #if SIMDUTF_GCC11ORMORE
SIMDUTF_POP_DISABLE_WARNINGS
    #endif

  #endif // SIMDUTF_VISUAL_STUDIO
#elif SIMDUTF_IS_ARM64
  // TODO(verify): the ARM escape-hatch arms (vqtbl/vsh) are not exercised yet.
  #include <arm_neon.h>
#endif

#endif // SIMDUTF_STDSIMD_INTRINSICS_H
