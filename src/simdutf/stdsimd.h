#ifndef SIMDUTF_STDSIMD_H
#define SIMDUTF_STDSIMD_H

#ifdef SIMDUTF_FALLBACK_H
  #error "stdsimd.h must be included before fallback.h"
#endif

#include "simdutf/portability.h"

// The stdsimd backend is ADDITIVE and FORCE-only: it never auto-shadows a tuned
// backend. It is enabled only when the CMake option turns it on AND the toolchain
// actually provides C++26 <simd> on GCC 16+. Otherwise it compiles out entirely.
//
// The CMake build defines SIMDUTF_IMPLEMENTATION_STDSIMD=1 when the option is ON.
// We additionally guard on the real availability of the feature so that a stray
// define cannot produce a broken translation unit.
#ifndef SIMDUTF_IMPLEMENTATION_STDSIMD
  #define SIMDUTF_IMPLEMENTATION_STDSIMD 0
#endif

#if SIMDUTF_IMPLEMENTATION_STDSIMD
  #if !((defined(__GNUC__) && (__GNUC__ >= 16)) &&                             \
        (defined(__cplusplus) && (__cplusplus > 202302L)) &&                  \
        (defined(__glibcxx_simd) || __has_include(<simd>)))
    // The toolchain cannot actually provide the C++26 std::simd backend; turn
    // it back off rather than emit a broken translation unit.
    #undef SIMDUTF_IMPLEMENTATION_STDSIMD
    #define SIMDUTF_IMPLEMENTATION_STDSIMD 0
  #endif
#endif

// The stdsimd backend is reachable only via SIMDUTF_FORCE_IMPLEMENTATION=stdsimd.
// It must never be selected by runtime auto-detection, so it can never "always
// run" — it is registered below all tuned backends.
#define SIMDUTF_CAN_ALWAYS_RUN_STDSIMD 0

#if SIMDUTF_IMPLEMENTATION_STDSIMD

  // The stdsimd backend compiles at the same AVX2 tier as haswell: it applies
  // the identical target region ("avx2,bmi,lzcnt,popcnt").
  #define SIMDUTF_TARGET_STDSIMD SIMDUTF_TARGET_REGION("avx2,bmi,lzcnt,popcnt")

namespace simdutf {
/**
 * Implementation for the C++26 std::simd backend (GCC 16+), compiled at the
 * AVX2 tier. FORCE-only.
 */
namespace stdsimd {} // namespace stdsimd
} // namespace simdutf

  //
  // These two need to be included outside SIMDUTF_TARGET_REGION
  //
  #include "simdutf/stdsimd/implementation.h"
  #include "simdutf/stdsimd/intrinsics.h"

  //
  // The rest need to be inside the region
  //
  #include "simdutf/stdsimd/begin.h"
  // Declarations
  #include "simdutf/stdsimd/bitmanipulation.h"
  #include "simdutf/stdsimd/simd.h"

  #include "simdutf/stdsimd/end.h"

#endif // SIMDUTF_IMPLEMENTATION_STDSIMD
#endif // SIMDUTF_STDSIMD_H
