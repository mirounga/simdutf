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

// ---------------------------------------------------------------------------
// THREE runtime-selectable tiers. Which tier THIS translation unit emits is
// driven entirely by SIMDUTF_STDSIMD_VEC_BYTES (16/32/64), set on the command
// line by the dedicated per-tier TU (src/simdutf_stdsimd_{sse,avx2,avx512}.cpp)
// before it re-includes simdutf.cpp. We derive everything else from it:
//   - SIMDUTF_STDSIMD_TIER       : 1 (SSE) | 2 (AVX2) | 3 (AVX512)
//   - SIMDUTF_STDSIMD_NS         : namespace token (stdsimd_sse/_avx2/_avx512)
//   - SIMDUTF_STDSIMD_NAME       : backend name string used by name()
//   - SIMDUTF_TARGET_STDSIMD     : per-tier function target region
//   - SIMDUTF_STDSIMD_ISA        : per-tier required instruction-set bitmask
// When no width is set (e.g. the primary, non-stdsimd TU just including the
// header to see the declarations), default to the AVX2 tier so the class
// declarations stay valid.
// ---------------------------------------------------------------------------
#define SIMDUTF_STDSIMD_TIER_SSE 1
#define SIMDUTF_STDSIMD_TIER_AVX2 2
#define SIMDUTF_STDSIMD_TIER_AVX512 3

#ifndef SIMDUTF_STDSIMD_VEC_BYTES
  #define SIMDUTF_STDSIMD_VEC_BYTES 32
#endif

#ifndef SIMDUTF_STDSIMD_TIER
  #if SIMDUTF_STDSIMD_VEC_BYTES == 16
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_SSE
  #elif SIMDUTF_STDSIMD_VEC_BYTES == 32
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_AVX2
  #elif SIMDUTF_STDSIMD_VEC_BYTES == 64
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_AVX512
  #else
    #error "SIMDUTF_STDSIMD_VEC_BYTES must be 16, 32, or 64"
  #endif
#endif

#if SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_SSE
  #define SIMDUTF_STDSIMD_NS stdsimd_sse
  #define SIMDUTF_STDSIMD_NAME "stdsimd_sse"
#elif SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX2
  #define SIMDUTF_STDSIMD_NS stdsimd_avx2
  #define SIMDUTF_STDSIMD_NAME "stdsimd_avx2"
#elif SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX512
  #define SIMDUTF_STDSIMD_NS stdsimd_avx512
  #define SIMDUTF_STDSIMD_NAME "stdsimd_avx512"
#else
  #error "Unknown SIMDUTF_STDSIMD_TIER"
#endif

// Several conversion/validation families in the stdsimd backend reuse haswell's
// 256-bit AVX2 intrinsic kernels VERBATIM (avx2_convert_utf8_to_utf16/utf32,
// avx2_validate_utf16, avx2_convert_utf8_to_latin1) or carry hand-adapted
// 256-bit kernels (stdsimd/convert_utf16_to_utf8/_to_utf32, convert_utf32_to_*,
// utf16fix, the count_utf8 / utf32 surrogate-count loops, the SIMD base64
// family). Those 256-bit intrinsics physically require the AVX2 ISA, so they are
// built ONLY for the AVX2 tier. The SSE (128-bit) and AVX512 (512-bit) tiers
// route those specific families to the scalar reference (exactly like the
// fallback backend); a native 128-bit / 512-bit std::simd port of each kernel is
// future work. The genuinely tier-generic families that go through the std::simd
// wrapper alone (validate_utf8, validate_ascii, validate_utf32, find, detect)
// run as REAL std::simd on every tier.
#if SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX2
  #define SIMDUTF_STDSIMD_AVX2_KERNELS 1
#else
  #define SIMDUTF_STDSIMD_AVX2_KERNELS 0
#endif

// The generic UTF-8 validator (generic/utf8_validation/utf8_lookup4_algorithm.h)
// tiles each 64-byte block into 1, 2 or 4 chunks. The stdsimd wrapper's simd8x64
// has 64 / VEC_BYTES chunks: 4 (SSE), 2 (AVX2), 1 (AVX512). The validator now has
// a NUM_CHUNKS==1 branch, so the real generic checker runs on ALL tiers including
// the single-chunk AVX512 tier.
#define SIMDUTF_STDSIMD_UTF8_LOOKUP4 1

#if SIMDUTF_IMPLEMENTATION_STDSIMD

  // Per-tier function-target region for the intrinsic escape hatches. The
  // std::simd vec widths are pinned explicitly so they are independent of the
  // command-line ISA, but the escape-hatch intrinsics still need the region.
  #if SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_SSE
    #define SIMDUTF_TARGET_STDSIMD SIMDUTF_TARGET_REGION("sse4.2")
  #elif SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX2
    #define SIMDUTF_TARGET_STDSIMD                                              \
      SIMDUTF_TARGET_REGION("avx2,bmi,lzcnt,popcnt")
  #else // AVX512
    #define SIMDUTF_TARGET_STDSIMD                                              \
      SIMDUTF_TARGET_REGION("avx512f,avx512bw,avx512cd,avx512dq,avx512vl,"      \
                            "avx512vbmi,bmi,bmi2,lzcnt,popcnt")
  #endif

namespace simdutf {
/**
 * Implementation for the C++26 std::simd backend (GCC 16+). Three tiers exist,
 * one per namespace; each is compiled in its own translation unit. FORCE-only.
 */
namespace SIMDUTF_STDSIMD_NS {} // namespace SIMDUTF_STDSIMD_NS
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
