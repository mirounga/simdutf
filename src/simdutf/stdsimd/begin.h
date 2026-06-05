// The active tier's namespace token (stdsimd_sse / stdsimd_avx2 /
// stdsimd_avx512) is derived from SIMDUTF_STDSIMD_VEC_BYTES in stdsimd.h.
#define SIMDUTF_IMPLEMENTATION SIMDUTF_STDSIMD_NS
#define SIMDUTF_SIMD_HAS_BYTEMASK 1

// Each stdsimd tier applies its own function-target region (set in stdsimd.h as
// SIMDUTF_TARGET_STDSIMD). The std::simd widths are pinned explicitly
// (vec<uint8_t,BYTES>) so they are independent of the command-line ISA, but the
// escape-hatch intrinsics still need the region.
#if SIMDUTF_CAN_ALWAYS_RUN_STDSIMD
// nothing needed.
#else
SIMDUTF_TARGET_STDSIMD
#endif

#if SIMDUTF_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
// clang-format off
SIMDUTF_DISABLE_GCC_WARNING(-Wmaybe-uninitialized)
// clang-format on
#endif // end of workaround
