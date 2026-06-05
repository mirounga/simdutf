#define SIMDUTF_IMPLEMENTATION stdsimd
#define SIMDUTF_SIMD_HAS_BYTEMASK 1

// The stdsimd backend compiles at the same AVX2 tier as haswell: it applies the
// identical target region ("avx2,bmi,lzcnt,popcnt"). The std::simd widths are
// pinned explicitly (vec<uint8_t,32> etc.) so they are independent of the
// command-line ISA, but the escape-hatch intrinsics still need the region.
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
