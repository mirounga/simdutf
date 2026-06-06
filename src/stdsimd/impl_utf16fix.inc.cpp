// ==== utf16fix family (REAL PORT -- all tiers) ====
//
// This partial holds the to_well_formed_utf16 methods for the stdsimd backend.
// It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/westmere implementation.cpp): utf16fix is
// a width-specific intrinsic kernel that never touches the arch simd:: types, so
// -- like stdsimd/convert_utf16_to_utf8.cpp -- the existing arch kernels are
// reused as x86 escape-hatch kernels with NO wrapper dependency. The tier-select
// wrapper (stdsimd/utf16fix.cpp) picks westmere's 128-bit kernel on the SSE tier
// and haswell's 256-bit kernel on the AVX2/AVX512 tiers, aliasing the entry to
// stdsimd_utf16fix. It is emitted in the backend's anonymous namespace so the
// implementation methods below can call it by unqualified name.

#if SIMDUTF_FEATURE_UTF16

namespace {
// utf16fix bulk kernel (tier-selected arch kernel: westmere SSE 128-bit on the
// SSE tier, haswell AVX2 256-bit on AVX2/AVX512). Self-tier-selecting.
  #include "stdsimd/utf16fix.cpp"
} // unnamed namespace

void implementation::to_well_formed_utf16le(const char16_t *input, size_t len,
                                            char16_t *output) const noexcept {
  return stdsimd_utf16fix<endianness::LITTLE>(input, len, output);
}

void implementation::to_well_formed_utf16be(const char16_t *input, size_t len,
                                            char16_t *output) const noexcept {
  return stdsimd_utf16fix<endianness::BIG>(input, len, output);
}
#endif // SIMDUTF_FEATURE_UTF16
