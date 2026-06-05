// ==== validate_utf32 family (REAL PORT) ====
//
// This partial holds the UTF-32 validation methods for the stdsimd backend. It
// is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region.
//
// REAL PORT: this family reuses src/generic/validate_utf32.h VERBATIM, exactly
// like haswell/implementation.cpp (which wires the same header for these two
// methods). The generic algorithm header self-wraps in
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ... } } }
// so it must be pulled in at FILE scope. Because this partial is textually
// included while simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly
// CLOSE those two namespaces, include the generic header at file scope, then
// REOPEN the namespaces so the rest of implementation.cpp continues unchanged.
// The namespace close/reopen is UNCONDITIONAL so the parent file's brace
// balance never depends on a feature macro.
//
// generic/validate_utf32.h drives everything through simd32<uint32_t> /
// simd32<bool> from the stdsimd wrapper (splat/zero/swap_bytes/max/operator+/
// operator&/operator==/operator>=/operator>/any), plus the global match_system
// helper and scalar::utf32::validate* for the tail. No backend-local helpers
// are required, so unlike validate_utf8 there is no anonymous-namespace prelude.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the generic header (which opens
// ---- its own simdutf::SIMDUTF_IMPLEMENTATION) nests at file scope. ----------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if SIMDUTF_FEATURE_UTF32 || SIMDUTF_FEATURE_DETECT_ENCODING
  #include "generic/validate_utf32.h"
#endif // SIMDUTF_FEATURE_UTF32 || SIMDUTF_FEATURE_DETECT_ENCODING

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ---------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF32 || SIMDUTF_FEATURE_DETECT_ENCODING
simdutf_warn_unused bool
implementation::validate_utf32(const char32_t *buf, size_t len) const noexcept {
  return utf32::validate(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF32 || SIMDUTF_FEATURE_DETECT_ENCODING

#if SIMDUTF_FEATURE_UTF32
simdutf_warn_unused result implementation::validate_utf32_with_errors(
    const char32_t *buf, size_t len) const noexcept {
  return utf32::validate_with_errors(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF32
