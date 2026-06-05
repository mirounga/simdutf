// ==== validate_utf16 family (REAL PORT) ====
//
// This partial holds the UTF-16 validation methods for the stdsimd backend. It
// is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp):
//   * validate_utf16 reuses src/generic/validate_utf16.h VERBATIM, exactly like
//     haswell. That generic header self-wraps in
//       namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ...}}}
//     so it must be pulled in at FILE scope. It calls utf16_gather_high_bytes()
//     by unqualified name from inside its own
//       simdutf::SIMDUTF_IMPLEMENTATION::{anon}::utf16
//     namespace. haswell supplies that helper from
//     haswell/avx2_convert... no: from haswell/avx2_validate_utf16.cpp, which it
//     #includes inside `namespace { using namespace simd; namespace utf16 { ...
//     } }`. We mirror that: avx2_validate_utf16.cpp is pure wrapper-level code
//     (simd16<>::shr, ::pack, operator&) so it adapts to the stdsimd wrapper
//     VERBATIM -- no x86 escape hatch needed at this layer (pack itself carries
//     the only escape hatch, inside the wrapper header).
//
// Because this partial is textually included while
// simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly CLOSE those two
// namespaces, do the file-scope work (the anonymous-namespace helper source and
// the generic header), then REOPEN the namespaces so the rest of
// implementation.cpp continues unchanged. The namespace close/reopen is
// UNCONDITIONAL so the parent file's brace balance never depends on a feature
// macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the included sources (which open
// ---- their own simdutf::SIMDUTF_IMPLEMENTATION) nest at file scope. ----------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if SIMDUTF_FEATURE_UTF16 || SIMDUTF_FEATURE_DETECT_ENCODING

// --- anonymous-namespace helper (utf16_gather_high_bytes) --------------------
// The generic validate_utf16 algorithm calls utf16_gather_high_bytes() by
// unqualified name from inside simdutf::SIMDUTF_IMPLEMENTATION::{anon}::utf16.
// haswell defines it there via haswell/avx2_validate_utf16.cpp; we reuse that
// same source VERBATIM. It only touches wrapper-level ops (operator&, shr<8>,
// simd16<uint16_t>::pack), which the stdsimd wrapper provides, so no x86 escape
// hatch is required at this layer.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;
namespace utf16 {
  #include "haswell/avx2_validate_utf16.cpp"
} // namespace utf16
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// --- generic validate_utf16 algorithm header (reused VERBATIM) ---------------
// Self-opens simdutf::SIMDUTF_IMPLEMENTATION::{anon}::utf16 and calls
// utf16_gather_high_bytes() (defined just above) unqualified.
  #include "generic/validate_utf16.h"

#endif // SIMDUTF_FEATURE_UTF16 || SIMDUTF_FEATURE_DETECT_ENCODING

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF16 || SIMDUTF_FEATURE_DETECT_ENCODING
simdutf_warn_unused bool
implementation::validate_utf16le(const char16_t *buf,
                                 size_t len) const noexcept {
  if (simdutf_unlikely(len == 0)) {
    // empty input is valid UTF-16. protect the implementation from
    // handling nullptr
    return true;
  }
  const auto res =
      utf16::validate_utf16_with_errors<endianness::LITTLE>(buf, len);
  if (res.is_err()) {
    return false;
  }

  if (res.count == len) {
    return true;
  }

  return scalar::utf16::validate<endianness::LITTLE>(buf + res.count,
                                                     len - res.count);
}
#endif // SIMDUTF_FEATURE_UTF16 || SIMDUTF_FEATURE_DETECT_ENCODING

#if SIMDUTF_FEATURE_UTF16
simdutf_warn_unused bool
implementation::validate_utf16be(const char16_t *buf,
                                 size_t len) const noexcept {
  if (simdutf_unlikely(len == 0)) {
    // empty input is valid UTF-16. protect the implementation from
    // handling nullptr
    return true;
  }
  const auto res = utf16::validate_utf16_with_errors<endianness::BIG>(buf, len);
  if (res.is_err()) {
    return false;
  }

  if (res.count == len) {
    return true;
  }

  return scalar::utf16::validate<endianness::BIG>(buf + res.count,
                                                  len - res.count);
}

simdutf_warn_unused result implementation::validate_utf16le_with_errors(
    const char16_t *buf, size_t len) const noexcept {
  const result res =
      utf16::validate_utf16_with_errors<endianness::LITTLE>(buf, len);
  if (res.count != len) {
    const result scalar_res =
        scalar::utf16::validate_with_errors<endianness::LITTLE>(
            buf + res.count, len - res.count);
    return result(scalar_res.error, res.count + scalar_res.count);
  } else {
    return res;
  }
}

simdutf_warn_unused result implementation::validate_utf16be_with_errors(
    const char16_t *buf, size_t len) const noexcept {
  const result res =
      utf16::validate_utf16_with_errors<endianness::BIG>(buf, len);
  if (res.count != len) {
    const result scalar_res =
        scalar::utf16::validate_with_errors<endianness::BIG>(buf + res.count,
                                                             len - res.count);
    return result(scalar_res.error, res.count + scalar_res.count);
  } else {
    return res;
  }
}
#endif // SIMDUTF_FEATURE_UTF16
