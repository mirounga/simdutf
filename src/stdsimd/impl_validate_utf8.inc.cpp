// ==== validate_utf8 (REAL PORT) ====
//
// This partial holds the validate_utf8 family for the stdsimd backend. It is
// #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace stdsimd { ... } }
// and inside the AVX2 target region.
//
// REAL PORT: this family reuses src/generic/utf8_validation/* VERBATIM, exactly
// like haswell/implementation.cpp. The generic algorithm headers self-wrap in
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ... } } }
// so they must be pulled in at FILE scope. Because this partial is textually
// included while simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly
// CLOSE those two namespaces, do the file-scope work (the two unnamed-namespace
// helpers the generic checker depends on -- is_ascii() and
// must_be_2_3_continuation() -- followed by the generic headers themselves),
// then REOPEN the namespaces so the rest of implementation.cpp continues
// unchanged. The namespace close/reopen is UNCONDITIONAL so the parent file's
// brace balance never depends on a feature macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the generic headers (which open
// ---- their own simdutf::SIMDUTF_IMPLEMENTATION) nest at file scope. ---------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if (SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING) &&                \
    SIMDUTF_STDSIMD_UTF8_LOOKUP4

// The generic UTF-8 checker calls two helpers by unqualified name from inside
// the backend's anonymous namespace:
//   * is_ascii(const simd::simd8x64<uint8_t> &)
//   * must_be_2_3_continuation(simd::simd8<uint8_t>, simd::simd8<uint8_t>)
// haswell defines them in its own anonymous namespace; we mirror that here so
// the verbatim reuse compiles. They are emitted BEFORE the generic headers.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;

simdutf_really_inline bool is_ascii(const simd8x64<uint8_t> &input) {
  return input.reduce_or().is_ascii();
}

simdutf_really_inline simd8<bool>
must_be_2_3_continuation(const simd8<uint8_t> prev2,
                         const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0xe0u - 0x80); // Only 111_____ will be > 0x80
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0xf0u - 0x80); // Only 1111____ will be > 0x80
  return simd8<bool>(is_third_byte | is_fourth_byte);
}
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// buf_block_reader.h has no include guard and is meant to be pulled in exactly
// once per backend translation unit. Guard with a TU-local sentinel so other
// family partials (utf8_utf16, base64) can include it too without colliding.
  #ifndef SIMDUTF_STDSIMD_BUF_BLOCK_READER_INCLUDED
    #define SIMDUTF_STDSIMD_BUF_BLOCK_READER_INCLUDED 1
    #include "generic/buf_block_reader.h"
  #endif

  #include "generic/utf8_validation/utf8_lookup4_algorithm.h"
  #include "generic/utf8_validation/utf8_validator.h"

// clang-format off
#endif // (SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING) && SIMDUTF_STDSIMD_UTF8_LOOKUP4
// clang-format on

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ---------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_STDSIMD_UTF8_LOOKUP4
// ===== SSE / AVX2 tiers: real generic SIMD UTF-8 validator. ==================

#if SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING
simdutf_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return SIMDUTF_IMPLEMENTATION::utf8_validation::generic_validate_utf8(buf,
                                                                        len);
}
#endif // SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING

#if SIMDUTF_FEATURE_UTF8
simdutf_warn_unused result implementation::validate_utf8_with_errors(
    const char *buf, size_t len) const noexcept {
  return SIMDUTF_IMPLEMENTATION::utf8_validation::
      generic_validate_utf8_with_errors(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF8

#else // !SIMDUTF_STDSIMD_UTF8_LOOKUP4
// ===== AVX512 tier (single 64-byte chunk): the generic lookup4 validator
// ===== requires 2 or 4 chunks, so delegate validate_utf8 to scalar. ==========

#if SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING
simdutf_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return scalar::utf8::validate(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF8 || SIMDUTF_FEATURE_DETECT_ENCODING

#if SIMDUTF_FEATURE_UTF8
simdutf_warn_unused result implementation::validate_utf8_with_errors(
    const char *buf, size_t len) const noexcept {
  return scalar::utf8::validate_with_errors(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF8

#endif // SIMDUTF_STDSIMD_UTF8_LOOKUP4
