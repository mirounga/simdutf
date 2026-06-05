// ==== utf8 <-> utf32 conversions family (REAL PORT) ====
//
// This partial holds the UTF-8 <-> UTF-32 conversion family for the stdsimd
// backend. It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace stdsimd { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp and the slice-1
// impl_utf8_utf16.inc.cpp):
//   * utf8 -> utf32 reuses src/generic/utf8_to_utf32/* VERBATIM. Those generic
//     algorithm headers self-wrap in
//       namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ...}}}
//     so they must be pulled in at FILE scope. The generic converter calls the
//     AVX2-specific helper convert_masked_utf8_to_utf32() by unqualified name,
//     which has no portable std::simd form (it is a pshufb/table driven masked
//     transcoder), so we PRAGMATICALLY reuse haswell's pure-intrinsic
//     avx2_convert_utf8_to_utf32.cpp VERBATIM as the escape hatch. It depends
//     only on <immintrin.h> and tables/utf8_to_utf16_tables.h (no haswell simd::
//     types), so it is backend agnostic.
//   * utf32 -> utf8 has no backend-agnostic generic header on the AVX2 path, so
//     it uses the adapted stdsimd/convert_utf32_to_utf8.cpp (which keeps the x86
//     AVX2 intrinsics as escape hatches per the stdsimd policy).
//
// Because this partial is textually included while
// simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly CLOSE those two
// namespaces, do the file-scope work (the anonymous-namespace helper sources and
// the generic headers), then REOPEN the namespaces so the rest of
// implementation.cpp continues unchanged. The namespace close/reopen is
// UNCONDITIONAL so the parent file's brace balance never depends on a feature
// macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the included sources (which open
// ---- their own simdutf::SIMDUTF_IMPLEMENTATION) nest at file scope. ----------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

// --- anonymous-namespace sources (escape-hatch intrinsic kernels) ------------
// Both kernels keep the x86 AVX2 intrinsics: they have no portable std::simd
// form (table-driven byte compression / masked transcoding). They are emitted
// in the backend's anonymous namespace, exactly like haswell does, so the
// generic utf8_to_utf32 converter can call convert_masked_utf8_to_utf32() by
// unqualified name from the enclosing scope.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;

// utf8 -> utf32 per-block masked transcoder (pure intrinsics + tables).
  #include "haswell/avx2_convert_utf8_to_utf32.cpp"

// utf32 -> utf8 bulk transcoder (adapted from haswell, x86 intrinsic escape
// hatches kept; lives in stdsimd/).
  #include "stdsimd/convert_utf32_to_utf8.cpp"
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// --- generic utf8 -> utf32 algorithm headers (reused VERBATIM) ---------------
// These self-open simdutf::SIMDUTF_IMPLEMENTATION::{anon}::utf8_to_utf32 and
// call convert_masked_utf8_to_utf32() (defined just above) unqualified.
  #include "generic/utf8_to_utf32/valid_utf8_to_utf32.h"
  #include "generic/utf8_to_utf32/utf8_to_utf32.h"

#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

// ===========================================================================
// utf8 -> utf32  (reuses generic utf8_to_utf32:: VERBATIM, like haswell)
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf8_to_utf32(
    const char *buf, size_t len, char32_t *utf32_output) const noexcept {
  utf8_to_utf32::validating_transcoder converter;
  return converter.convert(buf, len, utf32_output);
}

simdutf_warn_unused result implementation::convert_utf8_to_utf32_with_errors(
    const char *buf, size_t len, char32_t *utf32_output) const noexcept {
  utf8_to_utf32::validating_transcoder converter;
  return converter.convert_with_errors(buf, len, utf32_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf32(
    const char *input, size_t size, char32_t *utf32_output) const noexcept {
  return utf8_to_utf32::convert_valid(input, size, utf32_output);
}

// ===========================================================================
// utf32 -> utf8  (adapted avx2_convert_utf32_to_utf8 kernel + scalar tail)
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf32_to_utf8(
    const char32_t *buf, size_t len, char *utf8_output) const noexcept {
  std::pair<const char32_t *, char *> ret =
      avx2_convert_utf32_to_utf8(buf, len, utf8_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf8_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes = scalar::utf32_to_utf8::convert(
        ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result implementation::convert_utf32_to_utf8_with_errors(
    const char32_t *buf, size_t len, char *utf8_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char *> ret =
      avx2_convert_utf32_to_utf8_with_errors(buf, len, utf8_output);
  if (ret.first.count != len) {
    result scalar_res = scalar::utf32_to_utf8::convert_with_errors(
        buf + ret.first.count, len - ret.first.count, ret.second);
    if (scalar_res.error) {
      scalar_res.count += ret.first.count;
      return scalar_res;
    } else {
      ret.second += scalar_res.count;
    }
  }
  ret.first.count =
      ret.second -
      utf8_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused size_t implementation::convert_valid_utf32_to_utf8(
    const char32_t *buf, size_t len, char *utf8_output) const noexcept {
  return convert_utf32_to_utf8(buf, len, utf8_output);
}

#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
