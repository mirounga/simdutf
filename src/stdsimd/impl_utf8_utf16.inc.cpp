// ==== utf8_utf16 (REAL PORT) ====
//
// This partial holds the utf8 <-> utf16 conversion family (BOTH directions,
// LE + BE) for the stdsimd backend. It is #included into
// stdsimd/implementation.cpp inside
//   namespace simdutf { namespace stdsimd { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp):
//   * utf8 -> utf16 reuses src/generic/utf8_to_utf16/* VERBATIM. Those generic
//     algorithm headers self-wrap in
//       namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ...}}}
//     so they must be pulled in at FILE scope. The generic converter calls the
//     AVX2-specific helper convert_masked_utf8_to_utf16() by unqualified name,
//     which has no portable std::simd form (it is a pshufb/table driven masked
//     transcoder), so we PRAGMATICALLY reuse haswell's pure-intrinsic
//     avx2_convert_utf8_to_utf16.cpp VERBATIM as the escape hatch. It depends
//     only on <immintrin.h> and tables/utf8_to_utf16_tables.h (no haswell simd::
//     types), so it is backend agnostic.
//   * utf16 -> utf8 has no backend-agnostic generic header on the AVX2 path, so
//     it uses the adapted stdsimd/convert_utf16_to_utf8.cpp (which keeps the x86
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

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

// --- anonymous-namespace sources (escape-hatch intrinsic kernels) ------------
// These keep x86 intrinsics: they have no portable std::simd form (table-driven
// byte compression / masked transcoding). Emitted in the backend's anonymous
// namespace, exactly like haswell does, so the generic utf8_to_utf16 converter
// can call convert_masked_utf8_to_utf16() unqualified.
//
// The utf8 -> utf16 masked per-block kernel is fundamentally a 128-bit op (it
// loads a __m128i and emits <=12 utf16). We therefore use westmere's pure
// 128-bit kernel on the SSE and AVX512 tiers (where it runs natively) and
// haswell's 256-bit kernel on the AVX2 tier. The generic driver supplies the
// tier-width ASCII fast path + chunked validation (NUM_CHUNKS 1/2/4). This is
// what lets utf8 -> utf16 run as real SIMD on EVERY tier.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;

  #if SIMDUTF_STDSIMD_AVX2_KERNELS
    #include "haswell/avx2_convert_utf8_to_utf16.cpp"
  #else
    #include "westmere/sse_convert_utf8_to_utf16.cpp"
  #endif

// utf16 -> utf8 bulk transcoder: tier-selects westmere(128) / haswell(256),
// so it runs on EVERY tier (aliased to stdsimd_convert_utf16_to_utf8).
  #include "stdsimd/convert_utf16_to_utf8.cpp"
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// --- generic utf8 -> utf16 algorithm headers (reused VERBATIM, all tiers) -----
  #include "generic/utf8_to_utf16/valid_utf8_to_utf16.h"
  #include "generic/utf8_to_utf16/utf8_to_utf16.h"

// clang-format off
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
// clang-format on

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

// ===========================================================================
// utf8 -> utf16  (reuses generic utf8_to_utf16:: VERBATIM; real SIMD on ALL
// tiers via the 128/256-bit masked kernel above + the NUM_CHUNKS 1/2/4 driver)
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf8_to_utf16le(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  utf8_to_utf16::validating_transcoder converter;
  return converter.convert<endianness::LITTLE>(buf, len, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_utf8_to_utf16be(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  utf8_to_utf16::validating_transcoder converter;
  return converter.convert<endianness::BIG>(buf, len, utf16_output);
}

simdutf_warn_unused result implementation::convert_utf8_to_utf16le_with_errors(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  utf8_to_utf16::validating_transcoder converter;
  return converter.convert_with_errors<endianness::LITTLE>(buf, len,
                                                           utf16_output);
}

simdutf_warn_unused result implementation::convert_utf8_to_utf16be_with_errors(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  utf8_to_utf16::validating_transcoder converter;
  return converter.convert_with_errors<endianness::BIG>(buf, len, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf16le(
    const char *input, size_t size, char16_t *utf16_output) const noexcept {
  return utf8_to_utf16::convert_valid<endianness::LITTLE>(input, size,
                                                          utf16_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf16be(
    const char *input, size_t size, char16_t *utf16_output) const noexcept {
  return utf8_to_utf16::convert_valid<endianness::BIG>(input, size,
                                                       utf16_output);
}

// ===========================================================================
// utf16 -> utf8  (tier-selected 128/256-bit kernel + scalar tail): all tiers.
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf16le_to_utf8(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  std::pair<const char16_t *, char *> ret =
      stdsimd_convert_utf16_to_utf8<endianness::LITTLE>(buf, len, utf8_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf8_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_utf8::convert<endianness::LITTLE>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused size_t implementation::convert_utf16be_to_utf8(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  std::pair<const char16_t *, char *> ret =
      stdsimd_convert_utf16_to_utf8<endianness::BIG>(buf, len, utf8_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf8_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_utf8::convert<endianness::BIG>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result implementation::convert_utf16le_to_utf8_with_errors(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char *> ret =
      stdsimd_convert_utf16_to_utf8_with_errors<endianness::LITTLE>(buf, len,
                                                                 utf8_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_utf8::convert_with_errors<endianness::LITTLE>(
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

simdutf_warn_unused result implementation::convert_utf16be_to_utf8_with_errors(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char *> ret =
      stdsimd_convert_utf16_to_utf8_with_errors<endianness::BIG>(buf, len,
                                                              utf8_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_utf8::convert_with_errors<endianness::BIG>(
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

simdutf_warn_unused size_t implementation::convert_valid_utf16le_to_utf8(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  return convert_utf16le_to_utf8(buf, len, utf8_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf16be_to_utf8(
    const char16_t *buf, size_t len, char *utf8_output) const noexcept {
  return convert_utf16be_to_utf8(buf, len, utf8_output);
}

#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
