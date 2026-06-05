// ==== latin1 conversions family (REAL PORT) ====
//
// This partial holds ALL latin1 conversions in both directions for the stdsimd
// backend. It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp):
//   * utf8 -> latin1 reuses src/generic/utf8_to_latin1/* VERBATIM. Those generic
//     algorithm headers self-wrap in
//       namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ...}}}
//     so they must be pulled in at FILE scope. The generic converter calls the
//     AVX2-specific helper convert_masked_utf8_to_latin1() by unqualified name,
//     which has no portable std::simd form (it is a pshufb/table driven masked
//     transcoder), so we PRAGMATICALLY reuse haswell's pure-intrinsic
//     avx2_convert_utf8_to_latin1.cpp VERBATIM as the escape hatch. It depends
//     only on <immintrin.h> and tables/utf8_to_utf16_tables.h (no haswell simd::
//     types), so it is backend agnostic.
//   * the 6 remaining directions (latin1 -> utf8/utf16/utf32 and
//     utf16/utf32 -> latin1) have no backend-agnostic generic header on the AVX2
//     path, so they use the adapted stdsimd/convert_*.cpp kernels (which keep
//     the x86 AVX2 intrinsics as escape hatches per the stdsimd policy).
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

// --- anonymous-namespace sources (escape-hatch intrinsic kernels) ------------
// All of these keep the x86 AVX2 intrinsics: they have no portable std::simd
// form (table-driven byte compression / widening / narrowing / lane-crossing
// permutes). They are emitted in the backend's anonymous namespace, exactly
// like haswell does, so the generic utf8_to_latin1 converter can call
// convert_masked_utf8_to_latin1() by unqualified name from the enclosing scope,
// and so the implementation methods below can call the avx2_convert_* kernels.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
// latin1 -> utf8 bulk transcoder.
  #include "stdsimd/convert_latin1_to_utf8.cpp"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
// latin1 -> utf16 bulk transcoder.
  #include "stdsimd/convert_latin1_to_utf16.cpp"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
// latin1 -> utf32 bulk transcoder.
  #include "stdsimd/convert_latin1_to_utf32.cpp"
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
// utf16 -> latin1 bulk transcoder.
  #include "stdsimd/convert_utf16_to_latin1.cpp"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
// utf32 -> latin1 bulk transcoder.
  #include "stdsimd/convert_utf32_to_latin1.cpp"
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
// utf8 -> latin1 per-block masked transcoder (pure intrinsics + tables); the
// generic utf8_to_latin1 converter calls convert_masked_utf8_to_latin1() by
// unqualified name.
  #include "haswell/avx2_convert_utf8_to_latin1.cpp"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// --- generic utf8 -> latin1 algorithm headers (reused VERBATIM) --------------
// These self-open simdutf::SIMDUTF_IMPLEMENTATION::{anon}::utf8_to_latin1 and
// call convert_masked_utf8_to_latin1() (defined just above) unqualified.
#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
  #include "generic/utf8_to_latin1/utf8_to_latin1.h"
  #include "generic/utf8_to_latin1/valid_utf8_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

// ===========================================================================
// latin1 -> utf8  (adapted avx2_convert_latin1_to_utf8 kernel + scalar tail)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_latin1_to_utf8(
    const char *buf, size_t len, char *utf8_output) const noexcept {
  std::pair<const char *, char *> ret =
      avx2_convert_latin1_to_utf8(buf, len, utf8_output);
  size_t converted_chars = ret.second - utf8_output;

  if (ret.first != buf + len) {
    const size_t scalar_converted_chars = scalar::latin1_to_utf8::convert(
        ret.first, len - (ret.first - buf), ret.second);
    converted_chars += scalar_converted_chars;
  }

  return converted_chars;
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

// ===========================================================================
// latin1 -> utf16  (adapted avx2_convert_latin1_to_utf16 kernel + scalar tail)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_latin1_to_utf16le(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  std::pair<const char *, char16_t *> ret =
      avx2_convert_latin1_to_utf16<endianness::LITTLE>(buf, len, utf16_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t converted_chars = ret.second - utf16_output;
  if (ret.first != buf + len) {
    const size_t scalar_converted_chars =
        scalar::latin1_to_utf16::convert<endianness::LITTLE>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_converted_chars == 0) {
      return 0;
    }
    converted_chars += scalar_converted_chars;
  }
  return converted_chars;
}

simdutf_warn_unused size_t implementation::convert_latin1_to_utf16be(
    const char *buf, size_t len, char16_t *utf16_output) const noexcept {
  std::pair<const char *, char16_t *> ret =
      avx2_convert_latin1_to_utf16<endianness::BIG>(buf, len, utf16_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t converted_chars = ret.second - utf16_output;
  if (ret.first != buf + len) {
    const size_t scalar_converted_chars =
        scalar::latin1_to_utf16::convert<endianness::BIG>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_converted_chars == 0) {
      return 0;
    }
    converted_chars += scalar_converted_chars;
  }
  return converted_chars;
}
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1

// ===========================================================================
// latin1 -> utf32  (adapted avx2_convert_latin1_to_utf32 kernel + scalar tail)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_latin1_to_utf32(
    const char *buf, size_t len, char32_t *utf32_output) const noexcept {
  std::pair<const char *, char32_t *> ret =
      avx2_convert_latin1_to_utf32(buf, len, utf32_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t converted_chars = ret.second - utf32_output;
  if (ret.first != buf + len) {
    const size_t scalar_converted_chars = scalar::latin1_to_utf32::convert(
        ret.first, len - (ret.first - buf), ret.second);
    if (scalar_converted_chars == 0) {
      return 0;
    }
    converted_chars += scalar_converted_chars;
  }
  return converted_chars;
}
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

// ===========================================================================
// utf8 -> latin1  (reuses generic utf8_to_latin1:: VERBATIM, like haswell)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_utf8_to_latin1(
    const char *buf, size_t len, char *latin1_output) const noexcept {
  utf8_to_latin1::validating_transcoder converter;
  return converter.convert(buf, len, latin1_output);
}

simdutf_warn_unused result implementation::convert_utf8_to_latin1_with_errors(
    const char *buf, size_t len, char *latin1_output) const noexcept {
  utf8_to_latin1::validating_transcoder converter;
  return converter.convert_with_errors(buf, len, latin1_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_latin1(
    const char *buf, size_t len, char *latin1_output) const noexcept {
  return utf8_to_latin1::convert_valid(buf, len, latin1_output);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

// ===========================================================================
// utf16 -> latin1  (adapted avx2_convert_utf16_to_latin1 kernel + scalar tail)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_utf16le_to_latin1(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  std::pair<const char16_t *, char *> ret =
      avx2_convert_utf16_to_latin1<endianness::LITTLE>(buf, len, latin1_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - latin1_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_latin1::convert<endianness::LITTLE>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused size_t implementation::convert_utf16be_to_latin1(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  std::pair<const char16_t *, char *> ret =
      avx2_convert_utf16_to_latin1<endianness::BIG>(buf, len, latin1_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - latin1_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_latin1::convert<endianness::BIG>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result
implementation::convert_utf16le_to_latin1_with_errors(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  std::pair<result, char *> ret =
      avx2_convert_utf16_to_latin1_with_errors<endianness::LITTLE>(
          buf, len, latin1_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_latin1::convert_with_errors<endianness::LITTLE>(
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
      latin1_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused result
implementation::convert_utf16be_to_latin1_with_errors(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  std::pair<result, char *> ret =
      avx2_convert_utf16_to_latin1_with_errors<endianness::BIG>(buf, len,
                                                                latin1_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_latin1::convert_with_errors<endianness::BIG>(
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
      latin1_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused size_t implementation::convert_valid_utf16le_to_latin1(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  // optimization opportunity: implement a custom function
  return convert_utf16le_to_latin1(buf, len, latin1_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf16be_to_latin1(
    const char16_t *buf, size_t len, char *latin1_output) const noexcept {
  // optimization opportunity: implement a custom function
  return convert_utf16be_to_latin1(buf, len, latin1_output);
}
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1

// ===========================================================================
// utf32 -> latin1  (adapted avx2_convert_utf32_to_latin1 kernel + scalar tail)
// ===========================================================================
#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::convert_utf32_to_latin1(
    const char32_t *buf, size_t len, char *latin1_output) const noexcept {
  std::pair<const char32_t *, char *> ret =
      avx2_convert_utf32_to_latin1(buf, len, latin1_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - latin1_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes = scalar::utf32_to_latin1::convert(
        ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result implementation::convert_utf32_to_latin1_with_errors(
    const char32_t *buf, size_t len, char *latin1_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char *> ret =
      avx2_convert_utf32_to_latin1_with_errors(buf, len, latin1_output);
  if (ret.first.count != len) {
    result scalar_res = scalar::utf32_to_latin1::convert_with_errors(
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
      latin1_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused size_t implementation::convert_valid_utf32_to_latin1(
    const char32_t *buf, size_t len, char *latin1_output) const noexcept {
  return convert_utf32_to_latin1(buf, len, latin1_output);
}
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
