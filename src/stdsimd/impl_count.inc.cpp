// ==== count + length-estimation family (REAL PORT) ====
//
// This partial holds count_utf8, count_utf16le/be and ALL *_length_from_*
// methods for the stdsimd backend. It is #included into
// stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h. The
// #if SIMDUTF_FEATURE_* guards mirror the originals.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp method-by-method):
//   * count_utf8 / latin1_length_from_utf8 / utf32_length_from_utf8 reuse
//     generic/utf8.h (count_code_points / count_code_points_bytemask) VERBATIM.
//   * utf16_length_from_utf8 reuses generic/utf8/utf16_length_from_utf8_bytemask.h
//     VERBATIM.
//   * count_utf16le/be and utf32_length_from_utf16le/be reuse generic/utf16.h
//     (count_code_points / utf32_length_from_utf16) VERBATIM.
//   * utf8_length_from_utf16le/be and the *_with_replacement variants reuse
//     generic/utf16/utf8_length_from_utf16_bytemask.h VERBATIM.
//   * utf8_length_from_utf32 reuses generic/utf32.h VERBATIM.
//   * utf8_length_from_latin1 and utf16_length_from_utf32 have no
//     backend-agnostic generic header on the AVX2 path -- haswell inlines pure
//     _mm256_* kernels. We keep those as x86 escape hatches (guarded by
//     SIMDUTF_IS_X86_64) per the stdsimd escape-hatch policy, with a scalar
//     fallback for non-x86 targets.
//
// The generic algorithm headers self-wrap in
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ... } } }
// so they must be pulled in at FILE scope. Because this partial is textually
// included while simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly
// CLOSE those two namespaces, do the file-scope work (the escape-hatch kernels
// in the backend's anonymous namespace followed by the generic headers), then
// REOPEN the namespaces so the rest of implementation.cpp continues unchanged.
// The namespace close/reopen is UNCONDITIONAL so the parent file's brace
// balance never depends on a feature macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the generic headers (which open
// ---- their own simdutf::SIMDUTF_IMPLEMENTATION) nest at file scope. ----------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

// --- escape-hatch intrinsic kernels (anonymous namespace) --------------------
// These two helpers correspond to the pure-intrinsic kernels haswell inlines
// directly into implementation.cpp. They use only raw __m256i / _mm256_*
// (no simd:: wrapper types), so they are emitted as free helpers in the
// backend's anonymous namespace.
#if (SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1) ||                         \
    (SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32)
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {

  #if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
simdutf_really_inline size_t
stdsimd_utf8_length_from_latin1(const char *input, size_t len) {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(input);
    // 256-bit kernel: AVX2 tier only. SSE/AVX512 tiers fall through to scalar.
    #if SIMDUTF_STDSIMD_AVX2_KERNELS
  size_t answer = len / sizeof(__m256i) * sizeof(__m256i);
  size_t i = 0;
  if (answer >= 2048) { // long strings optimization
    __m256i four_64bits = _mm256_setzero_si256();
    while (i + sizeof(__m256i) <= len) {
      __m256i runner = _mm256_setzero_si256();
      // We can do up to 255 loops without overflow.
      size_t iterations = (len - i) / sizeof(__m256i);
      if (iterations > 255) {
        iterations = 255;
      }
      size_t max_i = i + iterations * sizeof(__m256i) - sizeof(__m256i);
      for (; i + 4 * sizeof(__m256i) <= max_i; i += 4 * sizeof(__m256i)) {
        __m256i input1 = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i input2 =
            _mm256_loadu_si256((const __m256i *)(data + i + sizeof(__m256i)));
        __m256i input3 = _mm256_loadu_si256(
            (const __m256i *)(data + i + 2 * sizeof(__m256i)));
        __m256i input4 = _mm256_loadu_si256(
            (const __m256i *)(data + i + 3 * sizeof(__m256i)));
        __m256i input12 =
            _mm256_add_epi8(_mm256_cmpgt_epi8(_mm256_setzero_si256(), input1),
                            _mm256_cmpgt_epi8(_mm256_setzero_si256(), input2));
        __m256i input23 =
            _mm256_add_epi8(_mm256_cmpgt_epi8(_mm256_setzero_si256(), input3),
                            _mm256_cmpgt_epi8(_mm256_setzero_si256(), input4));
        __m256i input1234 = _mm256_add_epi8(input12, input23);
        runner = _mm256_sub_epi8(runner, input1234);
      }
      for (; i <= max_i; i += sizeof(__m256i)) {
        __m256i input_256_chunk =
            _mm256_loadu_si256((const __m256i *)(data + i));
        runner = _mm256_sub_epi8(
            runner, _mm256_cmpgt_epi8(_mm256_setzero_si256(), input_256_chunk));
      }
      four_64bits = _mm256_add_epi64(
          four_64bits, _mm256_sad_epu8(runner, _mm256_setzero_si256()));
    }
    answer += _mm256_extract_epi64(four_64bits, 0) +
              _mm256_extract_epi64(four_64bits, 1) +
              _mm256_extract_epi64(four_64bits, 2) +
              _mm256_extract_epi64(four_64bits, 3);
  } else if (answer > 0) {
    for (; i + sizeof(__m256i) <= len; i += sizeof(__m256i)) {
      __m256i latin = _mm256_loadu_si256((const __m256i *)(data + i));
      uint32_t non_ascii = _mm256_movemask_epi8(latin);
      answer += count_ones(non_ascii);
    }
  }
  return answer + scalar::latin1::utf8_length_from_latin1(
                      reinterpret_cast<const char *>(data + i), len - i);
    #else  // !SIMDUTF_STDSIMD_AVX2_KERNELS
  return scalar::latin1::utf8_length_from_latin1(
      reinterpret_cast<const char *>(data), len);
    #endif // SIMDUTF_STDSIMD_AVX2_KERNELS
}
  #endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

  #if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
simdutf_really_inline size_t
stdsimd_utf16_length_from_utf32(const char32_t *input, size_t length) {
    // 256-bit kernel: AVX2 tier only. SSE/AVX512 tiers fall through to scalar.
    #if SIMDUTF_STDSIMD_AVX2_KERNELS
  const __m256i v_00000000 = _mm256_setzero_si256();
  const __m256i v_ffff0000 = _mm256_set1_epi32((uint32_t)0xffff0000);
  size_t pos = 0;
  size_t count = 0;
  for (; pos + 8 <= length; pos += 8) {
    __m256i in = _mm256_loadu_si256((__m256i *)(input + pos));
    const __m256i surrogate_bytemask =
        _mm256_cmpeq_epi32(_mm256_and_si256(in, v_ffff0000), v_00000000);
    const uint32_t surrogate_bitmask =
        static_cast<uint32_t>(_mm256_movemask_epi8(surrogate_bytemask));
    size_t surrogate_count = (32 - count_ones(surrogate_bitmask)) / 4;
    count += 8 + surrogate_count;
  }
  return count +
         scalar::utf32::utf16_length_from_utf32(input + pos, length - pos);
    #else  // !SIMDUTF_STDSIMD_AVX2_KERNELS
  return scalar::utf32::utf16_length_from_utf32(input, length);
    #endif // SIMDUTF_STDSIMD_AVX2_KERNELS
}
  #endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf
#endif // (SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1) ||
       // (SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32)

// --- generic count/length algorithm headers (reused VERBATIM) ----------------
// These mirror exactly which headers haswell/implementation.cpp pulls in.
#if SIMDUTF_FEATURE_UTF8
  #include "generic/utf8.h"
#endif // SIMDUTF_FEATURE_UTF8

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
  #include "generic/utf8/utf16_length_from_utf8_bytemask.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF16
  #include "generic/utf16.h"
  #include "generic/utf16/utf8_length_from_utf16_bytemask.h"
#endif // SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
  #include "generic/utf32.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF16
simdutf_warn_unused size_t implementation::count_utf16le(
    const char16_t *input, size_t length) const noexcept {
  return utf16::count_code_points<endianness::LITTLE>(input, length);
}

simdutf_warn_unused size_t implementation::count_utf16be(
    const char16_t *input, size_t length) const noexcept {
  return utf16::count_code_points<endianness::BIG>(input, length);
}
#endif // SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF8
simdutf_warn_unused size_t
implementation::count_utf8(const char *input, size_t length) const noexcept {
  return utf8::count_code_points_bytemask(input, length);
}
#endif // SIMDUTF_FEATURE_UTF8

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::latin1_length_from_utf8(
    const char *buf, size_t len) const noexcept {
  return count_utf8(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
simdutf_warn_unused size_t implementation::utf8_length_from_latin1(
    const char *input, size_t length) const noexcept {
  return stdsimd_utf8_length_from_latin1(input, length);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
simdutf_warn_unused size_t implementation::utf8_length_from_utf16le(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf8_length_from_utf16_bytemask<endianness::LITTLE>(input,
                                                                    length);
}

simdutf_warn_unused size_t implementation::utf8_length_from_utf16be(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf8_length_from_utf16_bytemask<endianness::BIG>(input, length);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
simdutf_warn_unused size_t implementation::utf32_length_from_utf16le(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf32_length_from_utf16<endianness::LITTLE>(input, length);
}

simdutf_warn_unused size_t implementation::utf32_length_from_utf16be(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf32_length_from_utf16<endianness::BIG>(input, length);
}
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
simdutf_warn_unused size_t implementation::utf16_length_from_utf8(
    const char *input, size_t length) const noexcept {
  return utf8::utf16_length_from_utf8_bytemask(input, length);
}

simdutf_warn_unused result
implementation::utf8_length_from_utf16le_with_replacement(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf8_length_from_utf16_with_replacement<endianness::LITTLE>(
      input, length);
}

simdutf_warn_unused result
implementation::utf8_length_from_utf16be_with_replacement(
    const char16_t *input, size_t length) const noexcept {
  return utf16::utf8_length_from_utf16_with_replacement<endianness::BIG>(input,
                                                                         length);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
simdutf_warn_unused size_t implementation::utf8_length_from_utf32(
    const char32_t *input, size_t length) const noexcept {
  return utf32::utf8_length_from_utf32(input, length);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
simdutf_warn_unused size_t implementation::utf16_length_from_utf32(
    const char32_t *input, size_t length) const noexcept {
  return stdsimd_utf16_length_from_utf32(input, length);
}
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
simdutf_warn_unused size_t implementation::utf32_length_from_utf8(
    const char *input, size_t length) const noexcept {
  return utf8::count_code_points(input, length);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
