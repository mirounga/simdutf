// ==== stdsimd utf32 -> utf16 conversion (tier-selected arch kernel) ====
//
// The utf32->utf16 bulk transcoder is a width-specific intrinsic kernel
// (testz / packus_epi32 32->16 narrowing + surrogate-pair expansion) with no
// portable std::simd form and no backend-agnostic generic header.
//   * AVX2/AVX512 tiers (256b): haswell/avx2_convert_utf32_to_utf16.cpp
//     (AVX-512 implies AVX2; pure x86 intrinsics, self-contained.)
//   * SSE tier (128-bit): westmere's sse_convert_utf32_to_utf16.cpp drives its
//     surrogate expansion through westmere's OWN simd:: wrapper library, which
//     would collide with stdsimd's simd:: namespace. So instead of reusing it we
//     carry a pure-intrinsic 128-bit kernel below, faithfully adapted from
//     haswell's algorithm (8 uint32 per iteration via two __m128i loads +
//     _mm_packus_epi32; scalar fallback for words needing a surrogate pair or
//     out of range). No simd:: dependency.
//
// Both branches expose stdsimd_convert_utf32_to_utf16[ _with_errors ].
//
// #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION::{anon} by
// stdsimd/impl_utf16_utf32.inc.cpp.

#if SIMDUTF_STDSIMD_HAS_AVX2
  #include "haswell/avx2_convert_utf32_to_utf16.cpp"
  #define stdsimd_convert_utf32_to_utf16 avx2_convert_utf32_to_utf16
  #define stdsimd_convert_utf32_to_utf16_with_errors                           \
    avx2_convert_utf32_to_utf16_with_errors
#else

template <endianness big_endian>
std::pair<const char32_t *, char16_t *>
stdsimd_convert_utf32_to_utf16(const char32_t *buf, size_t len,
                               char16_t *utf16_output) {
  const char32_t *end = buf + len;
  const size_t safety_margin = 12; // see simdutf issue #92
  __m128i forbidden_bytemask = _mm_setzero_si128();
  const __m128i v_ffff0000 = _mm_set1_epi32((int32_t)0xffff0000);
  const __m128i v_f800 = _mm_set1_epi32((int32_t)0x0000f800);
  const __m128i v_d800 = _mm_set1_epi32((int32_t)0x0000d800);
  const __m128i swap =
      _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

  while (end - buf >= std::ptrdiff_t(8 + safety_margin)) {
    const __m128i in0 = _mm_loadu_si128((const __m128i *)buf);
    const __m128i in1 = _mm_loadu_si128((const __m128i *)(buf + 4));
    if (simdutf_likely(_mm_testz_si128(_mm_or_si128(in0, in1), v_ffff0000))) {
      // all 8 values fit in 16 bits -> pack without surrogate pairs
      forbidden_bytemask = _mm_or_si128(
          forbidden_bytemask,
          _mm_or_si128(_mm_cmpeq_epi32(_mm_and_si128(in0, v_f800), v_d800),
                       _mm_cmpeq_epi32(_mm_and_si128(in1, v_f800), v_d800)));
      __m128i utf16_packed = _mm_packus_epi32(in0, in1);
      if (big_endian) {
        utf16_packed = _mm_shuffle_epi8(utf16_packed, swap);
      }
      _mm_storeu_si128((__m128i *)utf16_output, utf16_packed);
      utf16_output += 8;
      buf += 8;
    } else {
      size_t forward = 7;
      size_t k = 0;
      if (size_t(end - buf) < forward + 1) {
        forward = size_t(end - buf - 1);
      }
      for (; k < forward; k++) {
        uint32_t word = buf[k];
        if ((word & 0xFFFF0000) == 0) {
          if (word >= 0xD800 && word <= 0xDFFF) {
            return std::make_pair(nullptr, utf16_output);
          }
          *utf16_output++ =
              big_endian
                  ? char16_t((uint16_t(word) >> 8) | (uint16_t(word) << 8))
                  : char16_t(word);
        } else {
          if (word > 0x10FFFF) {
            return std::make_pair(nullptr, utf16_output);
          }
          word -= 0x10000;
          uint16_t high_surrogate = uint16_t(0xD800 + (word >> 10));
          uint16_t low_surrogate = uint16_t(0xDC00 + (word & 0x3FF));
          if (big_endian) {
            high_surrogate =
                uint16_t((high_surrogate >> 8) | (high_surrogate << 8));
            low_surrogate =
                uint16_t((low_surrogate >> 8) | (low_surrogate << 8));
          }
          *utf16_output++ = char16_t(high_surrogate);
          *utf16_output++ = char16_t(low_surrogate);
        }
      }
      buf += k;
    }
  }

  if (static_cast<uint16_t>(_mm_movemask_epi8(forbidden_bytemask)) != 0) {
    return std::make_pair(nullptr, utf16_output);
  }
  return std::make_pair(buf, utf16_output);
}

template <endianness big_endian>
std::pair<result, char16_t *>
stdsimd_convert_utf32_to_utf16_with_errors(const char32_t *buf, size_t len,
                                           char16_t *utf16_output) {
  const char32_t *start = buf;
  const char32_t *end = buf + len;
  const size_t safety_margin = 12;
  const __m128i v_ffff0000 = _mm_set1_epi32((int32_t)0xffff0000);
  const __m128i v_f800 = _mm_set1_epi32((int32_t)0x0000f800);
  const __m128i v_d800 = _mm_set1_epi32((int32_t)0x0000d800);
  const __m128i swap =
      _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

  while (end - buf >= std::ptrdiff_t(8 + safety_margin)) {
    const __m128i in0 = _mm_loadu_si128((const __m128i *)buf);
    const __m128i in1 = _mm_loadu_si128((const __m128i *)(buf + 4));
    if (simdutf_likely(_mm_testz_si128(_mm_or_si128(in0, in1), v_ffff0000))) {
      const __m128i forbidden_bytemask =
          _mm_or_si128(_mm_cmpeq_epi32(_mm_and_si128(in0, v_f800), v_d800),
                       _mm_cmpeq_epi32(_mm_and_si128(in1, v_f800), v_d800));
      if (static_cast<uint16_t>(_mm_movemask_epi8(forbidden_bytemask)) != 0) {
        return std::make_pair(result(error_code::SURROGATE, buf - start),
                              utf16_output);
      }
      __m128i utf16_packed = _mm_packus_epi32(in0, in1);
      if (big_endian) {
        utf16_packed = _mm_shuffle_epi8(utf16_packed, swap);
      }
      _mm_storeu_si128((__m128i *)utf16_output, utf16_packed);
      utf16_output += 8;
      buf += 8;
    } else {
      size_t forward = 7;
      size_t k = 0;
      if (size_t(end - buf) < forward + 1) {
        forward = size_t(end - buf - 1);
      }
      for (; k < forward; k++) {
        uint32_t word = buf[k];
        if ((word & 0xFFFF0000) == 0) {
          if (word >= 0xD800 && word <= 0xDFFF) {
            return std::make_pair(
                result(error_code::SURROGATE, buf - start + k), utf16_output);
          }
          *utf16_output++ =
              big_endian
                  ? char16_t((uint16_t(word) >> 8) | (uint16_t(word) << 8))
                  : char16_t(word);
        } else {
          if (word > 0x10FFFF) {
            return std::make_pair(
                result(error_code::TOO_LARGE, buf - start + k), utf16_output);
          }
          word -= 0x10000;
          uint16_t high_surrogate = uint16_t(0xD800 + (word >> 10));
          uint16_t low_surrogate = uint16_t(0xDC00 + (word & 0x3FF));
          if (big_endian) {
            high_surrogate =
                uint16_t((high_surrogate >> 8) | (high_surrogate << 8));
            low_surrogate =
                uint16_t((low_surrogate >> 8) | (low_surrogate << 8));
          }
          *utf16_output++ = char16_t(high_surrogate);
          *utf16_output++ = char16_t(low_surrogate);
        }
      }
      buf += k;
    }
  }
  return std::make_pair(result(error_code::SUCCESS, buf - start), utf16_output);
}

#endif // SIMDUTF_STDSIMD_HAS_AVX2
