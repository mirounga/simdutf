// ==== stdsimd utf16 -> utf32 conversion (adapted from haswell) ====
//
// There is no backend-agnostic generic header for the AVX2 utf16->utf32 path,
// so this file is adapted from src/haswell/avx2_convert_utf16_to_utf32.cpp.
//
// Per the stdsimd escape-hatch policy, the operations used here (movemask /
// cmpeq / 16->32 zero-extension / per-128-lane byte swap) have no portable
// std::simd form on this AVX2 tier, so we PRAGMATICALLY keep the x86 AVX2
// intrinsics. They compile and run under the AVX2 target region applied by
// stdsimd/begin.h (verified equivalent to haswell's). The algorithm never
// leaves raw __m256i, so the vec<->__m256i bridge is not needed here.
//
// This file is #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION by
// stdsimd/impl_utf16_utf32.inc.cpp.

/*
  Returns a pair: the first unprocessed byte from buf and utf32_output
  A scalar routing should carry on the conversion of the tail.
*/
template <endianness big_endian>
std::pair<const char16_t *, char32_t *>
avx2_convert_utf16_to_utf32(const char16_t *buf, size_t len,
                            char32_t *utf32_output) {
  const char16_t *end = buf + len;
  const __m256i v_f800 = _mm256_set1_epi16((int16_t)0xf800);
  const __m256i v_d800 = _mm256_set1_epi16((int16_t)0xd800);

  while (end - buf >= 16) {
    __m256i in = _mm256_loadu_si256((__m256i *)buf);
    if (big_endian) {
      const __m256i swap = _mm256_setr_epi8(
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
          21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
      in = _mm256_shuffle_epi8(in, swap);
    }

    // 1. Check if there are any surrogate word in the input chunk.
    //    We have also deal with situation when there is a surrogate word
    //    at the end of a chunk.
    const __m256i surrogates_bytemask =
        _mm256_cmpeq_epi16(_mm256_and_si256(in, v_f800), v_d800);

    // bitmask = 0x0000 if there are no surrogates
    //         = 0xc000 if the last word is a surrogate
    const uint32_t surrogates_bitmask =
        static_cast<uint32_t>(_mm256_movemask_epi8(surrogates_bytemask));
    // It might seem like checking for surrogates_bitmask == 0xc000 could help.
    // However, it is likely an uncommon occurrence.
    if (surrogates_bitmask == 0x00000000) {
      // case: we extend all sixteen 16-bit code units to sixteen 32-bit code
      // units
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(utf32_output),
                          _mm256_cvtepu16_epi32(_mm256_castsi256_si128(in)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(utf32_output + 8),
          _mm256_cvtepu16_epi32(_mm256_extractf128_si256(in, 1)));
      utf32_output += 16;
      buf += 16;
      // surrogate pair(s) in a register
    } else {
      // Let us do a scalar fallback.
      // It may seem wasteful to use scalar code, but being efficient with SIMD
      // in the presence of surrogate pairs may require non-trivial tables.
      size_t forward = 15;
      size_t k = 0;
      if (size_t(end - buf) < forward + 1) {
        forward = size_t(end - buf - 1);
      }
      for (; k < forward; k++) {
        uint16_t word = scalar::utf16::swap_if_needed<big_endian>(buf[k]);
        if ((word & 0xF800) != 0xD800) {
          // No surrogate pair
          *utf32_output++ = char32_t(word);
        } else {
          // must be a surrogate pair
          uint16_t diff = uint16_t(word - 0xD800);
          uint16_t next_word =
              scalar::utf16::swap_if_needed<big_endian>(buf[k + 1]);
          k++;
          uint16_t diff2 = uint16_t(next_word - 0xDC00);
          if ((diff | diff2) > 0x3FF) {
            return std::make_pair(nullptr, utf32_output);
          }
          uint32_t value = (diff << 10) + diff2 + 0x10000;
          *utf32_output++ = char32_t(value);
        }
      }
      buf += k;
    }
  } // while
  return std::make_pair(buf, utf32_output);
}

/*
  Returns a pair: a result struct and utf8_output.
  If there is an error, the count field of the result is the position of the
  error. Otherwise, it is the position of the first unprocessed byte in buf
  (even if finished). A scalar routing should carry on the conversion of the
  tail if needed.
*/
template <endianness big_endian>
std::pair<result, char32_t *>
avx2_convert_utf16_to_utf32_with_errors(const char16_t *buf, size_t len,
                                        char32_t *utf32_output) {
  const char16_t *start = buf;
  const char16_t *end = buf + len;
  const __m256i v_f800 = _mm256_set1_epi16((int16_t)0xf800);
  const __m256i v_d800 = _mm256_set1_epi16((int16_t)0xd800);

  while (end - buf >= 16) {
    __m256i in = _mm256_loadu_si256((__m256i *)buf);
    if (big_endian) {
      const __m256i swap = _mm256_setr_epi8(
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
          21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
      in = _mm256_shuffle_epi8(in, swap);
    }

    // 1. Check if there are any surrogate word in the input chunk.
    //    We have also deal with situation when there is a surrogate word
    //    at the end of a chunk.
    const __m256i surrogates_bytemask =
        _mm256_cmpeq_epi16(_mm256_and_si256(in, v_f800), v_d800);

    // bitmask = 0x0000 if there are no surrogates
    //         = 0xc000 if the last word is a surrogate
    const uint32_t surrogates_bitmask =
        static_cast<uint32_t>(_mm256_movemask_epi8(surrogates_bytemask));
    // It might seem like checking for surrogates_bitmask == 0xc000 could help.
    // However, it is likely an uncommon occurrence.
    if (surrogates_bitmask == 0x00000000) {
      // case: we extend all sixteen 16-bit code units to sixteen 32-bit code
      // units
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(utf32_output),
                          _mm256_cvtepu16_epi32(_mm256_castsi256_si128(in)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(utf32_output + 8),
          _mm256_cvtepu16_epi32(_mm256_extractf128_si256(in, 1)));
      utf32_output += 16;
      buf += 16;
      // surrogate pair(s) in a register
    } else {
      // Let us do a scalar fallback.
      // It may seem wasteful to use scalar code, but being efficient with SIMD
      // in the presence of surrogate pairs may require non-trivial tables.
      size_t forward = 15;
      size_t k = 0;
      if (size_t(end - buf) < forward + 1) {
        forward = size_t(end - buf - 1);
      }
      for (; k < forward; k++) {
        uint16_t word = scalar::utf16::swap_if_needed<big_endian>(buf[k]);
        if ((word & 0xF800) != 0xD800) {
          // No surrogate pair
          *utf32_output++ = char32_t(word);
        } else {
          // must be a surrogate pair
          uint16_t diff = uint16_t(word - 0xD800);
          uint16_t next_word =
              scalar::utf16::swap_if_needed<big_endian>(buf[k + 1]);
          k++;
          uint16_t diff2 = uint16_t(next_word - 0xDC00);
          if ((diff | diff2) > 0x3FF) {
            return std::make_pair(
                result(error_code::SURROGATE, buf - start + k - 1),
                utf32_output);
          }
          uint32_t value = (diff << 10) + diff2 + 0x10000;
          *utf32_output++ = char32_t(value);
        }
      }
      buf += k;
    }
  } // while
  return std::make_pair(result(error_code::SUCCESS, buf - start), utf32_output);
}
