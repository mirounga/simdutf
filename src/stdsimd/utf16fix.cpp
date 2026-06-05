// ==== stdsimd utf16fix (to_well_formed_utf16) kernel ====
//
// Adapted from src/haswell/avx2_utf16fix.cpp.
//
// There is a backend-agnostic generic header (src/generic/utf16/to_well_formed.h)
// but it relies on simd16<> wrapper members that the stdsimd wrapper does not yet
// expose (byte_right_shift<>, first(), is_zero(), as_vector_u16()). The haswell
// AVX2 kernel, by contrast, operates entirely on raw __m256i / __m128i values and
// never touches the haswell simd:: types, so -- exactly like
// stdsimd/convert_utf16_to_utf8.cpp -- it is reused as an x86 escape-hatch kernel
// with NO wrapper dependency. The byte-lane shifts (_mm256_bsrli/bslli),
// cross-lane extracts/zext, _mm256_testz_si256, _mm256_blendv_epi8 and
// _mm256_cvtsi256_si32 have no portable std::simd form, so per the stdsimd
// escape-hatch policy the AVX2 intrinsics are kept under #if SIMDUTF_IS_X86_64.
// A portable scalar fallback arm is provided for non-x86 targets.
//
// This file is #included inside namespace simdutf::SIMDUTF_IMPLEMENTATION by
// stdsimd/impl_utf16fix.inc.cpp, within the AVX2 target region applied by
// stdsimd/begin.h.

#if SIMDUTF_IS_X86_64

/*
 * Process one block of 16 characters.  If in_place is false,
 * copy the block from in to out.  If there is a sequencing
 * error in the block, overwrite the illsequenced characters
 * with the replacement character.  This function reads one
 * character before the beginning of the buffer as a lookback.
 * If that character is illsequenced, it too is overwritten.
 */
template <endianness big_endian, bool in_place>
void utf16fix_block(char16_t *out, const char16_t *in) {
  auto swap_if_needed = [](uint16_t x) constexpr -> uint16_t {
    return scalar::utf16::swap_if_needed<big_endian>(x);
  };
  const char16_t replacement = scalar::utf16::replacement<big_endian>();
  __m256i lookback, block, lb_masked, block_masked, lb_is_high, block_is_low;
  __m256i illseq, lb_illseq, block_illseq, lb_illseq_shifted;

  lookback = _mm256_loadu_si256((const __m256i *)(in - 1));
  block = _mm256_loadu_si256((const __m256i *)in);
  lb_masked =
      _mm256_and_si256(lookback, _mm256_set1_epi16(swap_if_needed(0xfc00u)));
  block_masked =
      _mm256_and_si256(block, _mm256_set1_epi16(swap_if_needed(0xfc00u)));
  lb_is_high =
      _mm256_cmpeq_epi16(lb_masked, _mm256_set1_epi16(swap_if_needed(0xd800u)));
  block_is_low = _mm256_cmpeq_epi16(block_masked,
                                    _mm256_set1_epi16(swap_if_needed(0xdc00u)));

  illseq = _mm256_xor_si256(lb_is_high, block_is_low);
  if (!_mm256_testz_si256(illseq, illseq)) {
    int lb;

    /* compute the cause of the illegal sequencing */
    lb_illseq = _mm256_andnot_si256(block_is_low, lb_is_high);
    lb_illseq_shifted =
        _mm256_or_si256(_mm256_bsrli_epi128(lb_illseq, 2),
                        _mm256_zextsi128_si256(_mm_bslli_si128(
                            _mm256_extracti128_si256(lb_illseq, 1), 14)));
    block_illseq = _mm256_or_si256(
        _mm256_andnot_si256(lb_is_high, block_is_low), lb_illseq_shifted);

    /* fix illegal sequencing in the lookback */
    lb = _mm256_cvtsi256_si32(lb_illseq);
    lb = (lb & replacement) | (~lb & out[-1]);
    out[-1] = char16_t(lb);

    /* fix illegal sequencing in the main block */
    block =
        _mm256_blendv_epi8(block, _mm256_set1_epi16(replacement), block_illseq);
    _mm256_storeu_si256((__m256i *)out, block);
  } else if (!in_place) {
    _mm256_storeu_si256((__m256i *)out, block);
  }
}

template <endianness big_endian>
void utf16fix_avx(const char16_t *in, size_t n, char16_t *out) {
  const char16_t replacement = scalar::utf16::replacement<big_endian>();
  size_t i;

  if (n < 17) {
    scalar::utf16::to_well_formed_utf16<big_endian>(in, n, out);
    return;
  }

  out[0] =
      scalar::utf16::is_low_surrogate<big_endian>(in[0]) ? replacement : in[0];

  /* duplicate code to have the compiler specialise utf16fix_block() */
  if (in == out) {
    for (i = 1; i + 16 < n; i += 16) {
      utf16fix_block<big_endian, true>(out + i, in + i);
    }

    utf16fix_block<big_endian, true>(out + n - 16, in + n - 16);
  } else {
    for (i = 1; i + 16 < n; i += 16) {
      utf16fix_block<big_endian, false>(out + i, in + i);
    }

    utf16fix_block<big_endian, false>(out + n - 16, in + n - 16);
  }

  out[n - 1] = scalar::utf16::is_high_surrogate<big_endian>(out[n - 1])
                   ? replacement
                   : out[n - 1];
}

#else // !SIMDUTF_IS_X86_64

// Portable fallback for non-x86 targets: defer to the scalar reference. The AVX2
// kernel above relies on cross-lane byte shuffles with no portable std::simd
// form; a std::simd rewrite would require wrapper members (byte_right_shift,
// first, is_zero) that the stdsimd wrapper does not yet expose.
template <endianness big_endian>
void utf16fix_avx(const char16_t *in, size_t n, char16_t *out) {
  scalar::utf16::to_well_formed_utf16<big_endian>(in, n, out);
}

#endif // SIMDUTF_IS_X86_64
