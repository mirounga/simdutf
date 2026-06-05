#ifndef SIMDUTF_STDSIMD_TIER_OPS_AVX2_H
#define SIMDUTF_STDSIMD_TIER_OPS_AVX2_H

// AVX2 tier (vec<uint8_t,32>, 256-bit). Implements the tier_* escape-hatch API
// declared in tier_ops.h. Pulled in by tier_ops.h when
// SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX2 (the default). This is the
// code that previously lived inline in simd.h / simd*-inl.h, moved verbatim.
//
// All helpers operate on the pinned-width vec aliases (v8/v16/v32/v64 from
// tier_ops.h) and bridge to __m256i via std::bit_cast, which is a free
// reinterpret at matching byte width.

static_assert(SIMDUTF_STDSIMD_VEC_BYTES == 32,
              "tier_ops_avx2.h requires a 32-byte (256-bit) vector width");

#if !SIMDUTF_IS_X86_64
  #error "tier_ops_avx2.h requires x86-64"
#endif

// ---- native bridges -----------------------------------------------------
simdutf_really_inline __m256i tier_to_native(const v8 v) {
  return std::bit_cast<__m256i>(v);
}
simdutf_really_inline v8 tier_from_native(const __m256i r) {
  return std::bit_cast<v8>(r);
}

// ---- to_bitmask ---------------------------------------------------------
simdutf_really_inline uint64_t tier_movemask8(const v8 v) {
  return uint32_t(_mm256_movemask_epi8(std::bit_cast<__m256i>(v)));
}

// ---- lookup_16 (per-128-lane pshufb; table pre-broadcast to both lanes) --
simdutf_really_inline v8 tier_shuffle(const v8 table, const v8 index) {
  return std::bit_cast<v8>(_mm256_shuffle_epi8(std::bit_cast<__m256i>(table),
                                               std::bit_cast<__m256i>(index)));
}

// Broadcast a 16-byte lane to both 128-bit lanes of the 256-bit vector.
simdutf_really_inline v8 tier_repeat16(const std::uint8_t lane[16]) {
  return std::bit_cast<v8>(_mm256_broadcastsi128_si256(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(lane))));
}

// ---- prev<N> : cross-lane concatenation byte shift ----------------------
template <int N>
simdutf_really_inline v8 tier_alignr(const v8 cur, const v8 prev) {
  const __m256i c = std::bit_cast<__m256i>(cur);
  const __m256i p = std::bit_cast<__m256i>(prev);
  return std::bit_cast<v8>(
      _mm256_alignr_epi8(c, _mm256_permute2x128_si256(p, c, 0x21), 16 - N));
}

// ---- saturating unsigned byte subtract ----------------------------------
simdutf_really_inline v8 tier_subs_epu8(const v8 a, const v8 b) {
  return std::bit_cast<v8>(
      _mm256_subs_epu8(std::bit_cast<__m256i>(a), std::bit_cast<__m256i>(b)));
}

// ---- horizontal byte sum ------------------------------------------------
simdutf_really_inline uint64_t tier_sum_bytes(const v8 v) {
  const __m256i tmp =
      _mm256_sad_epu8(std::bit_cast<__m256i>(v), _mm256_setzero_si256());
  return _mm256_extract_epi64(tmp, 0) + _mm256_extract_epi64(tmp, 1) +
         _mm256_extract_epi64(tmp, 2) + _mm256_extract_epi64(tmp, 3);
}

// ---- per-8-byte horizontal sums into u64 lanes --------------------------
simdutf_really_inline v64 tier_sad_8groups(const v8 v) {
  return std::bit_cast<v64>(
      _mm256_sad_epu8(std::bit_cast<__m256i>(v), _mm256_setzero_si256()));
}

// ---- store ascii bytes as utf16 (zero-extend, optional BE swap) ----------
simdutf_really_inline void tier_cvt8to16(const v8 v, char16_t *ptr,
                                         bool big_endian) {
  const __m256i self = std::bit_cast<__m256i>(v);
  __m256i first = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(self));
  __m256i second = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(self, 1));
  if (big_endian) {
    const __m256i swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
        21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
    first = _mm256_shuffle_epi8(first, swap);
    second = _mm256_shuffle_epi8(second, swap);
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), first);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 16), second);
}

// ---- store ascii bytes as utf32 (zero-extend) ---------------------------
simdutf_really_inline void tier_cvt8to32(const v8 v, char32_t *ptr) {
  const __m256i self = std::bit_cast<__m256i>(v);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr),
                      _mm256_cvtepu8_epi32(_mm256_castsi256_si128(self)));
  _mm256_storeu_si256(
      reinterpret_cast<__m256i *>(ptr + 8),
      _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_srli_si256(self, 8))));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 16),
                      _mm256_cvtepu8_epi32(_mm256_extractf128_si256(self, 1)));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 24),
                      _mm256_cvtepu8_epi32(_mm_srli_si128(
                          _mm256_extractf128_si256(self, 1), 8)));
}

// ---- per-u16-lane byte swap ---------------------------------------------
simdutf_really_inline v16 tier_byteswap16(const v16 v) {
  const __m256i swap = _mm256_setr_epi8(
      1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21,
      20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
  return std::bit_cast<v16>(
      _mm256_shuffle_epi8(std::bit_cast<__m256i>(v), swap));
}

// ---- per-u32-lane byte swap ---------------------------------------------
simdutf_really_inline v32 tier_byteswap32(const v32 v) {
  const __m256i shuffle =
      _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 3,
                       2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12);
  return std::bit_cast<v32>(
      _mm256_shuffle_epi8(std::bit_cast<__m256i>(v), shuffle));
}

// ---- saturating pack of two u16 vectors into one byte vector ------------
// Result ordering is the natural [a..., b...] (lane-crossing undone).
simdutf_really_inline v8 tier_pack16(const v16 a, const v16 b) {
  const __m256i va = std::bit_cast<__m256i>(a);
  const __m256i vb = std::bit_cast<__m256i>(b);
  const __m128i lo_0 = _mm256_extracti128_si256(va, 0);
  const __m128i lo_1 = _mm256_extracti128_si256(vb, 0);
  const __m128i hi_0 = _mm256_extracti128_si256(va, 1);
  const __m128i hi_1 = _mm256_extracti128_si256(vb, 1);
  const __m256i t0 = _mm256_permute2f128_si256(
      _mm256_castsi128_si256(lo_0), _mm256_castsi128_si256(lo_1), 0x20);
  const __m256i t1 = _mm256_permute2f128_si256(
      _mm256_castsi128_si256(hi_0), _mm256_castsi128_si256(hi_1), 0x20);
  return std::bit_cast<v8>(_mm256_packus_epi16(t0, t1));
}

#endif // SIMDUTF_STDSIMD_TIER_OPS_AVX2_H
