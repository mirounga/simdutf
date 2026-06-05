#ifndef SIMDUTF_STDSIMD_TIER_OPS_SSE_H
#define SIMDUTF_STDSIMD_TIER_OPS_SSE_H

// SSE4.2 tier (vec<uint8_t,16>, 128-bit). Implements the tier_* escape-hatch
// API declared in tier_ops.h using single-128-bit _mm_* intrinsics, mirroring
// src/westmere. Pulled in by tier_ops.h when
// SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_SSE.
//
// All helpers operate on the pinned-width vec aliases (v8/v16/v32/v64 from
// tier_ops.h) and bridge to __m128i via std::bit_cast, which is a free
// reinterpret at matching byte width.

static_assert(SIMDUTF_STDSIMD_VEC_BYTES == 16,
              "tier_ops_sse.h requires a 16-byte (128-bit) vector width");

#if !SIMDUTF_IS_X86_64
  #error "tier_ops_sse.h requires x86-64"
#endif

// ---- native bridges -----------------------------------------------------
simdutf_really_inline __m128i tier_to_native(const v8 v) {
  return std::bit_cast<__m128i>(v);
}
simdutf_really_inline v8 tier_from_native(const __m128i r) {
  return std::bit_cast<v8>(r);
}

// ---- to_bitmask ---------------------------------------------------------
simdutf_really_inline uint64_t tier_movemask8(const v8 v) {
  return uint32_t(_mm_movemask_epi8(std::bit_cast<__m128i>(v)));
}

// ---- lookup_16 (per-128-lane pshufb; table pre-broadcast to the lane) ----
simdutf_really_inline v8 tier_shuffle(const v8 table, const v8 index) {
  return std::bit_cast<v8>(_mm_shuffle_epi8(std::bit_cast<__m128i>(table),
                                            std::bit_cast<__m128i>(index)));
}

// Broadcast a 16-byte lane to every 128-bit lane of the vector (here: identity).
simdutf_really_inline v8 tier_repeat16(const std::uint8_t lane[16]) {
  return std::bit_cast<v8>(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(lane)));
}

// ---- prev<N> : concatenation byte shift (single 128-bit lane) -----------
// out[i] = (i >= N) ? cur[i-N] : prev[16-N+i].
// _mm_alignr_epi8(cur, prev, 16-N) yields exactly that within one lane.
template <int N>
simdutf_really_inline v8 tier_alignr(const v8 cur, const v8 prev) {
  const __m128i c = std::bit_cast<__m128i>(cur);
  const __m128i p = std::bit_cast<__m128i>(prev);
  return std::bit_cast<v8>(_mm_alignr_epi8(c, p, 16 - N));
}

// ---- saturating unsigned byte subtract ----------------------------------
simdutf_really_inline v8 tier_subs_epu8(const v8 a, const v8 b) {
  return std::bit_cast<v8>(
      _mm_subs_epu8(std::bit_cast<__m128i>(a), std::bit_cast<__m128i>(b)));
}

// ---- horizontal byte sum ------------------------------------------------
simdutf_really_inline uint64_t tier_sum_bytes(const v8 v) {
  const __m128i tmp =
      _mm_sad_epu8(std::bit_cast<__m128i>(v), _mm_setzero_si128());
  return uint64_t(_mm_extract_epi64(tmp, 0)) +
         uint64_t(_mm_extract_epi64(tmp, 1));
}

// ---- per-8-byte horizontal sums into u64 lanes --------------------------
simdutf_really_inline v64 tier_sad_8groups(const v8 v) {
  return std::bit_cast<v64>(
      _mm_sad_epu8(std::bit_cast<__m128i>(v), _mm_setzero_si128()));
}

// ---- store ascii bytes as utf16 (zero-extend, optional BE swap) ----------
simdutf_really_inline void tier_cvt8to16(const v8 v, char16_t *ptr,
                                         bool big_endian) {
  const __m128i self = std::bit_cast<__m128i>(v);
  __m128i first = _mm_cvtepu8_epi16(self);
  __m128i second = _mm_cvtepu8_epi16(_mm_srli_si128(self, 8));
  if (big_endian) {
    const __m128i swap =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    first = _mm_shuffle_epi8(first, swap);
    second = _mm_shuffle_epi8(second, swap);
  }
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), first);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + 8), second);
}

// ---- store ascii bytes as utf32 (zero-extend) ---------------------------
simdutf_really_inline void tier_cvt8to32(const v8 v, char32_t *ptr) {
  const __m128i self = std::bit_cast<__m128i>(v);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), _mm_cvtepu8_epi32(self));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + 4),
                   _mm_cvtepu8_epi32(_mm_srli_si128(self, 4)));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + 8),
                   _mm_cvtepu8_epi32(_mm_srli_si128(self, 8)));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + 12),
                   _mm_cvtepu8_epi32(_mm_srli_si128(self, 12)));
}

// ---- per-u16-lane byte swap ---------------------------------------------
simdutf_really_inline v16 tier_byteswap16(const v16 v) {
  const __m128i swap =
      _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
  return std::bit_cast<v16>(_mm_shuffle_epi8(std::bit_cast<__m128i>(v), swap));
}

// ---- per-u32-lane byte swap ---------------------------------------------
simdutf_really_inline v32 tier_byteswap32(const v32 v) {
  const __m128i shuffle =
      _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12);
  return std::bit_cast<v32>(
      _mm_shuffle_epi8(std::bit_cast<__m128i>(v), shuffle));
}

// ---- saturating pack of two u16 vectors into one byte vector ------------
// Result ordering is the natural [a..., b...]; at 128-bit there is no
// lane-crossing to undo, so _mm_packus_epi16 already yields [a..., b...].
simdutf_really_inline v8 tier_pack16(const v16 a, const v16 b) {
  return std::bit_cast<v8>(
      _mm_packus_epi16(std::bit_cast<__m128i>(a), std::bit_cast<__m128i>(b)));
}

#endif // SIMDUTF_STDSIMD_TIER_OPS_SSE_H
