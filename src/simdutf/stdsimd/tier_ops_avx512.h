#ifndef SIMDUTF_STDSIMD_TIER_OPS_AVX512_H
#define SIMDUTF_STDSIMD_TIER_OPS_AVX512_H

// AVX-512 tier (vec<uint8_t,64>, 512-bit). Implements the tier_* escape-hatch
// API declared in tier_ops.h. Pulled in by tier_ops.h when
// SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX512.
//
// All helpers operate on the pinned-width vec aliases (v8/v16/v32/v64 from
// tier_ops.h) and bridge to __m512i via std::bit_cast, which is a free
// reinterpret at matching byte width (vec<u8,64> is exactly 64 bytes).
//
// Target ISA: avx512 f/bw/cd/dq/vl/vbmi (base64 additionally needs vbmi2).
// vbmi is required here for tier_alignr's _mm512_permutex2var_epi8 byte-granular
// cross-lane shuffle; everything else is f/bw/dq.

static_assert(SIMDUTF_STDSIMD_VEC_BYTES == 64,
              "tier_ops_avx512.h requires a 64-byte (512-bit) vector width");

#if !SIMDUTF_IS_X86_64
  #error "tier_ops_avx512.h requires x86-64"
#endif

// ---- native bridges -----------------------------------------------------
simdutf_really_inline __m512i tier_to_native(const v8 v) {
  return std::bit_cast<__m512i>(v);
}
simdutf_really_inline v8 tier_from_native(const __m512i r) {
  return std::bit_cast<v8>(r);
}

// ---- to_bitmask ---------------------------------------------------------
// No movemask at 512 bits; _mm512_movepi8_mask packs each byte's high bit into
// a __mmask64 register, which is exactly the 64-bit bitmask we want.
simdutf_really_inline uint64_t tier_movemask8(const v8 v) {
  return uint64_t(_mm512_movepi8_mask(std::bit_cast<__m512i>(v)));
}

// ---- lookup_16 (per-128-lane pshufb; table pre-broadcast to all 4 lanes) --
// The wrapper's repeat_16 already replicates the 16-byte table across all 64
// bytes, so a plain _mm512_shuffle_epi8 (which is per-128-bit-lane) is correct.
simdutf_really_inline v8 tier_shuffle(const v8 table, const v8 index) {
  return std::bit_cast<v8>(_mm512_shuffle_epi8(std::bit_cast<__m512i>(table),
                                               std::bit_cast<__m512i>(index)));
}

// Broadcast a 16-byte lane to all four 128-bit lanes of the 512-bit vector.
simdutf_really_inline v8 tier_repeat16(const std::uint8_t lane[16]) {
  return std::bit_cast<v8>(_mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(lane))));
}

// ---- prev<N> : cross-lane concatenation byte shift ----------------------
// out[i] = (i>=N) ? cur[i-N] : prev[64-N+i]. N is a compile-time constant.
// _mm512_permutex2var_epi8(a, idx, b) gathers: idx in 0..63 -> a (=prev),
// idx in 64..127 -> b (=cur). Setting idx[i] = (64 - N + i) gives:
//   i <  N : idx = 64-N+i in [64-N, 63]      -> prev[64-N+i]   (low half)
//   i >= N : idx = 64-N+i in [64, 127-N]     -> cur[(64-N+i)-64] = cur[i-N]
// Requires avx512vbmi for the byte-granular permute.
template <int N>
simdutf_really_inline v8 tier_alignr(const v8 cur, const v8 prev) {
  static_assert(N >= 0 && N <= 64, "tier_alignr: N out of range");
  const __m512i idx = _mm512_set_epi8(
      char(64 - N + 63), char(64 - N + 62), char(64 - N + 61), char(64 - N + 60),
      char(64 - N + 59), char(64 - N + 58), char(64 - N + 57), char(64 - N + 56),
      char(64 - N + 55), char(64 - N + 54), char(64 - N + 53), char(64 - N + 52),
      char(64 - N + 51), char(64 - N + 50), char(64 - N + 49), char(64 - N + 48),
      char(64 - N + 47), char(64 - N + 46), char(64 - N + 45), char(64 - N + 44),
      char(64 - N + 43), char(64 - N + 42), char(64 - N + 41), char(64 - N + 40),
      char(64 - N + 39), char(64 - N + 38), char(64 - N + 37), char(64 - N + 36),
      char(64 - N + 35), char(64 - N + 34), char(64 - N + 33), char(64 - N + 32),
      char(64 - N + 31), char(64 - N + 30), char(64 - N + 29), char(64 - N + 28),
      char(64 - N + 27), char(64 - N + 26), char(64 - N + 25), char(64 - N + 24),
      char(64 - N + 23), char(64 - N + 22), char(64 - N + 21), char(64 - N + 20),
      char(64 - N + 19), char(64 - N + 18), char(64 - N + 17), char(64 - N + 16),
      char(64 - N + 15), char(64 - N + 14), char(64 - N + 13), char(64 - N + 12),
      char(64 - N + 11), char(64 - N + 10), char(64 - N + 9), char(64 - N + 8),
      char(64 - N + 7), char(64 - N + 6), char(64 - N + 5), char(64 - N + 4),
      char(64 - N + 3), char(64 - N + 2), char(64 - N + 1), char(64 - N + 0));
  // permutex2var(a, idx, b): a = prev (indices 0..63), b = cur (indices 64..127)
  return std::bit_cast<v8>(_mm512_permutex2var_epi8(
      std::bit_cast<__m512i>(prev), idx, std::bit_cast<__m512i>(cur)));
}

// ---- saturating unsigned byte subtract ----------------------------------
simdutf_really_inline v8 tier_subs_epu8(const v8 a, const v8 b) {
  return std::bit_cast<v8>(
      _mm512_subs_epu8(std::bit_cast<__m512i>(a), std::bit_cast<__m512i>(b)));
}

// ---- horizontal byte sum ------------------------------------------------
simdutf_really_inline uint64_t tier_sum_bytes(const v8 v) {
  const __m512i tmp =
      _mm512_sad_epu8(std::bit_cast<__m512i>(v), _mm512_setzero_si512());
  return uint64_t(_mm512_reduce_add_epi64(tmp));
}

// ---- per-8-byte horizontal sums into u64 lanes --------------------------
// _mm512_sad_epu8 yields one u64 sum per 8-byte group (8 groups -> vec<u64,8>).
simdutf_really_inline v64 tier_sad_8groups(const v8 v) {
  return std::bit_cast<v64>(
      _mm512_sad_epu8(std::bit_cast<__m512i>(v), _mm512_setzero_si512()));
}

// ---- store ascii bytes as utf16 (zero-extend, optional BE swap) ----------
// 64 bytes -> 64 u16. Two halves: low 256 bits and high 256 bits, each widened
// to a full 512-bit register of 32 u16.
simdutf_really_inline void tier_cvt8to16(const v8 v, char16_t *ptr,
                                         bool big_endian) {
  const __m512i self = std::bit_cast<__m512i>(v);
  __m512i first = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(self));
  __m512i second =
      _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(self, 1));
  if (big_endian) {
    const __m512i swap = _mm512_broadcast_i32x4(_mm_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
    first = _mm512_shuffle_epi8(first, swap);
    second = _mm512_shuffle_epi8(second, swap);
  }
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), first);
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr + 32), second);
}

// ---- store ascii bytes as utf32 (zero-extend) ---------------------------
// 64 bytes -> 64 u32. Four 128-bit quarters, each widened to 16 u32.
simdutf_really_inline void tier_cvt8to32(const v8 v, char32_t *ptr) {
  const __m512i self = std::bit_cast<__m512i>(v);
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr),
                      _mm512_cvtepu8_epi32(_mm512_castsi512_si128(self)));
  _mm512_storeu_si512(
      reinterpret_cast<__m512i *>(ptr + 16),
      _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(self, 1)));
  _mm512_storeu_si512(
      reinterpret_cast<__m512i *>(ptr + 32),
      _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(self, 2)));
  _mm512_storeu_si512(
      reinterpret_cast<__m512i *>(ptr + 48),
      _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(self, 3)));
}

// ---- per-u16-lane byte swap ---------------------------------------------
simdutf_really_inline v16 tier_byteswap16(const v16 v) {
  const __m512i swap = _mm512_broadcast_i32x4(
      _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
  return std::bit_cast<v16>(
      _mm512_shuffle_epi8(std::bit_cast<__m512i>(v), swap));
}

// ---- per-u32-lane byte swap ---------------------------------------------
simdutf_really_inline v32 tier_byteswap32(const v32 v) {
  const __m512i swap = _mm512_broadcast_i32x4(
      _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
  return std::bit_cast<v32>(
      _mm512_shuffle_epi8(std::bit_cast<__m512i>(v), swap));
}

// ---- saturating pack of two u16 vectors into one byte vector ------------
// Result ordering is the natural [a..., b...]. _mm512_packus_epi16 interleaves
// per-128-bit-lane (lanes: a0,b0,a1,b1,...), so undo the lane crossing with a
// 64-bit-granular permute (0,2,4,6,1,3,5,7), mirroring src/icelake load_block.
simdutf_really_inline v8 tier_pack16(const v16 a, const v16 b) {
  const __m512i p = _mm512_packus_epi16(std::bit_cast<__m512i>(a),
                                        std::bit_cast<__m512i>(b));
  const __m512i ordered = _mm512_permutexvar_epi64(
      _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), p);
  return std::bit_cast<v8>(ordered);
}

#endif // SIMDUTF_STDSIMD_TIER_OPS_AVX512_H
