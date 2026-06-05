#ifndef SIMDUTF_STDSIMD_TIER_OPS_H
#define SIMDUTF_STDSIMD_TIER_OPS_H

// Width/tier-parameterized escape-hatch helpers for the std::simd wrapper.
//
// The wrapper (simd.h + simd16/32/64-inl.h) presents the haswell interface over
// pinned-width std::simd::vec. Most operations are portable std::simd; a small
// set has no portable form and is implemented per ISA tier with intrinsics.
// THIS header is the only place those intrinsics live. The wrapper calls ONLY
// the tier_* helpers below; it never spells a raw _mm* intrinsic.
//
// Tier selection / vector width
// -----------------------------
//   SIMDUTF_STDSIMD_VEC_BYTES : 16 (SSE) | 32 (AVX2) | 64 (AVX512). The wrapper
//                               pins v8 = vec<uint8_t, BYTES>, simd16 = BYTES/2
//                               u16 lanes, simd8x64 NUM_CHUNKS = 64/BYTES, etc.
//   SIMDUTF_STDSIMD_TIER      : SIMDUTF_STDSIMD_TIER_{SSE,AVX2,AVX512}, derived
//                               from VEC_BYTES if not set explicitly. Selects
//                               which per-tier implementation header is pulled.
//
// Each per-tier header (tier_ops_{sse,avx2,avx512}.h) defines the helpers below
// as simdutf_really_inline functions in this same namespace, operating on the
// pinned-width byte vector v8 (= ss::vec<uint8_t, BYTES>) and the half/quarter
// width vectors v16/v32. Bridging to native registers is internal to each
// per-tier header via std::bit_cast (a free reinterpret at matching N bytes).
//
// Helper API (implemented per tier)
// ---------------------------------
//   uint64_t tier_movemask8(v8 byte_vector)
//       Pack the high bit of every byte lane into a bitmask.
//       SSE:16b  AVX2:32b  AVX512:64b. Used by to_bitmask.
//
//   v8 tier_shuffle(v8 table, v8 index)
//       Per-128-bit-lane pshufb. The 16-byte table is assumed pre-broadcast to
//       every 128-bit lane (the wrapper's repeat_16 already does this). Used by
//       lookup_16.
//
//   template <int N> v8 tier_alignr(v8 cur, v8 prev)
//       Cross-lane concatenation byte-shift: out[i] = (i>=N) ? cur[i-N]
//       : prev[BYTES-N+i]. N is a compile-time constant. Used by prev<N>.
//
//   v8 tier_subs_epu8(v8 a, v8 b)              saturating unsigned byte subtract.
//
//   uint64_t tier_sum_bytes(v8 v)              horizontal sum of all byte lanes.
//
//   v64 tier_sad_8groups(v8 v)                 per-8-byte horizontal sums into
//                                              u64 lanes (vec<u64, BYTES/8>).
//                                              Used by sum_8bytes.
//
//   void tier_cvt8to16(v8 v, char16_t *ptr, bool big_endian)
//       Zero-extend every byte to u16 and store BYTES code units at ptr,
//       byteswapped iff big_endian. Used by store_ascii_as_utf16.
//
//   void tier_cvt8to32(v8 v, char32_t *ptr)
//       Zero-extend every byte to u32 and store BYTES code units at ptr.
//       Used by store_ascii_as_utf32.
//
//   v16 tier_byteswap16(v16 v)                 per-u16-lane byte swap (BE<->LE).
//   v32 tier_byteswap32(v32 v)                 per-u32-lane byte swap (BE<->LE).
//
//   v8 tier_pack16(v16 a, v16 b)
//       Unsigned-saturate two u16 vectors into one byte vector with the natural
//       (non-lane-crossed) ordering [a..., b...]. Used by simd16::pack.

// ---- tier id constants --------------------------------------------------
#define SIMDUTF_STDSIMD_TIER_SSE 1
#define SIMDUTF_STDSIMD_TIER_AVX2 2
#define SIMDUTF_STDSIMD_TIER_AVX512 3

// ---- default width = AVX2 if unset --------------------------------------
#ifndef SIMDUTF_STDSIMD_VEC_BYTES
  #define SIMDUTF_STDSIMD_VEC_BYTES 32
#endif

// ---- derive TIER from width if not explicitly set -----------------------
#ifndef SIMDUTF_STDSIMD_TIER
  #if SIMDUTF_STDSIMD_VEC_BYTES == 16
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_SSE
  #elif SIMDUTF_STDSIMD_VEC_BYTES == 32
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_AVX2
  #elif SIMDUTF_STDSIMD_VEC_BYTES == 64
    #define SIMDUTF_STDSIMD_TIER SIMDUTF_STDSIMD_TIER_AVX512
  #else
    #error "SIMDUTF_STDSIMD_VEC_BYTES must be 16, 32, or 64"
  #endif
#endif

// ---- pinned-width vector aliases (derived from VEC_BYTES) ----------------
// These are the storage types the whole wrapper is built on. They live in the
// simd namespace (this header is included from inside it) so simd.h and the
// -inl.h headers share one definition.
namespace ss = std::simd;

using v8 = ss::vec<std::uint8_t, SIMDUTF_STDSIMD_VEC_BYTES>;
using m8 = ss::mask<std::uint8_t, SIMDUTF_STDSIMD_VEC_BYTES>;
using v16 = ss::vec<std::uint16_t, SIMDUTF_STDSIMD_VEC_BYTES / 2>;
using m16 = ss::mask<std::uint16_t, SIMDUTF_STDSIMD_VEC_BYTES / 2>;
using v32 = ss::vec<std::uint32_t, SIMDUTF_STDSIMD_VEC_BYTES / 4>;
using m32 = ss::mask<std::uint32_t, SIMDUTF_STDSIMD_VEC_BYTES / 4>;
using v64 = ss::vec<std::uint64_t, SIMDUTF_STDSIMD_VEC_BYTES / 8>;

// ---- pull in the matching per-tier implementation -----------------------
#if SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_SSE
  #include "simdutf/stdsimd/tier_ops_sse.h"
#elif SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX2
  #include "simdutf/stdsimd/tier_ops_avx2.h"
#elif SIMDUTF_STDSIMD_TIER == SIMDUTF_STDSIMD_TIER_AVX512
  #include "simdutf/stdsimd/tier_ops_avx512.h"
#else
  #error "Unknown SIMDUTF_STDSIMD_TIER"
#endif

#endif // SIMDUTF_STDSIMD_TIER_OPS_H
