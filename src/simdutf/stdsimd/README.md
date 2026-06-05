# `stdsimd` backend — C++26 `std::simd` (GCC 16+)

An **experimental, additive** simdutf backend whose SIMD kernels are written against
C++26 `std::simd` (`<simd>`) instead of per-architecture intrinsics. It ships as
**three runtime-selectable tiers**, each compiled in its own translation unit with a
per-tier `-march` baseline and a pinned `std::simd` vector width:

| Tier | name | vec | TU flags | runtime ISA |
|---|---|---|---|---|
| SSE | `stdsimd_sse` | `vec<u8,16>` | `-msse4.2` | SSE4.2 |
| AVX2 | `stdsimd_avx2` | `vec<u8,32>` | `-mavx2 -mbmi -mbmi2` | AVX2+BMI |
| AVX-512 | `stdsimd_avx512` | `vec<u8,64>` | `-mavx512f/bw/cd/dq/vl/vbmi/vbmi2` | AVX-512 |

## Status

- **Off by default** (`-DSIMDUTF_IMPLEMENTATION_STDSIMD=ON`; GCC ≥ 16, C++26). Self-
  disables on any other toolchain.
- **Additive & FORCE-only.** All three tiers register below the tuned backends, so
  auto-detection never selects them. Reach a tier with
  `SIMDUTF_FORCE_IMPLEMENTATION=stdsimd_{sse,avx2,avx512}`.
- The full suite passes forced to each tier (modulo the two known packaging
  artifacts, `amalgamation_demo` / `nostdlibcxx`, which can't host `std::simd`).

## Per-tier feature coverage

The wrapper (`simd.h` + `simd16/32/64-inl.h`) is width-parameterized and routes the
few non-portable ops through `tier_ops.h` (per-tier impl in
`tier_ops_{sse,avx2,avx512}.h`). On top of that:

- **Fully `std::simd` on EVERY tier** (wrapper-only, no arch kernel): `validate_utf8`,
  `validate_ascii`, `validate_utf16`, `validate_utf32`, `count_*`, `find`, `detect`.
  These run as real SIMD at SSE/AVX2/AVX512 widths. (e.g. AVX-512 `validate_utf8`
  ~26 GB/s, matching/beating icelake.)
- **AVX2 tier only** (256-bit hand-adapted kernels): the transcoders that call a
  per-block masked kernel — `utf8↔utf16/utf32`, all `latin1` conversions,
  `utf16↔utf32`, `utf16fix`, and `base64`. On the SSE and AVX-512 tiers these
  families currently delegate to the scalar reference. Porting the masked per-block
  kernels (`convert_masked_utf8_to_*`, `avx2_convert_*`, `avx2_base64`) to native
  128-bit / 512-bit `std::simd` is the remaining work for full per-tier SIMD.

## Design notes

- **ABI / dedicated TU (critical).** `std::simd`'s vector ABI follows the TU's
  command-line `-march`, **not** the `target(...)` pragma. Each tier is therefore a
  separate TU (`simdutf_stdsimd_{sse,avx2,avx512}.cpp`, each re-including
  `simdutf.cpp` with `SIMDUTF_STDSIMD_ONLY` + the tier's `SIMDUTF_STDSIMD_VEC_BYTES`)
  built with the matching `-march`, giving the vec the XMM/YMM/ZMM register ABI.
  Without this the vec is passed through memory (~16× slower).
- **Mask model.** `mask<u8,N>` is a `__mmask` register on AVX-512, so the wrapper
  keeps `simd8<bool>` as a 0xFF/0x00 byte vector everywhere; only `to_bitmask`
  differs per tier (`movemask` vs `movepi8_mask`).
- **Shared-algorithm width fixes.** The generic UTF-8 validator gained a
  `NUM_CHUNKS==1` path (for the single-64-byte-chunk AVX-512 tier), `max_array` was
  widened to 64 bytes (the tail-load was width-dependent), and `repeat_16` now
  broadcasts the 16-byte lookup table across all 128-bit lanes. These are additive
  and behaviour-preserving for the existing 16/32-byte backends.
