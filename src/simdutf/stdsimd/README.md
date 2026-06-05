# `stdsimd` backend — C++26 `std::simd` (GCC 16+)

An **experimental, additive** simdutf backend whose SIMD kernels are written once
against C++26 `std::simd` (`<simd>`) instead of per-architecture intrinsics. It is
the first slice of a longer-term effort to collapse the per-arch wrappers into a
single portable source.

## Status

- **Off by default.** Enable with `-DSIMDUTF_IMPLEMENTATION_STDSIMD=ON` (requires
  **GCC ≥ 16** and **C++26**). On any other toolchain the option self-disables and
  the build is unchanged.
- **Additive & FORCE-only.** It registers below every tuned backend, so runtime
  auto-detection never selects it. Reach it with
  `SIMDUTF_FORCE_IMPLEMENTATION=stdsimd`. The existing 9 backends are untouched.
- **AVX2 tier.** `std::simd::vec<uint8_t,32>` (width pinned to 32). Requires AVX2 at
  runtime (`required_instruction_sets = AVX2|BMI1|BMI2`).
- **Families implemented in `std::simd`:** `validate_utf8`, `utf8↔utf16` (LE/BE,
  both directions, incl. `with_errors`/`valid`), `base64` (encode/decode). Every
  other public method delegates to the scalar reference, so the backend is complete
  and correct (it passes the full suite forced to `stdsimd`).

## Design notes

- **Wrapper:** `simd.h`, `simd16/32/64-inl.h` replicate the haswell wrapper
  interface over `std::simd`, so `src/generic/*` algorithms compile against it
  verbatim. Policy is portable `std::simd` by default; a few ops with no portable
  form use guarded intrinsic escape hatches (`lookup_16`→`pshufb`, `prev<N>`→
  `alignr`, `to_bitmask`→`movemask`, `saturating_sub`→`subs_epu8`, BE swap) bridged
  by `std::bit_cast<__m256i>` (free — same 32-byte layout).
- **The ABI / dedicated TU (important).** `std::simd`'s vector ABI follows the TU's
  command-line `-march`, **not** the per-function `target("avx2")` pragma. Under the
  no-AVX baseline the 32-byte vec is passed through memory, which made the kernels
  ~16× slower. The backend is therefore compiled in its **own translation unit**
  (`src/simdutf_stdsimd.cpp`, which re-includes `simdutf.cpp` with
  `SIMDUTF_STDSIMD_ONLY=1`) using a `-mavx2` command-line baseline, giving the vec
  the YMM register ABI. We cannot put `-mavx2` on the whole library — it would
  VEX-encode the runtime dispatcher and SIGILL on pre-AVX CPUs.

## Performance (GCC 16, Ryzen AI 9 HX PRO 370, forced backends)

With the dedicated-TU ABI fix, the portable `std::simd` kernels match hand-written
AVX2 intrinsics:

| Procedure | stdsimd | haswell |
|---|---|---|
| `validate_utf8` | ~26.7 GB/s | ~19.5 |
| `validate_utf8_with_errors` | ~24 | ~26 |
| `convert_utf8_to_utf16le` | ~2.74 (~98%) | ~2.79 |
| `convert_valid_utf8_to_utf16le` | ~3.18 (~99%) | ~3.22 |

## Known limitations

- **Single-header / amalgamation** does not include the dedicated-TU machinery, so a
  single-header build that enables stdsimd gets the slow (memory-ABI) path. Use the
  CMake library build for the fast path.
- **`nostdlibcxx`** cannot host stdsimd (`std::simd` needs libstdc++); it is correctly
  absent there. Forcing `stdsimd` onto binaries that don't include it (e.g.
  `amalgamation_demo` built strict `-std=c++26`, `nostdlibcxx_c_api_test`) falls
  through to unsupported — an artifact of global force, not a defect. Default
  (auto-detect) runs of those binaries pass.
- Escape-hatch non-x86 arms (arm/scalar) are written but unexercised at this AVX2
  tier (`// TODO(verify)`), pending the cross-arch and multi-tier slices.
