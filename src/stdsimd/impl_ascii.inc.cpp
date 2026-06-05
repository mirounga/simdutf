// ==== ascii validation family (REAL PORT) ====
//
// This partial holds the ASCII-validation methods for the stdsimd backend. It
// is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp):
//   * validate_ascii / validate_ascii_with_errors reuse
//     src/generic/ascii_validation.h VERBATIM. That header drives the whole
//     check off simd::simd8x64<uint8_t>::is_ascii(), which is already proven in
//     the stdsimd backend (validate_utf8 uses the same reduce_or().is_ascii()).
//   * validate_utf16{le,be}_as_ascii: haswell delegates these to
//     generic/validate_utf16.h's validate_utf16_as_ascii_with_errors<>(). We do
//     NOT pull that whole header in here, because it ALSO defines the full
//     UTF-16 surrogate validator, which depends on a backend-specific
//     utf16_gather_high_bytes() helper (haswell supplies it from
//     avx2_validate_utf16.cpp). That helper -- and the surrogate validator that
//     uses it -- belong to the (still-stubbed) validate_utf16 family, not ascii.
//     So we re-implement the self-contained as-ASCII scan inline below using the
//     exact same algorithm (simd16x32<uint16_t>::lteq(0x7f) + scalar tail),
//     yielding identical output to validate_utf16_as_ascii_with_errors<>().
//
// generic/ascii_validation.h self-wraps in
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ... } } }
// so it must be pulled in at FILE scope. Because this partial is textually
// included while simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly
// CLOSE those two namespaces, do the file-scope work (the generic header), then
// REOPEN the namespaces so the rest of implementation.cpp continues unchanged.
// The namespace close/reopen is UNCONDITIONAL so the parent file's brace
// balance never depends on a feature macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the generic header (which opens
// ---- its own simdutf::SIMDUTF_IMPLEMENTATION) nests at file scope. ----------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if SIMDUTF_FEATURE_ASCII
// buf_block_reader.h has no include guard and is meant to be pulled in exactly
// once per backend translation unit. Guard with a TU-local sentinel so other
// family partials can include it too without colliding.
  #ifndef SIMDUTF_STDSIMD_BUF_BLOCK_READER_INCLUDED
    #define SIMDUTF_STDSIMD_BUF_BLOCK_READER_INCLUDED 1
    #include "generic/buf_block_reader.h"
  #endif

  #include "generic/ascii_validation.h"
#endif // SIMDUTF_FEATURE_ASCII

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ---------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_ASCII
simdutf_warn_unused bool
implementation::validate_ascii(const char *buf, size_t len) const noexcept {
  return SIMDUTF_IMPLEMENTATION::ascii_validation::generic_validate_ascii(buf,
                                                                          len);
}

simdutf_warn_unused result implementation::validate_ascii_with_errors(
    const char *buf, size_t len) const noexcept {
  return SIMDUTF_IMPLEMENTATION::ascii_validation::
      generic_validate_ascii_with_errors(buf, len);
}
#endif // SIMDUTF_FEATURE_ASCII

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_ASCII
namespace {
// Self-contained "is this UTF-16 buffer entirely ASCII (<= 0x7F)?" scan. This
// reproduces generic/validate_utf16.h's validate_utf16_as_ascii_with_errors<>()
// without dragging in the full UTF-16 surrogate validator (which is the
// validate_utf16 family's responsibility). 32 code units per SIMD block via
// simd16x32<uint16_t>::lteq(); scalar tail for the remainder.
template <endianness big_endian>
simdutf_really_inline bool generic_validate_utf16_as_ascii(const char16_t *input,
                                                           size_t size) {
  using namespace simd;
  if (simdutf_unlikely(size == 0)) {
    return true;
  }
  size_t pos = 0;
  for (; pos < size / 32 * 32; pos += 32) {
    simd16x32<uint16_t> input_vec(
        reinterpret_cast<const uint16_t *>(input + pos));
    if constexpr (!match_system(big_endian)) {
      input_vec.swap_bytes();
    }
    uint64_t matches = input_vec.lteq(uint16_t(0x7f));
    if (~matches) {
      return false;
    }
  }
  while (pos < size) {
    char16_t v = scalar::utf16::swap_if_needed<big_endian>(input[pos]);
    if (v > 0x7F) {
      return false;
    }
    pos++;
  }
  return true;
}
} // unnamed namespace

simdutf_warn_unused bool
implementation::validate_utf16le_as_ascii(const char16_t *buf,
                                          size_t len) const noexcept {
  return generic_validate_utf16_as_ascii<endianness::LITTLE>(buf, len);
}

simdutf_warn_unused bool
implementation::validate_utf16be_as_ascii(const char16_t *buf,
                                          size_t len) const noexcept {
  return generic_validate_utf16_as_ascii<endianness::BIG>(buf, len);
}
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_ASCII
