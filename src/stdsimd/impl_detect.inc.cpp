// ==== detect_encodings + sweep family (STUB -> scalar:: / self) ====
//
// This partial holds detect_encodings plus the SWEEP of any remaining
// implementation:: methods that are still scalar-delegated and are not owned by
// another family partial:
//   * change_endianness_utf16
//   * convert_utf16le_to_utf8_with_replacement
//   * convert_utf16be_to_utf8_with_replacement
// It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and currently delegates to scalar:: (detect_encodings drives the backend's
// own validate_* methods) just like the fallback backend until the real ports
// land. The #if SIMDUTF_FEATURE_* guards mirror the originals.
//
// NOTE: autodetect_encoding is NOT defined per-backend; it lives once in the
// shared src/implementation.cpp (base class default), so it is intentionally
// absent here.

#if SIMDUTF_FEATURE_DETECT_ENCODING
simdutf_warn_unused int
implementation::detect_encodings(const char *input,
                                 size_t length) const noexcept {
  // If there is a BOM, then we trust it.
  auto bom_encoding = simdutf::BOM::check_bom(input, length);
  if (bom_encoding != encoding_type::unspecified) {
    return bom_encoding;
  }
  int out = 0;
  // todo: reimplement as a one-pass algorithm.
  if (validate_utf8(input, length)) {
    out |= encoding_type::UTF8;
  }
  if ((length % 2) == 0) {
    if (validate_utf16le(reinterpret_cast<const char16_t *>(input),
                         length / 2)) {
      out |= encoding_type::UTF16_LE;
    }
  }
  if ((length % 4) == 0) {
    if (validate_utf32(reinterpret_cast<const char32_t *>(input), length / 4)) {
      out |= encoding_type::UTF32_LE;
    }
  }
  return out;
}
#endif // SIMDUTF_FEATURE_DETECT_ENCODING

#if SIMDUTF_FEATURE_UTF16
void implementation::change_endianness_utf16(const char16_t *input,
                                             size_t length,
                                             char16_t *output) const noexcept {
  scalar::utf16::change_endianness_utf16(input, length, output);
}
#endif // SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
simdutf_warn_unused size_t
implementation::convert_utf16le_to_utf8_with_replacement(
    const char16_t *input, size_t length, char *utf8_buffer) const noexcept {
  return scalar::utf16_to_utf8::convert_with_replacement<endianness::LITTLE>(
      input, length, utf8_buffer);
}

simdutf_warn_unused size_t
implementation::convert_utf16be_to_utf8_with_replacement(
    const char16_t *input, size_t length, char *utf8_buffer) const noexcept {
  return scalar::utf16_to_utf8::convert_with_replacement<endianness::BIG>(
      input, length, utf8_buffer);
}
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
