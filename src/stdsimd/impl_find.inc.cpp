// ==== find family (REAL PORT) ====
//
// This partial holds the find() methods for the stdsimd backend. It is
// #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
//
// REAL PORT: this family reuses src/generic/find.h VERBATIM, exactly like
// haswell/implementation.cpp (which wires generic/find.h under
// SIMDUTF_FEATURE_BASE64 and calls util::find). util::find is the vectorized
// implementation built on the stdsimd wrapper's simd8x64<uint8_t> /
// simd16x32<uint16_t> and their .eq() reductions; it is NOT the scalar
// fallback. The generic header self-wraps in
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { util } } }
// so it must be pulled in at FILE scope. It is already included (once per TU)
// by impl_base64.inc.cpp under the same feature gate, so these method bodies
// simply forward to that vectorized util::find. The bodies were moved out of
// impl_base64.inc.cpp so the find family owns one file.

const char *implementation::find(const char *start, const char *end,
                                 char character) const noexcept {
  return util::find(start, end, character);
}

const char16_t *implementation::find(const char16_t *start, const char16_t *end,
                                     char16_t character) const noexcept {
  return util::find(start, end, character);
}
