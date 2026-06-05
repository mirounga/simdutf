// ==== base64 (stdsimd backend) ====
//
// This partial holds the base64 family for the stdsimd backend. It is #included
// into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace stdsimd { ... } }
// and inside the AVX2 target region.
//
// The SIMD building blocks (block64, compress, base64_decode_block*,
// avx2_encode_base64_impl, encode_base64, avx2_binary_length_from_base64) are
// adapted from haswell/avx2_base64.cpp into stdsimd/base64.cpp. Base64 is
// dominated by ops with no portable std::simd form (pshufb, multishift,
// movemask, thintable compress, lane-crossing alignr), so per the escape-hatch
// policy that file is expressed with x86 __m256i / _mm256_* intrinsics. The
// backend-agnostic driver generic/base64.h then consumes the block64 type
// VERBATIM, and generic/find.h provides the vectorized find() via the wrapper.
//
// NAMESPACE CHOREOGRAPHY (mirrors impl_validate_utf8.inc.cpp): haswell includes
// avx2_base64.cpp inside its anonymous backend namespace, then closes the
// backend namespaces and pulls in generic/base64.h + generic/find.h at FILE
// scope (those generic headers self-open
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { namespace { ... } } }
// so they MUST be at file scope, or they would nest as
//   simdutf::stdsimd::simdutf::stdsimd::... and fail to resolve simdutf::scalar).
// This partial is textually included while simdutf::SIMDUTF_IMPLEMENTATION is
// already open, so we briefly CLOSE those two namespaces, do the file-scope work,
// then REOPEN them for the implementation:: method definitions. The close/reopen
// is UNCONDITIONAL so the parent file's brace balance never depends on a macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the generic headers (which open
// ---- their own simdutf::SIMDUTF_IMPLEMENTATION) nest at file scope. ---------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

// The SIMD base64 building blocks in stdsimd/base64.cpp are an escape-hatch port
// of haswell/avx2_base64.cpp expressed entirely in 256-bit __m256i / _mm256_*
// intrinsics (pshufb / multishift / movemask / thintable compress have no
// portable std::simd form). Those intrinsics physically require the AVX2 ISA, so
// the SIMD base64 path is built ONLY for the AVX2 tier. The SSE (128-bit) and
// AVX512 (512-bit) tiers delegate the base64 family to scalar::base64 (exactly
// like the fallback backend); a dedicated 128-bit / 512-bit std::simd base64
// port is left as future work. See the per-method bodies below.
  #define SIMDUTF_STDSIMD_BASE64_SIMD SIMDUTF_STDSIMD_AVX2_KERNELS

#if SIMDUTF_FEATURE_BASE64 && SIMDUTF_STDSIMD_BASE64_SIMD

// SIMD building blocks (block64 + free helpers) consumed by generic/base64.h.
// base64.cpp opens NO namespace of its own (mirroring haswell/avx2_base64.cpp),
// so emit it inside the backend's anonymous namespace, exactly where haswell
// places it. It needs tables::base64::*, scalar::base64::*, and the
// bitmanipulation helpers (count_ones/trailing_zeroes/is_power_of_two), all of
// which are visible at this scope.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
  #include "stdsimd/base64.cpp"
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// Backend-agnostic decode driver: compress_decode_base64<...>(block64). Self-
// wraps in simdutf::SIMDUTF_IMPLEMENTATION::{anonymous}::base64, so it MUST be
// at file scope.
  #include "generic/base64.h"
#endif // SIMDUTF_FEATURE_BASE64 && SIMDUTF_STDSIMD_BASE64_SIMD

#if SIMDUTF_FEATURE_BASE64 || SIMDUTF_FEATURE_DETECT_ENCODING
// Vectorized find() (util::find) consumed by the find() methods below; uses the
// stdsimd wrapper's simd8x64/simd16x32 by UNQUALIFIED name, so it needs
// `using namespace simd;` in the enclosing anonymous namespace. On the AVX2/SSE
// tiers a preceding partial (e.g. the utf8 validator) already establishes that
// using-directive, but on the AVX512 tier those partials route to scalar and
// emit nothing, so we establish it explicitly here. Self-wraps in
// simdutf::SIMDUTF_IMPLEMENTATION::{anonymous}::util.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;
}
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf
  #include "generic/find.h"
#endif // SIMDUTF_FEATURE_BASE64 || SIMDUTF_FEATURE_DETECT_ENCODING

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ---------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_BASE64

#if SIMDUTF_STDSIMD_BASE64_SIMD
// ===== AVX2 tier: real SIMD base64 (escape-hatch port of avx2_base64.cpp). ===

simdutf_warn_unused result implementation::base64_to_binary(
    const char *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  if (options & base64_default_or_url) {
    if (options == base64_options::base64_default_or_url_accept_garbage) {
      return base64::compress_decode_base64<false, true, true>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, true>(
          output, input, length, options, last_chunk_options);
    }
  } else if (options & base64_url) {
    if (options == base64_options::base64_url_accept_garbage) {
      return base64::compress_decode_base64<true, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<true, false, false>(
          output, input, length, options, last_chunk_options);
    }
  } else {
    if (options == base64_options::base64_default_accept_garbage) {
      return base64::compress_decode_base64<false, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, false>(
          output, input, length, options, last_chunk_options);
    }
  }
}

simdutf_warn_unused full_result implementation::base64_to_binary_details(
    const char *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  if (options & base64_default_or_url) {
    if (options == base64_options::base64_default_or_url_accept_garbage) {
      return base64::compress_decode_base64<false, true, true>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, true>(
          output, input, length, options, last_chunk_options);
    }
  } else if (options & base64_url) {
    if (options == base64_options::base64_url_accept_garbage) {
      return base64::compress_decode_base64<true, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<true, false, false>(
          output, input, length, options, last_chunk_options);
    }
  } else {
    if (options == base64_options::base64_default_accept_garbage) {
      return base64::compress_decode_base64<false, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, false>(
          output, input, length, options, last_chunk_options);
    }
  }
}

simdutf_warn_unused result implementation::base64_to_binary(
    const char16_t *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  if (options & base64_default_or_url) {
    if (options == base64_options::base64_default_or_url_accept_garbage) {
      return base64::compress_decode_base64<false, true, true>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, true>(
          output, input, length, options, last_chunk_options);
    }
  } else if (options & base64_url) {
    if (options == base64_options::base64_url_accept_garbage) {
      return base64::compress_decode_base64<true, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<true, false, false>(
          output, input, length, options, last_chunk_options);
    }
  } else {
    if (options == base64_options::base64_default_accept_garbage) {
      return base64::compress_decode_base64<false, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, false>(
          output, input, length, options, last_chunk_options);
    }
  }
}

simdutf_warn_unused full_result implementation::base64_to_binary_details(
    const char16_t *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  if (options & base64_default_or_url) {
    if (options == base64_options::base64_default_or_url_accept_garbage) {
      return base64::compress_decode_base64<false, true, true>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, true>(
          output, input, length, options, last_chunk_options);
    }
  } else if (options & base64_url) {
    if (options == base64_options::base64_url_accept_garbage) {
      return base64::compress_decode_base64<true, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<true, false, false>(
          output, input, length, options, last_chunk_options);
    }
  } else {
    if (options == base64_options::base64_default_accept_garbage) {
      return base64::compress_decode_base64<false, true, false>(
          output, input, length, options, last_chunk_options);
    } else {
      return base64::compress_decode_base64<false, false, false>(
          output, input, length, options, last_chunk_options);
    }
  }
}

size_t implementation::binary_to_base64(const char *input, size_t length,
                                        char *output,
                                        base64_options options) const noexcept {
  if (options & base64_url) {
    return encode_base64<true>(output, input, length, options);
  } else {
    return encode_base64<false>(output, input, length, options);
  }
}

size_t implementation::binary_to_base64_with_lines(
    const char *input, size_t length, char *output, size_t line_length,
    base64_options options) const noexcept {
  if (options & base64_url) {
    return avx2_encode_base64_impl<true, true>(output, input, length, options,
                                               line_length);
  } else {
    return avx2_encode_base64_impl<false, true>(output, input, length, options,
                                                line_length);
  }
}

// NOTE: the find() family was moved to stdsimd/impl_find.inc.cpp so the find
// Port agent owns a dedicated partial; it is included separately from
// implementation.cpp.

simdutf_warn_unused size_t implementation::binary_length_from_base64(
    const char *input, size_t length) const noexcept {
  return avx2_binary_length_from_base64(input, length);
}

simdutf_warn_unused size_t implementation::binary_length_from_base64(
    const char16_t *input, size_t length) const noexcept {
  return avx2_binary_length_from_base64(input, length);
}

#else // !SIMDUTF_STDSIMD_BASE64_SIMD
// ===== SSE / AVX512 tiers: delegate the base64 family to scalar (same as the
// ===== fallback backend). A 128-bit / 512-bit std::simd port is future work. ==

simdutf_warn_unused result implementation::base64_to_binary(
    const char *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  return scalar::base64::base64_to_binary_details_impl(
      input, length, output, options, last_chunk_options);
}

simdutf_warn_unused full_result implementation::base64_to_binary_details(
    const char *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  return scalar::base64::base64_to_binary_details_impl(
      input, length, output, options, last_chunk_options);
}

simdutf_warn_unused result implementation::base64_to_binary(
    const char16_t *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  return scalar::base64::base64_to_binary_details_impl(
      input, length, output, options, last_chunk_options);
}

simdutf_warn_unused full_result implementation::base64_to_binary_details(
    const char16_t *input, size_t length, char *output, base64_options options,
    last_chunk_handling_options last_chunk_options) const noexcept {
  return scalar::base64::base64_to_binary_details_impl(
      input, length, output, options, last_chunk_options);
}

size_t implementation::binary_to_base64(const char *input, size_t length,
                                        char *output,
                                        base64_options options) const noexcept {
  return scalar::base64::tail_encode_base64(output, input, length, options);
}

size_t implementation::binary_to_base64_with_lines(
    const char *input, size_t length, char *output, size_t line_length,
    base64_options options) const noexcept {
  return scalar::base64::tail_encode_base64_impl<true>(output, input, length,
                                                       options, line_length);
}

simdutf_warn_unused size_t implementation::binary_length_from_base64(
    const char *input, size_t length) const noexcept {
  return scalar::base64::binary_length_from_base64(input, length);
}

simdutf_warn_unused size_t implementation::binary_length_from_base64(
    const char16_t *input, size_t length) const noexcept {
  return scalar::base64::binary_length_from_base64(input, length);
}

#endif // SIMDUTF_STDSIMD_BASE64_SIMD
#endif // SIMDUTF_FEATURE_BASE64

#undef SIMDUTF_STDSIMD_BASE64_SIMD
