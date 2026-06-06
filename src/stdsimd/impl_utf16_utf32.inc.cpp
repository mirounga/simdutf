// ==== utf16 <-> utf32 conversions family (REAL PORT) ====
//
// This partial holds the UTF-16 <-> UTF-32 conversions for the stdsimd backend.
// It is #included into stdsimd/implementation.cpp inside
//   namespace simdutf { namespace SIMDUTF_IMPLEMENTATION { ... } }
// and inside the AVX2 target region applied by stdsimd/begin.h.
//
// REAL PORT strategy (mirrors haswell/implementation.cpp):
//   * There is no backend-agnostic generic header for either transcoding
//     direction here. Each direction tier-selects the existing arch kernels and
//     aliases them to a uniform stdsimd_ name (movemask / cmpeq / 16<->32
//     narrow/widen / per-128-lane byte swap have no portable std::simd form):
//       - utf16 -> utf32: westmere(128) on the SSE tier, haswell(256) on the
//         AVX2/AVX512 tiers. Both arch kernels are pure x86 intrinsics +
//         scalar::utf16::swap_if_needed / result / error_code, so this
//         direction runs as REAL SIMD on EVERY tier.
//       - utf32 -> utf16: haswell(256) on the AVX2/AVX512 tiers (real SIMD).
//         The westmere SSE kernel drives surrogate expansion through the full
//         westmere simd:: library (simd8/simd32 helpers the stdsimd simd::
//         namespace does not expose), so the SSE tier keeps utf32 -> utf16 on
//         the scalar path. See stdsimd/convert_utf32_to_utf16.cpp.
//   * The valid_ variants delegate to the non-valid ones, exactly like haswell.
//
// Because this partial is textually included while
// simdutf::SIMDUTF_IMPLEMENTATION is already open, we briefly CLOSE those two
// namespaces, emit the kernel sources at file scope inside the backend's
// anonymous namespace, then REOPEN the namespaces so the rest of
// implementation.cpp continues unchanged. The namespace close/reopen is
// UNCONDITIONAL so the parent file's brace balance never depends on a feature
// macro.

// ---- leave simdutf::SIMDUTF_IMPLEMENTATION so the included sources nest at
// ---- file scope. ------------------------------------------------------------
} // namespace SIMDUTF_IMPLEMENTATION (temporarily)
} // namespace simdutf (temporarily)

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

// --- anonymous-namespace sources (tier-selected intrinsic kernels) -----------
// Each wrapper tier-selects the width-specific arch kernel (westmere 128-bit /
// haswell 256-bit) and aliases it to a uniform stdsimd_ name. They have no
// portable std::simd form. Emitted in the backend's anonymous namespace,
// exactly like the utf8_utf16 family does for its kernels. The wrappers
// self-tier-select, so they are included on ALL tiers.
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
using namespace simd;

  #include "stdsimd/convert_utf16_to_utf32.cpp"
  #include "stdsimd/convert_utf32_to_utf16.cpp"
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

// clang-format off
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
// clang-format on

// ---- reopen simdutf::SIMDUTF_IMPLEMENTATION for the rest of the TU. ----------
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

// ===========================================================================
// utf16 -> utf32  (REAL SIMD on ALL tiers: tier-selected westmere(128) /
// haswell(256) kernel via stdsimd_convert_utf16_to_utf32 + scalar tail)
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf16le_to_utf32(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  std::pair<const char16_t *, char32_t *> ret =
      stdsimd_convert_utf16_to_utf32<endianness::LITTLE>(buf, len,
                                                         utf32_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf32_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_utf32::convert<endianness::LITTLE>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused size_t implementation::convert_utf16be_to_utf32(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  std::pair<const char16_t *, char32_t *> ret =
      stdsimd_convert_utf16_to_utf32<endianness::BIG>(buf, len, utf32_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf32_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf16_to_utf32::convert<endianness::BIG>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result implementation::convert_utf16le_to_utf32_with_errors(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char32_t *> ret =
      stdsimd_convert_utf16_to_utf32_with_errors<endianness::LITTLE>(
          buf, len, utf32_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_utf32::convert_with_errors<endianness::LITTLE>(
            buf + ret.first.count, len - ret.first.count, ret.second);
    if (scalar_res.error) {
      scalar_res.count += ret.first.count;
      return scalar_res;
    } else {
      ret.second += scalar_res.count;
    }
  }
  ret.first.count =
      ret.second -
      utf32_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused result implementation::convert_utf16be_to_utf32_with_errors(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char32_t *> ret =
      stdsimd_convert_utf16_to_utf32_with_errors<endianness::BIG>(
          buf, len, utf32_output);
  if (ret.first.error) {
    return ret.first;
  } // Can return directly since scalar fallback already found correct
    // ret.first.count
  if (ret.first.count != len) { // All good so far, but not finished
    result scalar_res =
        scalar::utf16_to_utf32::convert_with_errors<endianness::BIG>(
            buf + ret.first.count, len - ret.first.count, ret.second);
    if (scalar_res.error) {
      scalar_res.count += ret.first.count;
      return scalar_res;
    } else {
      ret.second += scalar_res.count;
    }
  }
  ret.first.count =
      ret.second -
      utf32_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused size_t implementation::convert_valid_utf16le_to_utf32(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  return convert_utf16le_to_utf32(buf, len, utf32_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf16be_to_utf32(
    const char16_t *buf, size_t len, char32_t *utf32_output) const noexcept {
  return convert_utf16be_to_utf32(buf, len, utf32_output);
}

// ===========================================================================
// utf32 -> utf16  (tier-selected kernel + scalar tail): REAL SIMD on all tiers.
//   AVX2/AVX512 use haswell's 256-bit kernel; SSE uses a pure-128-bit kernel
//   (both via stdsimd_convert_utf32_to_utf16; see convert_utf32_to_utf16.cpp).
// ===========================================================================
simdutf_warn_unused size_t implementation::convert_utf32_to_utf16le(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  std::pair<const char32_t *, char16_t *> ret =
      stdsimd_convert_utf32_to_utf16<endianness::LITTLE>(buf, len,
                                                         utf16_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf16_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf32_to_utf16::convert<endianness::LITTLE>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused size_t implementation::convert_utf32_to_utf16be(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  std::pair<const char32_t *, char16_t *> ret =
      stdsimd_convert_utf32_to_utf16<endianness::BIG>(buf, len, utf16_output);
  if (ret.first == nullptr) {
    return 0;
  }
  size_t saved_bytes = ret.second - utf16_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes =
        scalar::utf32_to_utf16::convert<endianness::BIG>(
            ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) {
      return 0;
    }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused result implementation::convert_utf32_to_utf16le_with_errors(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char16_t *> ret =
      stdsimd_convert_utf32_to_utf16_with_errors<endianness::LITTLE>(
          buf, len, utf16_output);
  if (ret.first.count != len) {
    result scalar_res =
        scalar::utf32_to_utf16::convert_with_errors<endianness::LITTLE>(
            buf + ret.first.count, len - ret.first.count, ret.second);
    if (scalar_res.error) {
      scalar_res.count += ret.first.count;
      return scalar_res;
    } else {
      ret.second += scalar_res.count;
    }
  }
  ret.first.count =
      ret.second -
      utf16_output; // Set count to the number of 8-bit code units written
  return ret.first;
}

simdutf_warn_unused result implementation::convert_utf32_to_utf16be_with_errors(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  // ret.first.count is always the position in the buffer, not the number of
  // code units written even if finished
  std::pair<result, char16_t *> ret =
      stdsimd_convert_utf32_to_utf16_with_errors<endianness::BIG>(
          buf, len, utf16_output);
  if (ret.first.count != len) {
    result scalar_res =
        scalar::utf32_to_utf16::convert_with_errors<endianness::BIG>(
            buf + ret.first.count, len - ret.first.count, ret.second);
    if (scalar_res.error) {
      scalar_res.count += ret.first.count;
      return scalar_res;
    } else {
      ret.second += scalar_res.count;
    }
  }
  ret.first.count =
      ret.second -
      utf16_output; // Set count to the number of 8-bit code units written
  return ret.first;
}


// utf32 -> utf16 valid_ variants delegate to the non-valid ones on every tier
// (exactly like haswell), so they pick up whichever path the tier selected.
simdutf_warn_unused size_t implementation::convert_valid_utf32_to_utf16le(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  return convert_utf32_to_utf16le(buf, len, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf32_to_utf16be(
    const char32_t *buf, size_t len, char16_t *utf16_output) const noexcept {
  return convert_utf32_to_utf16be(buf, len, utf16_output);
}

#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
