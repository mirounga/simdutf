#include "simdutf/stdsimd/begin.h"

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

// The target families each live in their own partial include to avoid merge
// races while the real ports land independently. The first three are REAL
// PORTS; the remaining ten are currently STUBBED to delegate to scalar:: (or
// util::) just like the fallback backend until their real ports land. Each
// partial preserves the exact #if SIMDUTF_FEATURE_* guards of the methods it
// owns. NOTHING is defined inline in this file anymore -- every implementation::
// method lives in exactly one partial.
#include "stdsimd/impl_validate_utf8.inc.cpp"
#include "stdsimd/impl_utf8_utf16.inc.cpp"
#include "stdsimd/impl_base64.inc.cpp"

#include "stdsimd/impl_ascii.inc.cpp"
#include "stdsimd/impl_validate_utf16.inc.cpp"
#include "stdsimd/impl_validate_utf32.inc.cpp"
#include "stdsimd/impl_utf16fix.inc.cpp"
#include "stdsimd/impl_latin1.inc.cpp"
#include "stdsimd/impl_utf8_utf32.inc.cpp"
#include "stdsimd/impl_utf16_utf32.inc.cpp"
#include "stdsimd/impl_count.inc.cpp"
#include "stdsimd/impl_find.inc.cpp"
#include "stdsimd/impl_detect.inc.cpp"

} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#include "simdutf/stdsimd/end.h"
