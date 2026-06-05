
option(SIMDUTF_SANITIZE "Sanitize addresses" OFF)
option(SIMDUTF_SANITIZE_UNDEFINED "Sanitize undefined behavior" OFF)
option(SIMDUTF_SANITIZE_THREAD "Sanitize threads" OFF)
option(SIMDUTF_ALWAYS_INCLUDE_FALLBACK "Always include fallback" OFF)
option(SIMDUTF_IMPLEMENTATION_STDSIMD "C++26 std::simd backend (GCC16+)" OFF)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  if(SIMDUTF_SANITIZE OR SIMDUTF_SANITIZE_UNDEFINED)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    # SIMDUTF_SANITIZE only applies to gcc/clang:
    message(STATUS "Setting debug optimization flag to -O1 -g.")
    set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g" CACHE STRING "" FORCE)
  else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  endif()
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake")

# simdutf requires C++17 or later.
set(SIMDUTF_CXX_STANDARD 17 CACHE STRING "the C++ standard to use for simdutf")
if(SIMDUTF_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "simdutf requires C++17 or later (got ${SIMDUTF_CXX_STANDARD}).")
endif()

# The additive C++26 std::simd backend ("stdsimd") is only buildable with
# GCC 16+ and the C++26 standard. If the option is ON but the toolchain cannot
# support it, warn and turn it back off rather than fail the configure.
if(SIMDUTF_IMPLEMENTATION_STDSIMD)
  if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16))
    message(WARNING
      "SIMDUTF_IMPLEMENTATION_STDSIMD requires GCC 16 or later "
      "(found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}); "
      "disabling the stdsimd backend.")
    set(SIMDUTF_IMPLEMENTATION_STDSIMD OFF CACHE BOOL
      "C++26 std::simd backend (GCC16+)" FORCE)
  else()
    # The C++26 <simd> header requires building at C++26.
    if(SIMDUTF_CXX_STANDARD LESS 26)
      message(STATUS
        "SIMDUTF_IMPLEMENTATION_STDSIMD is ON: bumping SIMDUTF_CXX_STANDARD to 26.")
      set(SIMDUTF_CXX_STANDARD 26 CACHE STRING
        "the C++ standard to use for simdutf" FORCE)
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD ${SIMDUTF_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
