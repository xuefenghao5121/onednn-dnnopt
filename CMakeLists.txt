cmake_minimum_required(VERSION 3.16)
project(dnn-opt
    VERSION 0.1.0
    LANGUAGES C CXX ASM
    DESCRIPTION "oneDNN ARM Platform Optimization Toolkit"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Architecture detection
# ============================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(DNNOPT_AARCH64 ON)
    message(STATUS "Target architecture: AArch64")
else()
    message(WARNING "This project targets AArch64. Current arch: ${CMAKE_SYSTEM_PROCESSOR}")
    set(DNNOPT_AARCH64 OFF)
endif()

# ============================================================
# Build options
# ============================================================
option(DNNOPT_BUILD_BENCHMARKS "Build benchmark suite" ON)
option(DNNOPT_BUILD_TESTS "Build regression tests" ON)
option(DNNOPT_USE_OPENMP "Enable OpenMP" ON)
option(DNNOPT_NATIVE_ARCH "Use -march=native" ON)

# ============================================================
# Compiler flags
# ============================================================
if(DNNOPT_AARCH64)
    if(DNNOPT_NATIVE_ARCH)
        set(ARCH_FLAGS "-march=native")
    else()
        # Minimum target: ARMv8.2-a with common extensions
        set(ARCH_FLAGS "-march=armv8.2-a+dotprod+fp16")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${ARCH_FLAGS}")
endif()

# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# OpenMP
# ============================================================
if(DNNOPT_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, disabling")
        set(DNNOPT_USE_OPENMP OFF)
    endif()
endif()

# ============================================================
# Source subdirectories
# ============================================================
# Core library: hardware detection + utilities
add_subdirectory(src)

if(DNNOPT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(DNNOPT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "=== DNN-Opt Build Configuration ===")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  AArch64:       ${DNNOPT_AARCH64}")
message(STATUS "  Arch flags:    ${ARCH_FLAGS}")
message(STATUS "  OpenMP:        ${DNNOPT_USE_OPENMP}")
message(STATUS "  Benchmarks:    ${DNNOPT_BUILD_BENCHMARKS}")
message(STATUS "  Tests:         ${DNNOPT_BUILD_TESTS}")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "===================================")
