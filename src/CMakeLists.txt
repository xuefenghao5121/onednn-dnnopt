add_library(dnnopt_core STATIC
    hwcaps/arm_hwcaps.cpp
    utils/timer.cpp
    utils/aligned_alloc.cpp
    gemm/gemm.cpp
    gemm/gemm_ukernel_registry.cpp
    gemm/gemm_driver_generic.cpp
    gemm/gemm_threading.cpp
    gemm/gemm_driver_fp32.cpp
    gemm/gemm_pack_fp32.cpp
    gemm/gemm_ukernel_fp32_neon.cpp
    gemm/gemm_smallm_fp32.cpp
    gemm/gemm_pack_bf16.cpp
    gemm/gemm_ukernel_bf16_neon.cpp
    gemm/gemm_driver_bf16.cpp
    gemm/gemm_pack_int8.cpp
    gemm/gemm_ukernel_int8_neon.cpp
    gemm/gemm_driver_int8.cpp
)

target_include_directories(dnnopt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================
# SVE kernels (conditional on compiler support)
# ============================================================
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)

# Test if SVE intrinsics are available
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-march=native")
check_cxx_source_compiles("
    #include <arm_sve.h>
    int main() { return (int)svcntw(); }
" DNNOPT_HAS_SVE)
set(CMAKE_REQUIRED_FLAGS "")

if(DNNOPT_HAS_SVE)
    message(STATUS "SVE support: YES")
    target_sources(dnnopt_core PRIVATE
        gemm/gemm_ukernel_fp32_sve.cpp
        gemm/gemm_ukernel_bf16_sve.cpp
        gemm/gemm_ukernel_int8_sve.cpp
    )
    # SVE files may need explicit -march=native to get SVE intrinsics
    set_source_files_properties(
        gemm/gemm_ukernel_fp32_sve.cpp
        gemm/gemm_ukernel_bf16_sve.cpp
        gemm/gemm_ukernel_int8_sve.cpp
        PROPERTIES COMPILE_FLAGS "-march=native"
    )
    target_compile_definitions(dnnopt_core PRIVATE DNNOPT_HAS_SVE=1)
else()
    message(STATUS "SVE support: NO (SVE kernels disabled)")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(dnnopt_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================
# SME kernels (opt-in, requires SME-capable assembler)
# ============================================================
option(DNNOPT_ENABLE_SME "Enable SME kernel compilation (requires assembler support)" OFF)

if(DNNOPT_ENABLE_SME)
    # Test if assembler supports SME directives
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_cxx_source_compiles("
        int main() {
            asm volatile(\".arch_extension sme\\n\\tsmstart sm\\n\\tsmstop sm\\n\\t\" ::: \"memory\");
            return 0;
        }
    " DNNOPT_ASM_HAS_SME)
    set(CMAKE_REQUIRED_FLAGS "")

    if(DNNOPT_ASM_HAS_SME)
        message(STATUS "SME support: YES (compile-only, needs SME hardware to run)")
        target_sources(dnnopt_core PRIVATE
            gemm/gemm_ukernel_fp32_sme.cpp
        )
        target_compile_definitions(dnnopt_core PRIVATE DNNOPT_HAS_SME=1)
    else()
        message(STATUS "SME support: NO (assembler does not support SME)")
    endif()
else()
    message(STATUS "SME support: DISABLED (use -DDNNOPT_ENABLE_SME=ON to enable)")
endif()
