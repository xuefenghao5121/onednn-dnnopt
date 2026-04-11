add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE dnnopt_core)

add_executable(bench_small_gemm bench_small_gemm.cpp)
target_link_libraries(bench_small_gemm PRIVATE dnnopt_core)

add_executable(bench_small_gemm_v2 bench_small_gemm_v2.cpp)
target_link_libraries(bench_small_gemm_v2 PRIVATE dnnopt_core)

add_executable(bench_threading bench_threading.cpp)
target_link_libraries(bench_threading PRIVATE dnnopt_core)

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(bench_conv PRIVATE dnnopt_core)

add_executable(bench_elementwise bench_elementwise.cpp)
target_link_libraries(bench_elementwise PRIVATE dnnopt_core)

add_executable(bench_pooling bench_pooling.cpp)
target_link_libraries(bench_pooling PRIVATE dnnopt_core)

if(DNNOPT_BUILD_BLAS)
    add_executable(bench_blas bench_blas.cpp)
    target_link_libraries(bench_blas PRIVATE dnnopt_blas ${CMAKE_DL_LIBS})

    add_executable(bench_small_vs_openblas bench_small_vs_openblas.cpp)
    target_link_libraries(bench_small_vs_openblas PRIVATE dnnopt_blas ${CMAKE_DL_LIBS})
endif()

add_executable(hwcaps_report hwcaps_report.cpp)
target_link_libraries(hwcaps_report PRIVATE dnnopt_core)

# Convenience target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND hwcaps_report
    COMMAND bench_gemm
    COMMAND bench_conv
    COMMAND bench_elementwise
    COMMAND bench_pooling
    COMMENT "Running all benchmarks"
    DEPENDS hwcaps_report bench_gemm bench_conv bench_elementwise bench_pooling
)

# Quick 4x16 kernel performance comparison
add_executable(bench_4x16_perf bench_4x16_perf.cpp)
target_link_libraries(bench_4x16_perf PRIVATE dnnopt_core)
