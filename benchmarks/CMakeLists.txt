add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE dnnopt_core)

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(bench_conv PRIVATE dnnopt_core)

add_executable(bench_elementwise bench_elementwise.cpp)
target_link_libraries(bench_elementwise PRIVATE dnnopt_core)

add_executable(bench_pooling bench_pooling.cpp)
target_link_libraries(bench_pooling PRIVATE dnnopt_core)

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
