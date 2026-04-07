#pragma once
/// @file gemm_threading.h
/// Thread control API for GEMM operations.

namespace dnnopt {

/// Set the number of threads for GEMM operations.
/// n=0 means auto (use all available cores).
/// n=1 disables threading.
void gemm_set_num_threads(int n);

/// Get the current thread count for GEMM operations.
int gemm_get_num_threads();

}  // namespace dnnopt
