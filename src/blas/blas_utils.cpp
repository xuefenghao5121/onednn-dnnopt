/// @file blas_utils.cpp
/// Thread control and utility functions for BLAS-compatible interface.
/// Provides OpenBLAS-compatible thread control API.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/gemm/gemm.h"

extern "C" {

void openblas_set_num_threads(int num_threads) {
    dnnopt::gemm_set_num_threads(num_threads);
}

int openblas_get_num_threads(void) {
    return dnnopt::gemm_get_num_threads();
}

void blas_set_num_threads(int num_threads) {
    dnnopt::gemm_set_num_threads(num_threads);
}

void omp_set_num_threads_blas(int num_threads) {
    dnnopt::gemm_set_num_threads(num_threads);
}

}  // extern "C"
