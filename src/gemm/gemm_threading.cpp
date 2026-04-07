/// @file gemm_threading.cpp
/// Thread control for GEMM operations.

#include "dnnopt/gemm/gemm_threading.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dnnopt {

static int g_gemm_num_threads = 0;  // 0 = auto

void gemm_set_num_threads(int n) {
    g_gemm_num_threads = (n < 0) ? 0 : n;
}

int gemm_get_num_threads() {
    if (g_gemm_num_threads > 0) return g_gemm_num_threads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace dnnopt
