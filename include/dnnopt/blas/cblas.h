#pragma once
/// @file cblas.h
/// Standard CBLAS interface backed by DNN-Opt optimized GEMM.
///
/// Drop-in replacement for OpenBLAS/MKL cblas.h.
/// Build as libdnnopt_blas.so, then:
///   LD_PRELOAD=libdnnopt_blas.so ./your_program

#ifdef __cplusplus
extern "C" {
#endif

/// CBLAS matrix order.
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

/// CBLAS transpose operation.
enum CBLAS_TRANSPOSE {
    CblasNoTrans   = 111,
    CblasTrans     = 112,
    CblasConjTrans = 113
};

/// Single-precision general matrix multiply.
/// C = alpha * op(A) * op(B) + beta * C
/// where op(X) = X or X^T depending on TransX.
void cblas_sgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K,
                 float alpha,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float beta,
                 float *C, int ldc);

/// Thread control (OpenBLAS-compatible).
void openblas_set_num_threads(int num_threads);
int  openblas_get_num_threads(void);

/// Alternate thread control names used by various BLAS libraries.
void blas_set_num_threads(int num_threads);
void omp_set_num_threads_blas(int num_threads);

#ifdef __cplusplus
}
#endif
