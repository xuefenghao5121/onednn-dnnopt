/// @file blas_fortran.cpp
/// Fortran BLAS sgemm_ interface: unpacks pointer arguments, delegates to cblas_sgemm.
/// Fortran uses column-major layout by convention.

#include "dnnopt/blas/cblas.h"

extern "C" {

/// Fortran-style SGEMM: all parameters passed by pointer, column-major layout.
/// This is the symbol that Fortran code and many BLAS consumers (including oneDNN)
/// call when using the Fortran BLAS interface.
void sgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const float* alpha,
            const float* a, const int* lda,
            const float* b, const int* ldb,
            const float* beta,
            float* c, const int* ldc) {
    enum CBLAS_TRANSPOSE ta = (*transa == 'T' || *transa == 't') ? CblasTrans :
                              (*transa == 'C' || *transa == 'c') ? CblasConjTrans :
                              CblasNoTrans;
    enum CBLAS_TRANSPOSE tb = (*transb == 'T' || *transb == 't') ? CblasTrans :
                              (*transb == 'C' || *transb == 'c') ? CblasConjTrans :
                              CblasNoTrans;

    cblas_sgemm(CblasColMajor, ta, tb,
                *m, *n, *k,
                *alpha, a, *lda,
                b, *ldb,
                *beta, c, *ldc);
}

/// Uppercase variant (some linkers look for SGEMM)
void SGEMM(const char* transa, const char* transb,
           const int* m, const int* n, const int* k,
           const float* alpha,
           const float* a, const int* lda,
           const float* b, const int* ldb,
           const float* beta,
           float* c, const int* ldc) {
    sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // extern "C"
