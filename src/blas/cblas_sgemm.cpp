/// @file cblas_sgemm.cpp
/// CBLAS sgemm implementation backed by dnnopt::gemm_fp32.
///
/// Handles all Order × TransA × TransB combinations by reducing them
/// to row-major, no-transpose calls that dnnopt::gemm_fp32 expects.
///
/// Key insight: ColMajor(A, B, C) = RowMajor(B^T, A^T, C^T)
/// so ColMajor can always be converted to RowMajor by swapping A↔B and M↔N.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <cstring>

namespace {

/// Fast row-major transpose: dst[N,M] = src[M,N]^T
/// NEON-accelerated 4x4 block transpose for interior, scalar for edges.
void transpose_f32(const float* src, float* dst,
                   int rows, int cols, int ld_src, int ld_dst) {
    int r = 0;
#ifdef __ARM_NEON
    for (; r + 3 < rows; r += 4) {
        int c = 0;
        for (; c + 3 < cols; c += 4) {
            // Load 4 rows of 4 elements each
            float32x4_t r0 = vld1q_f32(src + (r + 0) * ld_src + c);
            float32x4_t r1 = vld1q_f32(src + (r + 1) * ld_src + c);
            float32x4_t r2 = vld1q_f32(src + (r + 2) * ld_src + c);
            float32x4_t r3 = vld1q_f32(src + (r + 3) * ld_src + c);

            // 4x4 transpose via zip
            float32x4x2_t t01 = vzipq_f32(r0, r2);  // {r0[0],r2[0],r0[1],r2[1]}, {r0[2],r2[2],r0[3],r2[3]}
            float32x4x2_t t23 = vzipq_f32(r1, r3);

            float32x4x2_t u0 = vzipq_f32(t01.val[0], t23.val[0]);
            float32x4x2_t u1 = vzipq_f32(t01.val[1], t23.val[1]);

            vst1q_f32(dst + (c + 0) * ld_dst + r, u0.val[0]);
            vst1q_f32(dst + (c + 1) * ld_dst + r, u0.val[1]);
            vst1q_f32(dst + (c + 2) * ld_dst + r, u1.val[0]);
            vst1q_f32(dst + (c + 3) * ld_dst + r, u1.val[1]);
        }
        // Scalar tail for columns
        for (; c < cols; ++c) {
            for (int rr = r; rr < r + 4; ++rr) {
                dst[c * ld_dst + rr] = src[rr * ld_src + c];
            }
        }
    }
#endif
    // Scalar tail for rows
    for (; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[c * ld_dst + r] = src[r * ld_src + c];
        }
    }
}

/// Core: row-major GEMM with transpose support.
/// Handles RowMajor + any TransA/TransB combination.
void sgemm_row_major(enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                     int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc) {
    bool ta = (TransA != CblasNoTrans);
    bool tb = (TransB != CblasNoTrans);

    if (!ta && !tb) {
        // A[M,K] x B[K,N] -> C[M,N] — direct pass-through
        dnnopt::gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (ta && !tb) {
        // A is K×M stored row-major, need A^T which is M×K
        // Transpose A to temporary buffer
        auto A_T = dnnopt::aligned_array<float>((size_t)M * K);
        transpose_f32(A, A_T.get(), K, M, lda, K);
        dnnopt::gemm_fp32(M, N, K, alpha, A_T.get(), K, B, ldb, beta, C, ldc);
    } else if (!ta && tb) {
        // B is N×K stored row-major, need B^T which is K×N
        auto B_T = dnnopt::aligned_array<float>((size_t)K * N);
        transpose_f32(B, B_T.get(), N, K, ldb, N);
        dnnopt::gemm_fp32(M, N, K, alpha, A, lda, B_T.get(), N, beta, C, ldc);
    } else {
        // Both transposed
        auto A_T = dnnopt::aligned_array<float>((size_t)M * K);
        auto B_T = dnnopt::aligned_array<float>((size_t)K * N);
        transpose_f32(A, A_T.get(), K, M, lda, K);
        transpose_f32(B, B_T.get(), N, K, ldb, N);
        dnnopt::gemm_fp32(M, N, K, alpha, A_T.get(), K, B_T.get(), N, beta, C, ldc);
    }
}

}  // namespace

extern "C" {

void cblas_sgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (alpha == 0.0f && beta == 1.0f) return;

    if (Order == CblasRowMajor) {
        sgemm_row_major(TransA, TransB, M, N, K,
                        alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // ColMajor: C_col = op(A_col) × op(B_col)
        //
        // Duality: Column-major C[M,N] = A[M,K] × B[K,N]
        //       ≡ Row-major C^T[N,M] = B^T[N,K] × A^T[K,M]
        //
        // A column-major matrix with 'NoTrans' is already A^T in row-major memory,
        // so the swap + NoTrans gives us the correct result.
        // A column-major matrix with 'Trans' means op(A)=A^T, which when viewed
        // in row-major (where the memory is already A^T) means: op = (A^T)^T = A,
        // i.e. NoTrans in row-major.
        //
        // Mapping: ColMajor-NoTrans → keep as is (duality swaps), ColMajor-Trans → NoTrans
        //   ColMajor,N,N → RowMajor,N,N with swap(A,B), swap(M,N)
        //   ColMajor,T,N → RowMajor,N,T with swap(A,B), swap(M,N)
        //   ColMajor,N,T → RowMajor,T,N with swap(A,B), swap(M,N)
        //   ColMajor,T,T → RowMajor,T,T with swap(A,B), swap(M,N)
        //
        // After the A↔B swap, TransA applies to old-B and TransB applies to old-A.
        // The new RowMajor call has: new_TransA = TransB, new_TransB = TransA
        sgemm_row_major(TransB, TransA, N, M, K,
                        alpha, B, ldb, A, lda, beta, C, ldc);
    }
}

}  // extern "C"
