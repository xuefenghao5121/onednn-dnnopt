/// @file gemm_smallm_fp32.cpp
/// Small-M specialized FP32 GEMM driver and microkernel.
///
/// For M < Mr (8), the standard 8×12 BLIS path wastes compute on zero-padded
/// rows and incurs unnecessary A-packing overhead. This module provides:
///   - A 1×48 NEON microkernel (12 accumulators, 4x K-unroll)
///   - M=1 driver: no packing at all (GEMV-like, direct B access)
///   - M=2-7 driver: pack B only, iterate rows with 1×48 kernel

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// Nr for small-M path: 48 columns (12 NEON registers)
static constexpr int kSmallMNr = 48;

/// 1×48 microkernel: computes one row of C from unpacked A row and B block.
///
/// C[0, 0:48] = alpha * sum_k(A[k] * B[k, 0:48]) + beta * C[0, 0:48]
///
/// @param K      number of K iterations
/// @param A      pointer to A row (contiguous, stride irrelevant for 1 row)
/// @param B      pointer to B block, layout depends on packed flag
/// @param ldb    stride of B (used when B is not packed; ignored when packed)
/// @param C      output row pointer
/// @param alpha  scaling factor
/// @param beta   scaling factor for existing C
/// @param packed if true, B is packed as kSmallMNr-wide panels (contiguous per K)
static void gemm_ukernel_fp32_1x48(int K,
                                    const float* A,
                                    const float* B, int ldb,
                                    float* C,
                                    float alpha, float beta,
                                    bool packed) {
    // 12 accumulator registers for 48 columns
    float32x4_t c0  = vdupq_n_f32(0), c1  = vdupq_n_f32(0);
    float32x4_t c2  = vdupq_n_f32(0), c3  = vdupq_n_f32(0);
    float32x4_t c4  = vdupq_n_f32(0), c5  = vdupq_n_f32(0);
    float32x4_t c6  = vdupq_n_f32(0), c7  = vdupq_n_f32(0);
    float32x4_t c8  = vdupq_n_f32(0), c9  = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);

    int k = 0;

    if (packed) {
        // Packed B: 48 contiguous floats per K iteration
        for (; k + 3 < K; k += 4) {
            // Iteration 0
            {
                float32x4_t a = vdupq_n_f32(A[k]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 1
            {
                float32x4_t a = vdupq_n_f32(A[k + 1]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 2
            {
                float32x4_t a = vdupq_n_f32(A[k + 2]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 3
            {
                float32x4_t a = vdupq_n_f32(A[k + 3]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
        }
        for (; k < K; ++k) {
            float32x4_t a = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
            c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
            c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
            c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
            c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
            c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
            c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
            c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
            c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
            c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
            c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
            c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
            B += kSmallMNr;
        }
    } else {
        // Unpacked B: stride = ldb between rows
        for (; k + 3 < K; k += 4) {
            const float* b0 = B + (k)     * ldb;
            const float* b1 = B + (k + 1) * ldb;
            const float* b2 = B + (k + 2) * ldb;
            const float* b3 = B + (k + 3) * ldb;

            float32x4_t a0 = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a0, vld1q_f32(b0));
            c1  = vfmaq_f32(c1,  a0, vld1q_f32(b0 + 4));
            c2  = vfmaq_f32(c2,  a0, vld1q_f32(b0 + 8));
            c3  = vfmaq_f32(c3,  a0, vld1q_f32(b0 + 12));
            c4  = vfmaq_f32(c4,  a0, vld1q_f32(b0 + 16));
            c5  = vfmaq_f32(c5,  a0, vld1q_f32(b0 + 20));
            c6  = vfmaq_f32(c6,  a0, vld1q_f32(b0 + 24));
            c7  = vfmaq_f32(c7,  a0, vld1q_f32(b0 + 28));
            c8  = vfmaq_f32(c8,  a0, vld1q_f32(b0 + 32));
            c9  = vfmaq_f32(c9,  a0, vld1q_f32(b0 + 36));
            c10 = vfmaq_f32(c10, a0, vld1q_f32(b0 + 40));
            c11 = vfmaq_f32(c11, a0, vld1q_f32(b0 + 44));

            float32x4_t a1 = vdupq_n_f32(A[k + 1]);
            c0  = vfmaq_f32(c0,  a1, vld1q_f32(b1));
            c1  = vfmaq_f32(c1,  a1, vld1q_f32(b1 + 4));
            c2  = vfmaq_f32(c2,  a1, vld1q_f32(b1 + 8));
            c3  = vfmaq_f32(c3,  a1, vld1q_f32(b1 + 12));
            c4  = vfmaq_f32(c4,  a1, vld1q_f32(b1 + 16));
            c5  = vfmaq_f32(c5,  a1, vld1q_f32(b1 + 20));
            c6  = vfmaq_f32(c6,  a1, vld1q_f32(b1 + 24));
            c7  = vfmaq_f32(c7,  a1, vld1q_f32(b1 + 28));
            c8  = vfmaq_f32(c8,  a1, vld1q_f32(b1 + 32));
            c9  = vfmaq_f32(c9,  a1, vld1q_f32(b1 + 36));
            c10 = vfmaq_f32(c10, a1, vld1q_f32(b1 + 40));
            c11 = vfmaq_f32(c11, a1, vld1q_f32(b1 + 44));

            float32x4_t a2 = vdupq_n_f32(A[k + 2]);
            c0  = vfmaq_f32(c0,  a2, vld1q_f32(b2));
            c1  = vfmaq_f32(c1,  a2, vld1q_f32(b2 + 4));
            c2  = vfmaq_f32(c2,  a2, vld1q_f32(b2 + 8));
            c3  = vfmaq_f32(c3,  a2, vld1q_f32(b2 + 12));
            c4  = vfmaq_f32(c4,  a2, vld1q_f32(b2 + 16));
            c5  = vfmaq_f32(c5,  a2, vld1q_f32(b2 + 20));
            c6  = vfmaq_f32(c6,  a2, vld1q_f32(b2 + 24));
            c7  = vfmaq_f32(c7,  a2, vld1q_f32(b2 + 28));
            c8  = vfmaq_f32(c8,  a2, vld1q_f32(b2 + 32));
            c9  = vfmaq_f32(c9,  a2, vld1q_f32(b2 + 36));
            c10 = vfmaq_f32(c10, a2, vld1q_f32(b2 + 40));
            c11 = vfmaq_f32(c11, a2, vld1q_f32(b2 + 44));

            float32x4_t a3 = vdupq_n_f32(A[k + 3]);
            c0  = vfmaq_f32(c0,  a3, vld1q_f32(b3));
            c1  = vfmaq_f32(c1,  a3, vld1q_f32(b3 + 4));
            c2  = vfmaq_f32(c2,  a3, vld1q_f32(b3 + 8));
            c3  = vfmaq_f32(c3,  a3, vld1q_f32(b3 + 12));
            c4  = vfmaq_f32(c4,  a3, vld1q_f32(b3 + 16));
            c5  = vfmaq_f32(c5,  a3, vld1q_f32(b3 + 20));
            c6  = vfmaq_f32(c6,  a3, vld1q_f32(b3 + 24));
            c7  = vfmaq_f32(c7,  a3, vld1q_f32(b3 + 28));
            c8  = vfmaq_f32(c8,  a3, vld1q_f32(b3 + 32));
            c9  = vfmaq_f32(c9,  a3, vld1q_f32(b3 + 36));
            c10 = vfmaq_f32(c10, a3, vld1q_f32(b3 + 40));
            c11 = vfmaq_f32(c11, a3, vld1q_f32(b3 + 44));
        }
        for (; k < K; ++k) {
            const float* bk = B + k * ldb;
            float32x4_t a = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a, vld1q_f32(bk));
            c1  = vfmaq_f32(c1,  a, vld1q_f32(bk + 4));
            c2  = vfmaq_f32(c2,  a, vld1q_f32(bk + 8));
            c3  = vfmaq_f32(c3,  a, vld1q_f32(bk + 12));
            c4  = vfmaq_f32(c4,  a, vld1q_f32(bk + 16));
            c5  = vfmaq_f32(c5,  a, vld1q_f32(bk + 20));
            c6  = vfmaq_f32(c6,  a, vld1q_f32(bk + 24));
            c7  = vfmaq_f32(c7,  a, vld1q_f32(bk + 28));
            c8  = vfmaq_f32(c8,  a, vld1q_f32(bk + 32));
            c9  = vfmaq_f32(c9,  a, vld1q_f32(bk + 36));
            c10 = vfmaq_f32(c10, a, vld1q_f32(bk + 40));
            c11 = vfmaq_f32(c11, a, vld1q_f32(bk + 44));
        }
    }

    // Epilogue: C = alpha * acc + beta * C
    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

#define STORE_48(off, acc) do {                                        \
    if (beta == 0.0f)                                                  \
        vst1q_f32(C + (off), vmulq_f32(av, acc));                     \
    else                                                               \
        vst1q_f32(C + (off), vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + (off))), av, acc)); \
} while(0)

    STORE_48(0,  c0);  STORE_48(4,  c1);  STORE_48(8,  c2);
    STORE_48(12, c3);  STORE_48(16, c4);  STORE_48(20, c5);
    STORE_48(24, c6);  STORE_48(28, c7);  STORE_48(32, c8);
    STORE_48(36, c9);  STORE_48(40, c10); STORE_48(44, c11);

#undef STORE_48
}

/// Scalar tail for N-remainder < 4 (when N is not divisible by 4).
static void gemm_scalar_1xn(int K, int n_rem,
                             const float* A,
                             const float* B, int ldb,
                             float* C,
                             float alpha, float beta) {
    for (int j = 0; j < n_rem; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[k] * B[k * ldb + j];
        if (beta == 0.0f)
            C[j] = alpha * acc;
        else
            C[j] = alpha * acc + beta * C[j];
    }
}

/// NEON 1×(4n) helper for N-remainder between 4 and 47.
static void gemm_neon_1xn(int K, int n_cols,
                           const float* A,
                           const float* B, int ldb,
                           float* C,
                           float alpha, float beta) {
    // Process 4 columns at a time
    int j = 0;
    for (; j + 3 < n_cols; j += 4) {
        float32x4_t acc = vdupq_n_f32(0);
        for (int k = 0; k < K; ++k) {
            float32x4_t a = vdupq_n_f32(A[k]);
            acc = vfmaq_f32(acc, a, vld1q_f32(&B[k * ldb + j]));
        }
        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f)
            vst1q_f32(&C[j], vmulq_f32(av, acc));
        else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C[j])), av, acc));
        }
    }
    // Scalar tail
    if (j < n_cols)
        gemm_scalar_1xn(K, n_cols - j, A, &B[j], ldb, &C[j], alpha, beta);
}

// ============================================================
// Multi-row small-M kernels (M=2,4 parallel processing)
// ============================================================

/// 2×N microkernel: process 2 rows together for better SIMD utilization.
/// Each row shares the same B access pattern, reducing memory traffic.
/// K must be small enough to fit accumulators in registers.
static void gemm_ukernel_fp32_2xN_kblock(int k_len,
                                          const float* A0, const float* A1,
                                          const float* B, int ldb,
                                          float32x4_t c0, float32x4_t c1,
                                          int j_start, int j_len) {
    for (int k = 0; k < k_len; ++k) {
        float32x4_t bk = vld1q_f32(&B[k * ldb + j_start]);
        c0 = vfmaq_n_f32(c0, bk, A0[k]);
        c1 = vfmaq_n_f32(c1, bk, A1[k]);
    }
}

/// 4×N microkernel: process 4 rows together with K blocking.
static void gemm_ukernel_fp32_4xN_kblock(int k_len,
                                          const float* A, int lda,
                                          const float* B, int ldb,
                                          float32x4_t c0, float32x4_t c1,
                                          float32x4_t c2, float32x4_t c3,
                                          int j_start) {
    const float* A0 = A;
    const float* A1 = A + lda;
    const float* A2 = A + 2 * lda;
    const float* A3 = A + 3 * lda;

    for (int k = 0; k < k_len; ++k) {
        float32x4_t bk = vld1q_f32(&B[k * ldb + j_start]);
        c0 = vfmaq_n_f32(c0, bk, A0[k]);
        c1 = vfmaq_n_f32(c1, bk, A1[k]);
        c2 = vfmaq_n_f32(c2, bk, A2[k]);
        c3 = vfmaq_n_f32(c3, bk, A3[k]);
    }
}

/// 2×N with K-blocking for large K matrices.
static void gemm_ukernel_fp32_2xN(int N, int K,
                                   const float* A0, const float* A1, int lda,
                                   const float* B, int ldb,
                                   float* C0, float* C1, int ldc,
                                   float alpha, float beta) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;  // K-block size

    // Process N in chunks of 4
    int j = 0;
    for (; j + 3 < N; j += 4) {
        float32x4_t c0 = vdupq_n_f32(0);
        float32x4_t c1 = vdupq_n_f32(0);

        // K blocking for cache efficiency
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            for (int k = 0; k < kc; ++k) {
                float32x4_t bk = vld1q_f32(&B[(pc + k) * ldb + j]);
                c0 = vfmaq_n_f32(c0, bk, A0[pc + k]);
                c1 = vfmaq_n_f32(c1, bk, A1[pc + k]);
            }
        }

        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f) {
            vst1q_f32(&C0[j], vmulq_f32(av, c0));
            vst1q_f32(&C1[j], vmulq_f32(av, c1));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C0[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C0[j])), av, c0));
            vst1q_f32(&C1[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C1[j])), av, c1));
        }
    }

    // Scalar tail
    for (; j < N; ++j) {
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int k = 0; k < K; ++k) {
            float bkj = B[k * ldb + j];
            sum0 += A0[k] * bkj;
            sum1 += A1[k] * bkj;
        }
        C0[j] = (beta == 0.0f) ? alpha * sum0 : alpha * sum0 + beta * C0[j];
        C1[j] = (beta == 0.0f) ? alpha * sum1 : alpha * sum1 + beta * C1[j];
    }
}

/// 4×N with K-blocking for large K matrices.
static void gemm_ukernel_fp32_4xN(int N, int K,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float* C, int ldc,
                                   float alpha, float beta) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;  // K-block size

    const float* A0 = A;
    const float* A1 = A + lda;
    const float* A2 = A + 2 * lda;
    const float* A3 = A + 3 * lda;
    float* C0 = C;
    float* C1 = C + ldc;
    float* C2 = C + 2 * ldc;
    float* C3 = C + 3 * ldc;

    // Process N in chunks of 4
    int j = 0;
    for (; j + 3 < N; j += 4) {
        float32x4_t c0 = vdupq_n_f32(0);
        float32x4_t c1 = vdupq_n_f32(0);
        float32x4_t c2 = vdupq_n_f32(0);
        float32x4_t c3 = vdupq_n_f32(0);

        // K blocking for cache efficiency
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            for (int k = 0; k < kc; ++k) {
                float32x4_t bk = vld1q_f32(&B[(pc + k) * ldb + j]);
                c0 = vfmaq_n_f32(c0, bk, A0[pc + k]);
                c1 = vfmaq_n_f32(c1, bk, A1[pc + k]);
                c2 = vfmaq_n_f32(c2, bk, A2[pc + k]);
                c3 = vfmaq_n_f32(c3, bk, A3[pc + k]);
            }
        }

        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f) {
            vst1q_f32(&C0[j], vmulq_f32(av, c0));
            vst1q_f32(&C1[j], vmulq_f32(av, c1));
            vst1q_f32(&C2[j], vmulq_f32(av, c2));
            vst1q_f32(&C3[j], vmulq_f32(av, c3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C0[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C0[j])), av, c0));
            vst1q_f32(&C1[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C1[j])), av, c1));
            vst1q_f32(&C2[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C2[j])), av, c2));
            vst1q_f32(&C3[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C3[j])), av, c3));
        }
    }

    // Scalar tail
    for (; j < N; ++j) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (int k = 0; k < K; ++k) {
            float bkj = B[k * ldb + j];
            sum0 += A0[k] * bkj;
            sum1 += A1[k] * bkj;
            sum2 += A2[k] * bkj;
            sum3 += A3[k] * bkj;
        }
        C0[j] = (beta == 0.0f) ? alpha * sum0 : alpha * sum0 + beta * C0[j];
        C1[j] = (beta == 0.0f) ? alpha * sum1 : alpha * sum1 + beta * C1[j];
        C2[j] = (beta == 0.0f) ? alpha * sum2 : alpha * sum2 + beta * C2[j];
        C3[j] = (beta == 0.0f) ? alpha * sum3 : alpha * sum3 + beta * C3[j];
    }
}

// ============================================================
// Small-M drivers
// ============================================================

/// M=1 driver: optimized GEMV with panel-based processing.
/// Processes N in panels of 64 for L1 cache efficiency.
static void gemm_smallm1_fp32(int N, int K,
                               float alpha, const float* A, int lda,
                               const float* B, int ldb,
                               float beta, float* C, int ldc) {
    (void)lda; (void)ldc;  // M=1, stride not needed
    constexpr int kPanelN = 64;

    for (int j0 = 0; j0 < N; j0 += kPanelN) {
        int j_len = std::min(kPanelN, N - j0);

        // Initialize accumulators (16 SIMD vectors = 64 floats)
        float32x4_t acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0);

        // K-loop: each 4-wide segment of N uses its own accumulator
        // acc[i] corresponds to columns [4i, 4i+3] within this panel
        for (int k = 0; k < K; ++k) {
            float ak = A[k];
            float32x4_t av = vdupq_n_f32(ak);
            const float* bk = B + k * ldb + j0;

            for (int j = 0; j + 3 < j_len; j += 4) {
                int idx = j / 4;  // accumulator index: 0..15
                acc[idx] = vfmaq_f32(acc[idx], av, vld1q_f32(bk + j));
            }
        }

        // Store: each accumulator maps to its 4-wide segment
        float32x4_t av = vdupq_n_f32(alpha);
        float32x4_t bv = vdupq_n_f32(beta);
        for (int j = 0; j + 3 < j_len; j += 4) {
            int idx = j / 4;
            if (beta == 0.0f) {
                vst1q_f32(C + j0 + j, vmulq_f32(av, acc[idx]));
            } else {
                vst1q_f32(C + j0 + j, vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + j0 + j)), av, acc[idx]));
            }
        }
        // Scalar tail for N remainder (< 4)
        for (int j = (j_len / 4) * 4; j < j_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k] * B[k * ldb + j0 + j];
            }
            C[j0 + j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[j0 + j];
        }
    }
}

/// M=2-7 driver: process multiple rows together for better B cache reuse.
/// Uses 4-row and 2-row microkernels when possible.
static void gemm_smallm_multi_fp32(int M, int N, int K,
                                    float alpha, const float* A, int lda,
                                    const float* B, int ldb,
                                    float beta, float* C, int ldc) {
    int i = 0;

    // Process M=4 blocks
    for (; i + 3 < M; i += 4) {
        gemm_ukernel_fp32_4xN(N, K, A + i * lda, lda, B, ldb,
                               C + i * ldc, ldc, alpha, beta);
    }

    // Process M=2 blocks
    for (; i + 1 < M; i += 2) {
        gemm_ukernel_fp32_2xN(N, K,
                               A + i * lda, A + (i + 1) * lda, lda,
                               B, ldb,
                               C + i * ldc, C + (i + 1) * ldc, ldc,
                               alpha, beta);
    }

    // Process remaining single row
    if (i < M) {
        gemm_smallm1_fp32(N, K, alpha, A + i * lda, lda, B, ldb, beta, C + i * ldc, ldc);
    }
}

/// Public small-M driver entry point.
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
    if (M == 1) {
        gemm_smallm1_fp32(N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        gemm_smallm_multi_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
