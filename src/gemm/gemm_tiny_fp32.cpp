/// @file gemm_tiny_fp32.cpp
/// Specialized kernels for tiny/irregular GEMM shapes.
///
/// This module provides optimized implementations for:
///   - N=1 (matrix-vector multiply, each row is a dot product)
///   - M=1 (GEMV: row vector × matrix)
///   - Tiny blocks (M,N ≤ 4)
///
/// These shapes are critical for:
///   - Batch inference (M=1,2,4)
///   - Attention mechanisms (M small, N large)
///   - Fully connected layers with small batches

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// N=1: Matrix × Vector (dot products per row)
// C[i,0] = alpha * sum_k(A[i,k] * B[k,0]) + beta * C[i,0]
// B is K×1 column vector (ldb=1 if contiguous)
// ============================================================

/// M×1 GEMM: each row computes one dot product with B column vector.
/// Optimized to process 4 rows simultaneously with 4-element K unrolling.
void gemm_mx1_fp32(int M, int K,
                    float alpha, const float* A, int lda,
                    const float* B, int ldb,
                    float beta, float* C, int ldc) {
    // Process 4 rows at a time for better ILP and SIMD utilization
    int i = 0;

    // If B is contiguous (ldb==1), we can use vector loads
    bool b_contiguous = (ldb == 1);

    for (; i + 3 < M; i += 4) {
        const float* a0 = A + i * lda;
        const float* a1 = A + (i + 1) * lda;
        const float* a2 = A + (i + 2) * lda;
        const float* a3 = A + (i + 3) * lda;

        float32x4_t sum0 = vdupq_n_f32(0);
        float32x4_t sum1 = vdupq_n_f32(0);
        float32x4_t sum2 = vdupq_n_f32(0);
        float32x4_t sum3 = vdupq_n_f32(0);

        int k = 0;

        if (b_contiguous) {
            // B is contiguous, can use vector loads
            for (; k + 3 < K; k += 4) {
                float32x4_t b = vld1q_f32(B + k);

                float32x4_t av0 = vld1q_f32(a0 + k);
                float32x4_t av1 = vld1q_f32(a1 + k);
                float32x4_t av2 = vld1q_f32(a2 + k);
                float32x4_t av3 = vld1q_f32(a3 + k);

                sum0 = vfmaq_f32(sum0, av0, b);
                sum1 = vfmaq_f32(sum1, av1, b);
                sum2 = vfmaq_f32(sum2, av2, b);
                sum3 = vfmaq_f32(sum3, av3, b);
            }
        } else {
            // B has stride, load elements individually but still vectorize A
            for (; k + 3 < K; k += 4) {
                float b0 = B[k * ldb];
                float b1 = B[(k + 1) * ldb];
                float b2 = B[(k + 2) * ldb];
                float b3 = B[(k + 3) * ldb];
                float32x4_t b = {b0, b1, b2, b3};

                float32x4_t av0 = vld1q_f32(a0 + k);
                float32x4_t av1 = vld1q_f32(a1 + k);
                float32x4_t av2 = vld1q_f32(a2 + k);
                float32x4_t av3 = vld1q_f32(a3 + k);

                sum0 = vfmaq_f32(sum0, av0, b);
                sum1 = vfmaq_f32(sum1, av1, b);
                sum2 = vfmaq_f32(sum2, av2, b);
                sum3 = vfmaq_f32(sum3, av3, b);
            }
        }

        // Horizontal sum for each row
        float r0 = vaddvq_f32(sum0);
        float r1 = vaddvq_f32(sum1);
        float r2 = vaddvq_f32(sum2);
        float r3 = vaddvq_f32(sum3);

        // Scalar tail for remaining K
        for (; k < K; ++k) {
            float bk = B[k * ldb];
            r0 += a0[k] * bk;
            r1 += a1[k] * bk;
            r2 += a2[k] * bk;
            r3 += a3[k] * bk;
        }

        // Apply alpha and beta
        if (beta == 0.0f) {
            C[i * ldc]       = alpha * r0;
            C[(i + 1) * ldc] = alpha * r1;
            C[(i + 2) * ldc] = alpha * r2;
            C[(i + 3) * ldc] = alpha * r3;
        } else {
            C[i * ldc]       = alpha * r0 + beta * C[i * ldc];
            C[(i + 1) * ldc] = alpha * r1 + beta * C[(i + 1) * ldc];
            C[(i + 2) * ldc] = alpha * r2 + beta * C[(i + 2) * ldc];
            C[(i + 3) * ldc] = alpha * r3 + beta * C[(i + 3) * ldc];
        }
    }

    // Process remaining rows (1-3 rows)
    for (; i < M; ++i) {
        const float* ai = A + i * lda;
        float32x4_t vsum = vdupq_n_f32(0);

        int k = 0;

        if (b_contiguous) {
            for (; k + 3 < K; k += 4) {
                float32x4_t a = vld1q_f32(ai + k);
                float32x4_t b = vld1q_f32(B + k);
                vsum = vfmaq_f32(vsum, a, b);
            }
        } else {
            for (; k + 3 < K; k += 4) {
                float32x4_t a = vld1q_f32(ai + k);
                float b0 = B[k * ldb];
                float b1 = B[(k+1) * ldb];
                float b2 = B[(k+2) * ldb];
                float b3 = B[(k+3) * ldb];
                float32x4_t b = {b0, b1, b2, b3};
                vsum = vfmaq_f32(vsum, a, b);
            }
        }

        float sum = vaddvq_f32(vsum);

        for (; k < K; ++k) {
            sum += ai[k] * B[k * ldb];
        }

        C[i * ldc] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[i * ldc];
    }
}

// ============================================================
// M=1: Row Vector × Matrix (GEMV)
// C[0,j] = alpha * sum_k(A[0,k] * B[k,j]) + beta * C[0,j]
// A is 1×K row vector, B is K×N matrix
// ============================================================

/// M=1 GEMV: row vector × matrix.
/// Uses wide N-panel processing with K-unrolling for better cache utilization.
void gemm_1xn_fp32(int N, int K,
                    float alpha, const float* A, int /*lda*/,
                    const float* B, int ldb,
                    float beta, float* C, int /*ldc*/) {
    // Process N in panels of 64 columns for L1 cache efficiency
    // 64 floats = 256 bytes = 4 cache lines
    constexpr int kPanelN = 64;

    for (int j0 = 0; j0 < N; j0 += kPanelN) {
        int j_len = std::min(kPanelN, N - j0);
        float* C_panel = C + j0;

        // Initialize accumulators for this N-panel (up to 64 accumulators)
        // Use 16 SIMD registers (16 × 4 = 64 floats)
        float32x4_t acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0);

        // K-loop: each 4-wide segment of N uses its own accumulator
        // acc[i] corresponds to columns [4i, 4i+3] within this panel
        for (int k = 0; k < K; ++k) {
            float ak = A[k];
            float32x4_t av = vdupq_n_f32(ak);
            const float* bk = B + k * ldb + j0;  // B[k, j0:j0+j_len]

            for (int j = 0; j + 3 < j_len; j += 4) {
                int idx = j / 4;  // accumulator index: 0..15
                acc[idx] = vfmaq_f32(acc[idx], av, vld1q_f32(bk + j));
            }
        }

        // Store results: each accumulator maps to its 4-wide segment
        float32x4_t av = vdupq_n_f32(alpha);
        float32x4_t bv = vdupq_n_f32(beta);
        for (int j = 0; j + 3 < j_len; j += 4) {
            int idx = j / 4;
            if (beta == 0.0f) {
                vst1q_f32(C_panel + j, vmulq_f32(av, acc[idx]));
            } else {
                vst1q_f32(C_panel + j, vfmaq_f32(vmulq_f32(bv, vld1q_f32(C_panel + j)), av, acc[idx]));
            }
        }

        // Scalar tail for N remainder (< 4)
        for (int j = (j_len / 4) * 4; j < j_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k] * B[k * ldb + j0 + j];
            }
            C_panel[j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C_panel[j];
        }
    }
}

// ============================================================
// M×N small: both M and N are small (≤ 8)
// Process multiple rows together for better SIMD utilization
// ============================================================

/// M×N small where both M,N ≤ 8.
/// Uses M-row parallel processing with N-column accumulation.
void gemm_small_mn_fp32(int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc) {
    // For M,N ≤ 8, we can process all M rows in parallel
    // Each row gets its own set of accumulators

    // Process in N-chunks of 4
    for (int j0 = 0; j0 < N; j0 += 4) {
        int j_len = std::min(4, N - j0);

        // Accumulators: M rows × j_len cols (up to 8×4 = 32 floats)
        float32x4_t acc[8];  // acc[i] = row i, cols j0:j0+3
        for (int i = 0; i < M; ++i) acc[i] = vdupq_n_f32(0);

        // K-loop
        for (int k = 0; k < K; ++k) {
            // Load B[k, j0:j0+j_len]
            float32x4_t bk;
            if (j_len == 4) {
                bk = vld1q_f32(B + k * ldb + j0);
            } else {
                // Partial load for edge
                float tmp[4] = {0, 0, 0, 0};
                for (int jj = 0; jj < j_len; ++jj) {
                    tmp[jj] = B[k * ldb + j0 + jj];
                }
                bk = vld1q_f32(tmp);
            }

            // FMA for each row
            for (int i = 0; i < M; ++i) {
                float aik = A[i * lda + k];
                acc[i] = vfmaq_n_f32(acc[i], bk, aik);
            }
        }

        // Store results
        float32x4_t av = vdupq_n_f32(alpha);
        for (int i = 0; i < M; ++i) {
            if (beta == 0.0f) {
                if (j_len == 4) {
                    vst1q_f32(C + i * ldc + j0, vmulq_f32(av, acc[i]));
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, vmulq_f32(av, acc[i]));
                    for (int jj = 0; jj < j_len; ++jj) {
                        C[i * ldc + j0 + jj] = tmp[jj];
                    }
                }
            } else {
                float32x4_t bv = vdupq_n_f32(beta);
                if (j_len == 4) {
                    vst1q_f32(C + i * ldc + j0,
                              vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + i * ldc + j0)), av, acc[i]));
                } else {
                    float tmp_c[4], tmp_acc[4];
                    for (int jj = 0; jj < j_len; ++jj) {
                        tmp_c[jj] = C[i * ldc + j0 + jj];
                    }
                    vst1q_f32(tmp_c, vld1q_f32(tmp_c));
                    vst1q_f32(tmp_acc, acc[i]);
                    float32x4_t cv = vld1q_f32(tmp_c);
                    float32x4_t result = vfmaq_f32(vmulq_f32(bv, cv), av, acc[i]);
                    vst1q_f32(tmp_acc, result);
                    for (int jj = 0; jj < j_len; ++jj) {
                        C[i * ldc + j0 + jj] = tmp_acc[jj];
                    }
                }
            }
        }
    }
}

// ============================================================
// Tiny block kernels: M,N ≤ 4 with full unrolling
// ============================================================

/// 4×4 microkernel: fully unrolled for maximum ILP.
void gemm_4x4_fp32(int K,
                    float alpha, const float* A, int lda,
                    const float* B, int ldb,
                    float beta, float* C, int ldc) {
    // 4×4 = 16 accumulators
    float32x4_t c0 = vdupq_n_f32(0);  // row 0
    float32x4_t c1 = vdupq_n_f32(0);  // row 1
    float32x4_t c2 = vdupq_n_f32(0);  // row 2
    float32x4_t c3 = vdupq_n_f32(0);  // row 3

    for (int k = 0; k < K; ++k) {
        // Load A column k (4 elements from 4 rows)
        float a0k = A[k];
        float a1k = A[lda + k];
        float a2k = A[2 * lda + k];
        float a3k = A[3 * lda + k];

        // Load B row k (4 elements)
        float32x4_t bk = vld1q_f32(&B[k * ldb]);

        // FMA: ci += aik * bk
        c0 = vfmaq_n_f32(c0, bk, a0k);
        c1 = vfmaq_n_f32(c1, bk, a1k);
        c2 = vfmaq_n_f32(c2, bk, a2k);
        c3 = vfmaq_n_f32(c3, bk, a3k);
    }

    // Apply alpha and beta
    float32x4_t av = vdupq_n_f32(alpha);
    if (beta == 0.0f) {
        vst1q_f32(C,             vmulq_f32(av, c0));
        vst1q_f32(C + ldc,       vmulq_f32(av, c1));
        vst1q_f32(C + 2 * ldc,   vmulq_f32(av, c2));
        vst1q_f32(C + 3 * ldc,   vmulq_f32(av, c3));
    } else {
        float32x4_t bv = vdupq_n_f32(beta);
        vst1q_f32(C,           vfmaq_f32(vmulq_f32(bv, vld1q_f32(C)),           av, c0));
        vst1q_f32(C + ldc,     vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + ldc)),     av, c1));
        vst1q_f32(C + 2*ldc,   vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + 2*ldc)),   av, c2));
        vst1q_f32(C + 3*ldc,   vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + 3*ldc)),   av, c3));
    }
}

/// General tiny block: M,N ≤ 4, use small_mn driver or scalar fallback.
void gemm_tiny_block_fp32(int M, int N, int K,
                           float alpha, const float* A, int lda,
                           const float* B, int ldb,
                           float beta, float* C, int ldc) {
    // Fast paths for common sizes
    if (M == 4 && N == 4) {
        gemm_4x4_fp32(K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    if (M == 1 && N == 1) {
        // Pure dot product
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k] * B[k * ldb];
        }
        C[0] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[0];
        return;
    }

    // Use the small M×N driver for other cases
    gemm_small_mn_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ============================================================
// Public dispatch entry point
// ============================================================

/// Dispatch to specialized tiny kernels based on shape.
/// Returns true if handled, false if caller should use general GEMM.
bool gemm_tiny_dispatch_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
    // N=1: Matrix × Vector (dot product per row)
    if (N == 1) {
        gemm_mx1_fp32(M, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    // M=1: Row Vector × Matrix (GEMV)
    if (M == 1) {
        gemm_1xn_fp32(N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    // Tiny blocks: M,N ≤ 4
    if (M <= 4 && N <= 4) {
        gemm_tiny_block_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    // Small blocks: M,N ≤ 8
    if (M <= 8 && N <= 8) {
        gemm_small_mn_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    // Not a tiny shape
    return false;
}

#endif  // __ARM_NEON

}  // namespace dnnopt