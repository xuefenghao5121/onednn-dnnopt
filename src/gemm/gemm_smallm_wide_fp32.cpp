/// @file gemm_smallm_wide_fp32.cpp
/// Wide-panel small-M FP32 GEMM kernels for M=2-7.
///
/// Replaces the narrow 4-column-at-a-time approach with wide 48-column panels.
/// B is streamed once per panel (not N/4 times), giving ~12x bandwidth improvement.
///
/// Register strategy:
///   M=2, PanelW=48: 24 acc regs (fits in 32 NEON regs, no spill)
///   M=4, PanelW=48: 16-col sub-blocks → 16 acc + 4 B = 20 regs, no spill
///   M=3,5-7, PanelW=32: 16-col sub-blocks → M*4 acc + 4 B regs, no spill for M≤7

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// Wide panel width for M=2,4 path
static constexpr int kWidePanelN = 48;
// Wide panel width for M=3,5-7 path (reduced register pressure)
static constexpr int kMedPanelN = 32;
// Sub-block width for multi-row kernels
static constexpr int kSubBlockN = 16;
// Kc blocking for large-K: B[Kc][panelN] fits L1D.
// L1D=64KB, panelN=48: Kc = 64KB / (48*4) = 341 → round to 256 for alignment.
static constexpr int kWideKc = 256;

// ============================================================
// Epilogue helpers
// ============================================================

/// Store 16 floats from 4 SIMD accumulators, applying alpha/beta.
static inline void store_16(float* ptr, float32x4_t c0, float32x4_t c1,
                            float32x4_t c2, float32x4_t c3,
                            float alpha, float beta) {
    float32x4_t av = vdupq_n_f32(alpha);
    if (beta == 0.0f) {
        vst1q_f32(ptr,      vmulq_f32(av, c0));
        vst1q_f32(ptr + 4,  vmulq_f32(av, c1));
        vst1q_f32(ptr + 8,  vmulq_f32(av, c2));
        vst1q_f32(ptr + 12, vmulq_f32(av, c3));
    } else {
        float32x4_t bv = vdupq_n_f32(beta);
        vst1q_f32(ptr,      vfmaq_f32(vmulq_f32(bv, vld1q_f32(ptr)),      av, c0));
        vst1q_f32(ptr + 4,  vfmaq_f32(vmulq_f32(bv, vld1q_f32(ptr + 4)),  av, c1));
        vst1q_f32(ptr + 8,  vfmaq_f32(vmulq_f32(bv, vld1q_f32(ptr + 8)),  av, c2));
        vst1q_f32(ptr + 12, vfmaq_f32(vmulq_f32(bv, vld1q_f32(ptr + 12)), av, c3));
    }
}

/// Store tail < 16 floats from SIMD accumulators.
static inline void store_tail(float* ptr, int j_len,
                              float32x4_t c0, float32x4_t c1,
                              float32x4_t c2, float32x4_t c3,
                              float alpha, float beta) {
    float tmp[16];
    float32x4_t av = vdupq_n_f32(alpha);
    vst1q_f32(tmp,      vmulq_f32(av, c0));
    vst1q_f32(tmp + 4,  vmulq_f32(av, c1));
    vst1q_f32(tmp + 8,  vmulq_f32(av, c2));
    vst1q_f32(tmp + 12, vmulq_f32(av, c3));
    for (int j = 0; j < j_len; ++j) {
        ptr[j] = (beta == 0.0f) ? tmp[j] : tmp[j] + beta * ptr[j];
    }
}

// ============================================================
// M=2, PanelW=48: full register, no spill
// 2 rows × 12 acc = 24 SIMD registers + B working set
// ============================================================

static void gemm_wide_2x48(int K,
                            const float* A0, const float* A1,
                            const float* B, int ldb,
                            float* C0, float* C1, int ldc,
                            int j0, int j_len,
                            float alpha, float beta) {
    // Phase 11: Kc-outer, sub-block-inner loop ordering.
    // All sub-block accumulators persist across Kc blocks.
    // B[Kc][48] = 256*48*4 = 48KB fits L1D (64KB).
    //
    // For full 48-col panel: 2 rows * 3 sub-blocks * 4 acc = 24 SIMD regs.
    // Plus B working set (4 regs) = 28 total. Fits in 32 NEON registers.

    int n_subs = (j_len + kSubBlockN - 1) / kSubBlockN;

    // Allocate accumulators for ALL sub-blocks: 2 rows * n_subs * 4 acc
    // For full 48 cols: n_subs=3, total = 2*3*4 = 24 registers
    // Use flat array: acc[row][sub][col4] = acc[row * n_subs * 4 + sub * 4 + col4]
    constexpr int kMaxSubs = 3;  // 48/16
    float32x4_t r0[kMaxSubs * 4], r1[kMaxSubs * 4];
    for (int i = 0; i < n_subs * 4; ++i) {
        r0[i] = vdupq_n_f32(0);
        r1[i] = vdupq_n_f32(0);
    }

    // Kc-outer loop
    for (int pc = 0; pc < K; pc += kWideKc) {
        int kc = std::min(kWideKc, K - pc);

        // K-inner loop: for each k, process ALL sub-blocks
        int k = 0;
        for (; k + 1 < kc; k += 2) {
            float a00 = A0[pc + k], a10 = A1[pc + k];
            float a01 = A0[pc + k + 1], a11 = A1[pc + k + 1];

            for (int s = 0; s < n_subs; ++s) {
                int js = s * kSubBlockN;
                int sub_len = std::min(kSubBlockN, j_len - js);
                int base = s * 4;
                const float* bk0 = B + (pc + k) * ldb + j0 + js;
                const float* bk1 = B + (pc + k + 1) * ldb + j0 + js;

                // K iteration 0
                float32x4_t b0 = vld1q_f32(bk0);
                r0[base] = vfmaq_n_f32(r0[base], b0, a00);
                r1[base] = vfmaq_n_f32(r1[base], b0, a10);
                if (sub_len > 4)  { float32x4_t b = vld1q_f32(bk0+4);  r0[base+1] = vfmaq_n_f32(r0[base+1], b, a00); r1[base+1] = vfmaq_n_f32(r1[base+1], b, a10); }
                if (sub_len > 8)  { float32x4_t b = vld1q_f32(bk0+8);  r0[base+2] = vfmaq_n_f32(r0[base+2], b, a00); r1[base+2] = vfmaq_n_f32(r1[base+2], b, a10); }
                if (sub_len > 12) { float32x4_t b = vld1q_f32(bk0+12); r0[base+3] = vfmaq_n_f32(r0[base+3], b, a00); r1[base+3] = vfmaq_n_f32(r1[base+3], b, a10); }

                // K iteration 1
                b0 = vld1q_f32(bk1);
                r0[base] = vfmaq_n_f32(r0[base], b0, a01);
                r1[base] = vfmaq_n_f32(r1[base], b0, a11);
                if (sub_len > 4)  { float32x4_t b = vld1q_f32(bk1+4);  r0[base+1] = vfmaq_n_f32(r0[base+1], b, a01); r1[base+1] = vfmaq_n_f32(r1[base+1], b, a11); }
                if (sub_len > 8)  { float32x4_t b = vld1q_f32(bk1+8);  r0[base+2] = vfmaq_n_f32(r0[base+2], b, a01); r1[base+2] = vfmaq_n_f32(r1[base+2], b, a11); }
                if (sub_len > 12) { float32x4_t b = vld1q_f32(bk1+12); r0[base+3] = vfmaq_n_f32(r0[base+3], b, a01); r1[base+3] = vfmaq_n_f32(r1[base+3], b, a11); }
            }
        }
        // K tail
        if (k < kc) {
            float a00 = A0[pc + k], a10 = A1[pc + k];
            for (int s = 0; s < n_subs; ++s) {
                int js = s * kSubBlockN;
                int sub_len = std::min(kSubBlockN, j_len - js);
                int base = s * 4;
                const float* bk = B + (pc + k) * ldb + j0 + js;

                float32x4_t b0 = vld1q_f32(bk);
                r0[base] = vfmaq_n_f32(r0[base], b0, a00);
                r1[base] = vfmaq_n_f32(r1[base], b0, a10);
                if (sub_len > 4)  { float32x4_t b = vld1q_f32(bk+4);  r0[base+1] = vfmaq_n_f32(r0[base+1], b, a00); r1[base+1] = vfmaq_n_f32(r1[base+1], b, a10); }
                if (sub_len > 8)  { float32x4_t b = vld1q_f32(bk+8);  r0[base+2] = vfmaq_n_f32(r0[base+2], b, a00); r1[base+2] = vfmaq_n_f32(r1[base+2], b, a10); }
                if (sub_len > 12) { float32x4_t b = vld1q_f32(bk+12); r0[base+3] = vfmaq_n_f32(r0[base+3], b, a00); r1[base+3] = vfmaq_n_f32(r1[base+3], b, a10); }
            }
        }
    }

    // Store all sub-blocks
    for (int s = 0; s < n_subs; ++s) {
        int js = s * kSubBlockN;
        int sub_len = std::min(kSubBlockN, j_len - js);
        int base = s * 4;

        if (sub_len == 16) {
            store_16(C0 + j0 + js, r0[base], r0[base+1], r0[base+2], r0[base+3], alpha, beta);
            store_16(C1 + j0 + js, r1[base], r1[base+1], r1[base+2], r1[base+3], alpha, beta);
        } else {
            store_tail(C0 + j0 + js, sub_len, r0[base], r0[base+1], r0[base+2], r0[base+3], alpha, beta);
            store_tail(C1 + j0 + js, sub_len, r1[base], r1[base+1], r1[base+2], r1[base+3], alpha, beta);
        }
    }
}

// ============================================================
// M=4, PanelW=48: 16-col sub-block strategy
// 4 rows × 4 acc = 16 regs + 4 B = 20 total, fits in 32
// ============================================================

static void gemm_wide_4x16(int K,
                            const float* A0, const float* A1,
                            const float* A2, const float* A3,
                            const float* B, int ldb,
                            float* C0, float* C1, float* C2, float* C3,
                            int ldc, int j0, int js, int sub_len,
                            float alpha, float beta) {
    // 4 rows x 4 acc each = 16 accumulators
    float32x4_t r0c0 = vdupq_n_f32(0), r0c1 = vdupq_n_f32(0);
    float32x4_t r0c2 = vdupq_n_f32(0), r0c3 = vdupq_n_f32(0);
    float32x4_t r1c0 = vdupq_n_f32(0), r1c1 = vdupq_n_f32(0);
    float32x4_t r1c2 = vdupq_n_f32(0), r1c3 = vdupq_n_f32(0);
    float32x4_t r2c0 = vdupq_n_f32(0), r2c1 = vdupq_n_f32(0);
    float32x4_t r2c2 = vdupq_n_f32(0), r2c3 = vdupq_n_f32(0);
    float32x4_t r3c0 = vdupq_n_f32(0), r3c1 = vdupq_n_f32(0);
    float32x4_t r3c2 = vdupq_n_f32(0), r3c3 = vdupq_n_f32(0);

    // Kc-outer loop: limit B working set to Kc*16*4 bytes per block
    for (int pc = 0; pc < K; pc += kWideKc) {
        int kc = std::min(kWideKc, K - pc);

        int k = 0;
        // 4x K-unrolled main loop
        for (; k + 3 < kc; k += 4) {
            for (int ki = 0; ki < 4; ++ki) {
                int kk = pc + k + ki;
                const float* bk = B + kk * ldb + j0 + js;
                float a0 = A0[kk], a1 = A1[kk], a2 = A2[kk], a3 = A3[kk];

                float32x4_t b0 = vld1q_f32(bk);
                r0c0 = vfmaq_n_f32(r0c0, b0, a0);
                r1c0 = vfmaq_n_f32(r1c0, b0, a1);
                r2c0 = vfmaq_n_f32(r2c0, b0, a2);
                r3c0 = vfmaq_n_f32(r3c0, b0, a3);

                if (sub_len > 4) {
                    float32x4_t b1 = vld1q_f32(bk + 4);
                    r0c1 = vfmaq_n_f32(r0c1, b1, a0);
                    r1c1 = vfmaq_n_f32(r1c1, b1, a1);
                    r2c1 = vfmaq_n_f32(r2c1, b1, a2);
                    r3c1 = vfmaq_n_f32(r3c1, b1, a3);
                }
                if (sub_len > 8) {
                    float32x4_t b2 = vld1q_f32(bk + 8);
                    r0c2 = vfmaq_n_f32(r0c2, b2, a0);
                    r1c2 = vfmaq_n_f32(r1c2, b2, a1);
                    r2c2 = vfmaq_n_f32(r2c2, b2, a2);
                    r3c2 = vfmaq_n_f32(r3c2, b2, a3);
                }
                if (sub_len > 12) {
                    float32x4_t b3 = vld1q_f32(bk + 12);
                    r0c3 = vfmaq_n_f32(r0c3, b3, a0);
                    r1c3 = vfmaq_n_f32(r1c3, b3, a1);
                    r2c3 = vfmaq_n_f32(r2c3, b3, a2);
                    r3c3 = vfmaq_n_f32(r3c3, b3, a3);
                }
            }
        }
        // Scalar K tail within Kc block
        for (; k < kc; ++k) {
            int kk = pc + k;
            const float* bk = B + kk * ldb + j0 + js;
            float a0 = A0[kk], a1 = A1[kk], a2 = A2[kk], a3 = A3[kk];

            float32x4_t b0 = vld1q_f32(bk);
            r0c0 = vfmaq_n_f32(r0c0, b0, a0);
            r1c0 = vfmaq_n_f32(r1c0, b0, a1);
            r2c0 = vfmaq_n_f32(r2c0, b0, a2);
            r3c0 = vfmaq_n_f32(r3c0, b0, a3);

            if (sub_len > 4) {
                float32x4_t b1 = vld1q_f32(bk + 4);
                r0c1 = vfmaq_n_f32(r0c1, b1, a0);
                r1c1 = vfmaq_n_f32(r1c1, b1, a1);
                r2c1 = vfmaq_n_f32(r2c1, b1, a2);
                r3c1 = vfmaq_n_f32(r3c1, b1, a3);
            }
            if (sub_len > 8) {
                float32x4_t b2 = vld1q_f32(bk + 8);
                r0c2 = vfmaq_n_f32(r0c2, b2, a0);
                r1c2 = vfmaq_n_f32(r1c2, b2, a1);
                r2c2 = vfmaq_n_f32(r2c2, b2, a2);
                r3c2 = vfmaq_n_f32(r3c2, b2, a3);
            }
            if (sub_len > 12) {
                float32x4_t b3 = vld1q_f32(bk + 12);
                r0c3 = vfmaq_n_f32(r0c3, b3, a0);
                r1c3 = vfmaq_n_f32(r1c3, b3, a1);
                r2c3 = vfmaq_n_f32(r2c3, b3, a2);
                r3c3 = vfmaq_n_f32(r3c3, b3, a3);
            }
        }
    }

    if (sub_len == 16) {
        store_16(C0 + j0 + js, r0c0, r0c1, r0c2, r0c3, alpha, beta);
        store_16(C1 + j0 + js, r1c0, r1c1, r1c2, r1c3, alpha, beta);
        store_16(C2 + j0 + js, r2c0, r2c1, r2c2, r2c3, alpha, beta);
        store_16(C3 + j0 + js, r3c0, r3c1, r3c2, r3c3, alpha, beta);
    } else {
        store_tail(C0 + j0 + js, sub_len, r0c0, r0c1, r0c2, r0c3, alpha, beta);
        store_tail(C1 + j0 + js, sub_len, r1c0, r1c1, r1c2, r1c3, alpha, beta);
        store_tail(C2 + j0 + js, sub_len, r2c0, r2c1, r2c2, r2c3, alpha, beta);
        store_tail(C3 + j0 + js, sub_len, r3c0, r3c1, r3c2, r3c3, alpha, beta);
    }
}

static void gemm_wide_4x48(int K,
                            const float* A_rows[4],
                            const float* B, int ldb,
                            float* C_rows[4], int ldc,
                            int j0, int j_len,
                            float alpha, float beta) {
    // Process 48-col panel as 3 sub-blocks of 16 cols each
    for (int js = 0; js < j_len; js += kSubBlockN) {
        int sub_len = std::min(kSubBlockN, j_len - js);
        gemm_wide_4x16(K,
                       A_rows[0], A_rows[1], A_rows[2], A_rows[3],
                       B, ldb,
                       C_rows[0], C_rows[1], C_rows[2], C_rows[3],
                       ldc, j0, js, sub_len, alpha, beta);
    }
}

// ============================================================
// General MxN kernel with 16-col sub-blocks (M=3,5,6,7)
// ============================================================

static void gemm_wide_MxN(int M, int K,
                           const float** A_rows,
                           const float* B, int ldb,
                           float** C_rows, int ldc,
                           int j0, int j_len,
                           float alpha, float beta) {
    int sub_full4 = (j_len / 4) * 4;

    // Allocate stack-backed accumulators: M rows x ceil(j_len/4) SIMD vectors
    int n_acc = (j_len + 3) / 4;
    float32x4_t* acc = reinterpret_cast<float32x4_t*>(
        __builtin_alloca(M * n_acc * sizeof(float32x4_t)));
    for (int i = 0; i < M * n_acc; ++i)
        acc[i] = vdupq_n_f32(0);

    // Kc-outer, K-inner loop: B[Kc][j_len] fits L1D
    for (int pc = 0; pc < K; pc += kWideKc) {
        int kc = std::min(kWideKc, K - pc);

        for (int k = 0; k < kc; ++k) {
            int kk = pc + k;
            const float* bk = B + kk * ldb + j0;
            for (int j4 = 0; j4 < sub_full4; j4 += 4) {
                float32x4_t bv = vld1q_f32(bk + j4);
                int aidx = j4 / 4;
                for (int i = 0; i < M; ++i) {
                    float aik = A_rows[i][kk];
                    acc[i * n_acc + aidx] = vfmaq_n_f32(acc[i * n_acc + aidx], bv, aik);
                }
            }
        }
    }

    // Store
    float32x4_t av = vdupq_n_f32(alpha);
    for (int i = 0; i < M; ++i) {
        float* c_row = C_rows[i] + j0;
        for (int j4 = 0; j4 < sub_full4; j4 += 4) {
            int aidx = j4 / 4;
            if (beta == 0.0f) {
                vst1q_f32(c_row + j4, vmulq_f32(av, acc[i * n_acc + aidx]));
            } else {
                float32x4_t bv = vdupq_n_f32(beta);
                vst1q_f32(c_row + j4,
                    vfmaq_f32(vmulq_f32(bv, vld1q_f32(c_row + j4)), av, acc[i * n_acc + aidx]));
            }
        }
        // Scalar tail for N % 4
        for (int j = sub_full4; j < j_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A_rows[i][k] * B[k * ldb + j0 + j];
            c_row[j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * c_row[j];
        }
    }
}

// ============================================================
// Public driver
// ============================================================

void gemm_smallm_wide_driver_fp32(int M, int N, int K,
                                   float alpha, const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta, float* C, int ldc) {
    constexpr int kPanelN = kWidePanelN;  // 48

    int i = 0;

    // Process M in groups of 4
    for (; i + 3 < M; i += 4) {
        const float* a_rows[4] = {A + i*lda, A + (i+1)*lda, A + (i+2)*lda, A + (i+3)*lda};
        float* c_rows[4] = {C + i*ldc, C + (i+1)*ldc, C + (i+2)*ldc, C + (i+3)*ldc};

        for (int j0 = 0; j0 < N; j0 += kPanelN) {
            int j_len = std::min(kPanelN, N - j0);
            gemm_wide_4x48(K, a_rows, B, ldb, c_rows, ldc, j0, j_len, alpha, beta);
        }
    }

    // Process M in groups of 2
    for (; i + 1 < M; i += 2) {
        for (int j0 = 0; j0 < N; j0 += kPanelN) {
            int j_len = std::min(kPanelN, N - j0);
            gemm_wide_2x48(K,
                           A + i*lda, A + (i+1)*lda,
                           B, ldb,
                           C + i*ldc, C + (i+1)*ldc, ldc,
                           j0, j_len, alpha, beta);
        }
    }

    // Remaining single row: use inline M=1 GEMV with Kc blocking
    if (i < M) {
        constexpr int kGemvPanel = 64;
        const float* a_row = A + i * lda;
        float* c_row = C + i * ldc;

        for (int j0 = 0; j0 < N; j0 += kGemvPanel) {
            int j_len = std::min(kGemvPanel, N - j0);
            float32x4_t acc[16];
            for (int x = 0; x < 16; ++x) acc[x] = vdupq_n_f32(0);

            // Kc blocking
            for (int pc = 0; pc < K; pc += kWideKc) {
                int kc = std::min(kWideKc, K - pc);
                for (int k = 0; k < kc; ++k) {
                    int kk = pc + k;
                    float32x4_t av = vdupq_n_f32(a_row[kk]);
                    const float* bk = B + kk * ldb + j0;
                    for (int j = 0; j + 3 < j_len; j += 4)
                        acc[j/4] = vfmaq_f32(acc[j/4], av, vld1q_f32(bk + j));
                }
            }

            float32x4_t av = vdupq_n_f32(alpha);
            float32x4_t bv = vdupq_n_f32(beta);
            for (int j = 0; j + 3 < j_len; j += 4) {
                if (beta == 0.0f)
                    vst1q_f32(c_row + j0 + j, vmulq_f32(av, acc[j/4]));
                else
                    vst1q_f32(c_row + j0 + j,
                        vfmaq_f32(vmulq_f32(bv, vld1q_f32(c_row + j0 + j)), av, acc[j/4]));
            }
            for (int j = (j_len/4)*4; j < j_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                    sum += a_row[k] * B[k*ldb + j0 + j];
                c_row[j0+j] = (beta == 0.0f) ? alpha*sum : alpha*sum + beta*c_row[j0+j];
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
