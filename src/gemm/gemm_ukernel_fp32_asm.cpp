/// @file gemm_ukernel_fp32_asm.cpp
/// Packed 4x16 FP32 inline assembly micro-kernel for BLIS-style GEMM registry.
///
/// Uses autoGEMM techniques on packed contiguous A/B layouts:
///   packed_A: for each k, 4 contiguous floats (Mr=4 rows) → ldr q, use .s[0..3]
///   packed_B: for each k, 16 contiguous floats (Nr=16 cols) → 4 × ldr q
///
/// 8x K-unrolling: 8 A loads + 8 B loads per iteration, 128 FMLAs.
/// Registers: v0-v7 A values, v8-v11 B panel, v12-v27 accumulators.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#include <algorithm>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>

namespace dnnopt {

// ============================================================
// Pack functions for Mr=4, Nr=16
// ============================================================

static void pack_a_fp32_4x16(int m_len, int k_len,
                              const float* A, int lda,
                              float* packed_A) {
    constexpr int Mr = 4;
    for (int i = 0; i < m_len; i += Mr) {
        int m_rem = std::min(Mr, m_len - i);
        if (m_rem == Mr) {
            for (int k = 0; k < k_len; ++k) {
                packed_A[0] = A[(i + 0) * lda + k];
                packed_A[1] = A[(i + 1) * lda + k];
                packed_A[2] = A[(i + 2) * lda + k];
                packed_A[3] = A[(i + 3) * lda + k];
                packed_A += 4;
            }
        } else {
            for (int k = 0; k < k_len; ++k) {
                int r = 0;
                for (; r < m_rem; ++r)
                    packed_A[r] = A[(i + r) * lda + k];
                for (; r < Mr; ++r)
                    packed_A[r] = 0.0f;
                packed_A += 4;
            }
        }
    }
}

static void pack_b_fp32_4x16(int k_len, int n_len,
                              const float* B, int ldb,
                              float* packed_B) {
    constexpr int Nr = 16;
    for (int j = 0; j < n_len; j += Nr) {
        int n_rem = std::min(Nr, n_len - j);
        if (n_rem == Nr) {
            for (int k = 0; k < k_len; ++k) {
                const float* src = &B[k * ldb + j];
#ifdef __ARM_NEON
                vst1q_f32(packed_B,      vld1q_f32(src));
                vst1q_f32(packed_B + 4,  vld1q_f32(src + 4));
                vst1q_f32(packed_B + 8,  vld1q_f32(src + 8));
                vst1q_f32(packed_B + 12, vld1q_f32(src + 12));
#else
                memcpy(packed_B, src, Nr * sizeof(float));
#endif
                packed_B += Nr;
            }
        } else {
            for (int k = 0; k < k_len; ++k) {
                const float* src = &B[k * ldb + j];
                int c = 0;
                for (; c < n_rem; ++c)
                    packed_B[c] = src[c];
                for (; c < Nr; ++c)
                    packed_B[c] = 0.0f;
                packed_B += Nr;
            }
        }
    }
}

// ============================================================
// Packed 4x16 inline assembly micro-kernel
// ============================================================
//
// Register allocation:
//   v0-v7:   A values for K steps 0-7 (each q holds 4 rows, use .s[0..3])
//   v8-v11:  B panel (4 × float32x4 = 16 cols)
//   v12-v27: 16 accumulators (row0: v12-v15, row1: v16-v19,
//                              row2: v20-v23, row3: v24-v27)
//
//   x11: packed_A pointer (advances by 16 per K step)
//   x22: packed_B pointer (advances by 64 per K step)
//
// Sequential memory access, no stride calculations.

static void gemm_ukernel_fp32_packed_4x16_asm(
        int K,
        const float* __restrict__ packed_A,
        const float* __restrict__ packed_B,
        float* __restrict__ C, int ldc,
        float alpha, float beta) {

    int k_main = K / 8;
    int k_tail = K % 8;
    int ldc_val = ldc;

    asm volatile(
    // === Convert ldc to bytes ===
    "lsl     %w[ldc], %w[ldc], #2              \n"
    "sxtw    x17, %w[ldc]                      \n"

    // === Setup C row pointers ===
    "mov     x6,  %[C]                         \n"
    "add     x7,  x6, x17                      \n"
    "add     x8,  x6, x17, lsl #1              \n"
    "add     x9,  x7, x17, lsl #1              \n"

    // === Setup packed pointers ===
    "mov     x11, %[pA]                        \n"
    "mov     x22, %[pB]                        \n"

    // === Zero accumulators ===
    "movi    v12.4s, #0 \n"  "movi    v13.4s, #0 \n"
    "movi    v14.4s, #0 \n"  "movi    v15.4s, #0 \n"
    "movi    v16.4s, #0 \n"  "movi    v17.4s, #0 \n"
    "movi    v18.4s, #0 \n"  "movi    v19.4s, #0 \n"
    "movi    v20.4s, #0 \n"  "movi    v21.4s, #0 \n"
    "movi    v22.4s, #0 \n"  "movi    v23.4s, #0 \n"
    "movi    v24.4s, #0 \n"  "movi    v25.4s, #0 \n"
    "movi    v26.4s, #0 \n"  "movi    v27.4s, #0 \n"

    // === Main loop: 8 K values per iteration ===
    "cbz     %w[k_main], 20f                   \n"

    ".p2align 4                                \n"
    "10:                                       \n"

    // Load A[k0] and B[k0]
    "ldr     q0, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+0: 16 FMLAs
    "fmla    v12.4s, v8.4s,  v0.s[0] \n"  "fmla    v13.4s, v9.4s,  v0.s[0] \n"
    "fmla    v14.4s, v10.4s, v0.s[0] \n"  "fmla    v15.4s, v11.4s, v0.s[0] \n"
    "fmla    v16.4s, v8.4s,  v0.s[1] \n"  "fmla    v17.4s, v9.4s,  v0.s[1] \n"
    "fmla    v18.4s, v10.4s, v0.s[1] \n"  "fmla    v19.4s, v11.4s, v0.s[1] \n"
    "fmla    v20.4s, v8.4s,  v0.s[2] \n"  "fmla    v21.4s, v9.4s,  v0.s[2] \n"
    "fmla    v22.4s, v10.4s, v0.s[2] \n"  "fmla    v23.4s, v11.4s, v0.s[2] \n"
    "fmla    v24.4s, v8.4s,  v0.s[3] \n"  "fmla    v25.4s, v9.4s,  v0.s[3] \n"
    "fmla    v26.4s, v10.4s, v0.s[3] \n"  "fmla    v27.4s, v11.4s, v0.s[3] \n"

    // Load A[k1] and B[k1]
    "ldr     q1, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+1
    "fmla    v12.4s, v8.4s,  v1.s[0] \n"  "fmla    v13.4s, v9.4s,  v1.s[0] \n"
    "fmla    v14.4s, v10.4s, v1.s[0] \n"  "fmla    v15.4s, v11.4s, v1.s[0] \n"
    "fmla    v16.4s, v8.4s,  v1.s[1] \n"  "fmla    v17.4s, v9.4s,  v1.s[1] \n"
    "fmla    v18.4s, v10.4s, v1.s[1] \n"  "fmla    v19.4s, v11.4s, v1.s[1] \n"
    "fmla    v20.4s, v8.4s,  v1.s[2] \n"  "fmla    v21.4s, v9.4s,  v1.s[2] \n"
    "fmla    v22.4s, v10.4s, v1.s[2] \n"  "fmla    v23.4s, v11.4s, v1.s[2] \n"
    "fmla    v24.4s, v8.4s,  v1.s[3] \n"  "fmla    v25.4s, v9.4s,  v1.s[3] \n"
    "fmla    v26.4s, v10.4s, v1.s[3] \n"  "fmla    v27.4s, v11.4s, v1.s[3] \n"

    // Load A[k2] and B[k2]
    "ldr     q2, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+2
    "fmla    v12.4s, v8.4s,  v2.s[0] \n"  "fmla    v13.4s, v9.4s,  v2.s[0] \n"
    "fmla    v14.4s, v10.4s, v2.s[0] \n"  "fmla    v15.4s, v11.4s, v2.s[0] \n"
    "fmla    v16.4s, v8.4s,  v2.s[1] \n"  "fmla    v17.4s, v9.4s,  v2.s[1] \n"
    "fmla    v18.4s, v10.4s, v2.s[1] \n"  "fmla    v19.4s, v11.4s, v2.s[1] \n"
    "fmla    v20.4s, v8.4s,  v2.s[2] \n"  "fmla    v21.4s, v9.4s,  v2.s[2] \n"
    "fmla    v22.4s, v10.4s, v2.s[2] \n"  "fmla    v23.4s, v11.4s, v2.s[2] \n"
    "fmla    v24.4s, v8.4s,  v2.s[3] \n"  "fmla    v25.4s, v9.4s,  v2.s[3] \n"
    "fmla    v26.4s, v10.4s, v2.s[3] \n"  "fmla    v27.4s, v11.4s, v2.s[3] \n"

    // Load A[k3] and B[k3]
    "ldr     q3, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+3
    "fmla    v12.4s, v8.4s,  v3.s[0] \n"  "fmla    v13.4s, v9.4s,  v3.s[0] \n"
    "fmla    v14.4s, v10.4s, v3.s[0] \n"  "fmla    v15.4s, v11.4s, v3.s[0] \n"
    "fmla    v16.4s, v8.4s,  v3.s[1] \n"  "fmla    v17.4s, v9.4s,  v3.s[1] \n"
    "fmla    v18.4s, v10.4s, v3.s[1] \n"  "fmla    v19.4s, v11.4s, v3.s[1] \n"
    "fmla    v20.4s, v8.4s,  v3.s[2] \n"  "fmla    v21.4s, v9.4s,  v3.s[2] \n"
    "fmla    v22.4s, v10.4s, v3.s[2] \n"  "fmla    v23.4s, v11.4s, v3.s[2] \n"
    "fmla    v24.4s, v8.4s,  v3.s[3] \n"  "fmla    v25.4s, v9.4s,  v3.s[3] \n"
    "fmla    v26.4s, v10.4s, v3.s[3] \n"  "fmla    v27.4s, v11.4s, v3.s[3] \n"

    // Load A[k4] and B[k4]
    "ldr     q4, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+4
    "fmla    v12.4s, v8.4s,  v4.s[0] \n"  "fmla    v13.4s, v9.4s,  v4.s[0] \n"
    "fmla    v14.4s, v10.4s, v4.s[0] \n"  "fmla    v15.4s, v11.4s, v4.s[0] \n"
    "fmla    v16.4s, v8.4s,  v4.s[1] \n"  "fmla    v17.4s, v9.4s,  v4.s[1] \n"
    "fmla    v18.4s, v10.4s, v4.s[1] \n"  "fmla    v19.4s, v11.4s, v4.s[1] \n"
    "fmla    v20.4s, v8.4s,  v4.s[2] \n"  "fmla    v21.4s, v9.4s,  v4.s[2] \n"
    "fmla    v22.4s, v10.4s, v4.s[2] \n"  "fmla    v23.4s, v11.4s, v4.s[2] \n"
    "fmla    v24.4s, v8.4s,  v4.s[3] \n"  "fmla    v25.4s, v9.4s,  v4.s[3] \n"
    "fmla    v26.4s, v10.4s, v4.s[3] \n"  "fmla    v27.4s, v11.4s, v4.s[3] \n"

    // Load A[k5] and B[k5]
    "ldr     q5, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+5
    "fmla    v12.4s, v8.4s,  v5.s[0] \n"  "fmla    v13.4s, v9.4s,  v5.s[0] \n"
    "fmla    v14.4s, v10.4s, v5.s[0] \n"  "fmla    v15.4s, v11.4s, v5.s[0] \n"
    "fmla    v16.4s, v8.4s,  v5.s[1] \n"  "fmla    v17.4s, v9.4s,  v5.s[1] \n"
    "fmla    v18.4s, v10.4s, v5.s[1] \n"  "fmla    v19.4s, v11.4s, v5.s[1] \n"
    "fmla    v20.4s, v8.4s,  v5.s[2] \n"  "fmla    v21.4s, v9.4s,  v5.s[2] \n"
    "fmla    v22.4s, v10.4s, v5.s[2] \n"  "fmla    v23.4s, v11.4s, v5.s[2] \n"
    "fmla    v24.4s, v8.4s,  v5.s[3] \n"  "fmla    v25.4s, v9.4s,  v5.s[3] \n"
    "fmla    v26.4s, v10.4s, v5.s[3] \n"  "fmla    v27.4s, v11.4s, v5.s[3] \n"

    // Load A[k6] and B[k6]
    "ldr     q6, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+6
    "fmla    v12.4s, v8.4s,  v6.s[0] \n"  "fmla    v13.4s, v9.4s,  v6.s[0] \n"
    "fmla    v14.4s, v10.4s, v6.s[0] \n"  "fmla    v15.4s, v11.4s, v6.s[0] \n"
    "fmla    v16.4s, v8.4s,  v6.s[1] \n"  "fmla    v17.4s, v9.4s,  v6.s[1] \n"
    "fmla    v18.4s, v10.4s, v6.s[1] \n"  "fmla    v19.4s, v11.4s, v6.s[1] \n"
    "fmla    v20.4s, v8.4s,  v6.s[2] \n"  "fmla    v21.4s, v9.4s,  v6.s[2] \n"
    "fmla    v22.4s, v10.4s, v6.s[2] \n"  "fmla    v23.4s, v11.4s, v6.s[2] \n"
    "fmla    v24.4s, v8.4s,  v6.s[3] \n"  "fmla    v25.4s, v9.4s,  v6.s[3] \n"
    "fmla    v26.4s, v10.4s, v6.s[3] \n"  "fmla    v27.4s, v11.4s, v6.s[3] \n"

    // Load A[k7] and B[k7]
    "ldr     q7, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"
    // K+7
    "fmla    v12.4s, v8.4s,  v7.s[0] \n"  "fmla    v13.4s, v9.4s,  v7.s[0] \n"
    "fmla    v14.4s, v10.4s, v7.s[0] \n"  "fmla    v15.4s, v11.4s, v7.s[0] \n"
    "fmla    v16.4s, v8.4s,  v7.s[1] \n"  "fmla    v17.4s, v9.4s,  v7.s[1] \n"
    "fmla    v18.4s, v10.4s, v7.s[1] \n"  "fmla    v19.4s, v11.4s, v7.s[1] \n"
    "fmla    v20.4s, v8.4s,  v7.s[2] \n"  "fmla    v21.4s, v9.4s,  v7.s[2] \n"
    "fmla    v22.4s, v10.4s, v7.s[2] \n"  "fmla    v23.4s, v11.4s, v7.s[2] \n"
    "fmla    v24.4s, v8.4s,  v7.s[3] \n"  "fmla    v25.4s, v9.4s,  v7.s[3] \n"
    "fmla    v26.4s, v10.4s, v7.s[3] \n"  "fmla    v27.4s, v11.4s, v7.s[3] \n"

    "subs    %w[k_main], %w[k_main], #1        \n"
    "bne     10b                               \n"

    // === K tail: 1 at a time ===
    "20:                                       \n"
    "cbz     %w[k_tail], 30f                   \n"

    "21:                                       \n"
    "ldr     q0, [x11], #16                    \n"
    "ldr     q8,  [x22]       \n"  "ldr     q9,  [x22, #16]  \n"
    "ldr     q10, [x22, #32]  \n"  "ldr     q11, [x22, #48]  \n"
    "add     x22, x22, #64                     \n"

    "fmla    v12.4s, v8.4s,  v0.s[0] \n"  "fmla    v13.4s, v9.4s,  v0.s[0] \n"
    "fmla    v14.4s, v10.4s, v0.s[0] \n"  "fmla    v15.4s, v11.4s, v0.s[0] \n"
    "fmla    v16.4s, v8.4s,  v0.s[1] \n"  "fmla    v17.4s, v9.4s,  v0.s[1] \n"
    "fmla    v18.4s, v10.4s, v0.s[1] \n"  "fmla    v19.4s, v11.4s, v0.s[1] \n"
    "fmla    v20.4s, v8.4s,  v0.s[2] \n"  "fmla    v21.4s, v9.4s,  v0.s[2] \n"
    "fmla    v22.4s, v10.4s, v0.s[2] \n"  "fmla    v23.4s, v11.4s, v0.s[2] \n"
    "fmla    v24.4s, v8.4s,  v0.s[3] \n"  "fmla    v25.4s, v9.4s,  v0.s[3] \n"
    "fmla    v26.4s, v10.4s, v0.s[3] \n"  "fmla    v27.4s, v11.4s, v0.s[3] \n"

    "subs    %w[k_tail], %w[k_tail], #1        \n"
    "bne     21b                               \n"

    // === Epilogue: C = alpha * acc + beta * C ===
    "30:                                       \n"
    "ld1r    {v0.4s}, [%[alpha_ptr]]           \n"
    "ldr     s1, [%[beta_ptr]]                 \n"
    "fcmp    s1, #0.0                          \n"
    "beq     40f                               \n"

    // beta != 0
    "dup     v1.4s, v1.s[0]                    \n"

    // Row 0
    "ldr     q2, [x6]      \n"  "ldr     q3, [x6, #16]  \n"
    "ldr     q4, [x6, #32] \n"  "ldr     q5, [x6, #48]  \n"
    "fmul    v12.4s, v12.4s, v0.4s \n"  "fmla    v12.4s, v2.4s, v1.4s \n"
    "fmul    v13.4s, v13.4s, v0.4s \n"  "fmla    v13.4s, v3.4s, v1.4s \n"
    "fmul    v14.4s, v14.4s, v0.4s \n"  "fmla    v14.4s, v4.4s, v1.4s \n"
    "fmul    v15.4s, v15.4s, v0.4s \n"  "fmla    v15.4s, v5.4s, v1.4s \n"
    "str     q12, [x6]      \n"  "str     q13, [x6, #16]  \n"
    "str     q14, [x6, #32] \n"  "str     q15, [x6, #48]  \n"

    // Row 1
    "ldr     q2, [x7]      \n"  "ldr     q3, [x7, #16]  \n"
    "ldr     q4, [x7, #32] \n"  "ldr     q5, [x7, #48]  \n"
    "fmul    v16.4s, v16.4s, v0.4s \n"  "fmla    v16.4s, v2.4s, v1.4s \n"
    "fmul    v17.4s, v17.4s, v0.4s \n"  "fmla    v17.4s, v3.4s, v1.4s \n"
    "fmul    v18.4s, v18.4s, v0.4s \n"  "fmla    v18.4s, v4.4s, v1.4s \n"
    "fmul    v19.4s, v19.4s, v0.4s \n"  "fmla    v19.4s, v5.4s, v1.4s \n"
    "str     q16, [x7]      \n"  "str     q17, [x7, #16]  \n"
    "str     q18, [x7, #32] \n"  "str     q19, [x7, #48]  \n"

    // Row 2
    "ldr     q2, [x8]      \n"  "ldr     q3, [x8, #16]  \n"
    "ldr     q4, [x8, #32] \n"  "ldr     q5, [x8, #48]  \n"
    "fmul    v20.4s, v20.4s, v0.4s \n"  "fmla    v20.4s, v2.4s, v1.4s \n"
    "fmul    v21.4s, v21.4s, v0.4s \n"  "fmla    v21.4s, v3.4s, v1.4s \n"
    "fmul    v22.4s, v22.4s, v0.4s \n"  "fmla    v22.4s, v4.4s, v1.4s \n"
    "fmul    v23.4s, v23.4s, v0.4s \n"  "fmla    v23.4s, v5.4s, v1.4s \n"
    "str     q20, [x8]      \n"  "str     q21, [x8, #16]  \n"
    "str     q22, [x8, #32] \n"  "str     q23, [x8, #48]  \n"

    // Row 3
    "ldr     q2, [x9]      \n"  "ldr     q3, [x9, #16]  \n"
    "ldr     q4, [x9, #32] \n"  "ldr     q5, [x9, #48]  \n"
    "fmul    v24.4s, v24.4s, v0.4s \n"  "fmla    v24.4s, v2.4s, v1.4s \n"
    "fmul    v25.4s, v25.4s, v0.4s \n"  "fmla    v25.4s, v3.4s, v1.4s \n"
    "fmul    v26.4s, v26.4s, v0.4s \n"  "fmla    v26.4s, v4.4s, v1.4s \n"
    "fmul    v27.4s, v27.4s, v0.4s \n"  "fmla    v27.4s, v5.4s, v1.4s \n"
    "str     q24, [x9]      \n"  "str     q25, [x9, #16]  \n"
    "str     q26, [x9, #32] \n"  "str     q27, [x9, #48]  \n"
    "b       50f                               \n"

    // beta == 0
    "40:                                       \n"
    "fmul    v12.4s, v12.4s, v0.4s \n"  "fmul    v13.4s, v13.4s, v0.4s \n"
    "fmul    v14.4s, v14.4s, v0.4s \n"  "fmul    v15.4s, v15.4s, v0.4s \n"
    "str     q12, [x6]      \n"  "str     q13, [x6, #16]  \n"
    "str     q14, [x6, #32] \n"  "str     q15, [x6, #48]  \n"

    "fmul    v16.4s, v16.4s, v0.4s \n"  "fmul    v17.4s, v17.4s, v0.4s \n"
    "fmul    v18.4s, v18.4s, v0.4s \n"  "fmul    v19.4s, v19.4s, v0.4s \n"
    "str     q16, [x7]      \n"  "str     q17, [x7, #16]  \n"
    "str     q18, [x7, #32] \n"  "str     q19, [x7, #48]  \n"

    "fmul    v20.4s, v20.4s, v0.4s \n"  "fmul    v21.4s, v21.4s, v0.4s \n"
    "fmul    v22.4s, v22.4s, v0.4s \n"  "fmul    v23.4s, v23.4s, v0.4s \n"
    "str     q20, [x8]      \n"  "str     q21, [x8, #16]  \n"
    "str     q22, [x8, #32] \n"  "str     q23, [x8, #48]  \n"

    "fmul    v24.4s, v24.4s, v0.4s \n"  "fmul    v25.4s, v25.4s, v0.4s \n"
    "fmul    v26.4s, v26.4s, v0.4s \n"  "fmul    v27.4s, v27.4s, v0.4s \n"
    "str     q24, [x9]      \n"  "str     q25, [x9, #16]  \n"
    "str     q26, [x9, #32] \n"  "str     q27, [x9, #48]  \n"

    "50:                                       \n"

    : [k_main] "+r" (k_main),
      [k_tail] "+r" (k_tail),
      [ldc] "+r" (ldc_val)
    : [pA] "r" (packed_A),
      [pB] "r" (packed_B),
      [C] "r" (C),
      [alpha_ptr] "r" (&alpha),
      [beta_ptr] "r" (&beta)
    : "cc", "memory",
      "x6", "x7", "x8", "x9",
      "x11", "x17", "x22",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
      "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
      "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
    );
}

// ============================================================
// Registry wrappers + auto-registration
// ============================================================

namespace {

void ukernel_fp32_asm_4x16_wrap(int K, const void* packed_A, const void* packed_B,
                                 float* C, int ldc, float alpha, float beta,
                                 float /*extra*/) {
    gemm_ukernel_fp32_packed_4x16_asm(
        K,
        static_cast<const float*>(packed_A),
        static_cast<const float*>(packed_B),
        C, ldc, alpha, beta);
}

void pack_a_fp32_4x16_wrap(int m_len, int k_len, const float* A, int lda,
                            void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32_4x16(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

void pack_b_fp32_4x16_wrap(int k_len, int n_len, const float* B, int ldb,
                            void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32_4x16(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

const GemmMicrokernelDesc neon_fp32_asm_4x16_desc = {
    "neon_fp32_asm_4x16",
    GemmDataType::kFP32,
    kNEON,                // required_hwcaps
    4,                    // Mr = 4
    16,                   // Nr = 16
    1,                    // Kgroup
    false,                // nr_is_vla
    110,                  // priority (higher than 8x12's 100)
    sizeof(float),        // packed_a_elem_bytes
    sizeof(float),        // packed_b_elem_bytes
    0,                    // min_sve_bits
    ukernel_fp32_asm_4x16_wrap,
    pack_a_fp32_4x16_wrap,
    pack_b_fp32_4x16_wrap,
};

static RegisterKernel reg_neon_fp32_asm_4x16(neon_fp32_asm_4x16_desc);

}  // namespace

}  // namespace dnnopt

#endif  // __aarch64__
