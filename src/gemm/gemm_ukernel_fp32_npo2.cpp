/// @file gemm_ukernel_fp32_npo2.cpp
/// Clang-optimized NEON intrinsics GEMM kernels for non-power-of-2 M values.
///
/// Techniques from IAAT/IrGEMM papers:
///   - Vector A loads (4 K-values per ldr q) with .s[0..3] extraction
///   - 4x K-unrolling via .s[N] FMLA (fused broadcast + multiply-add)
///   - PRFM prefetch for B panels
///
/// Register budgets:
///   M=3: 12 acc + 4 B + 2 A = 18 regs (plenty of room)
///   M=5: 20 acc + 4 B + 2 A = 26 regs (fits in 32)
///   M=7: 28 acc + 4 B + 1 A = 33 regs (2-pass: 14+4+1=19 per pass)
///
/// These replace the scalar-A-load asm kernels in gemm_kernel_asm_fp32.cpp
/// for unpacked (adaptive tile) paths. For packed paths, the existing
/// 4x16 and 8x16 kernels handle M%4 and M%8 tails.

#include "dnnopt/gemm/gemm_config.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __aarch64__

namespace dnnopt {

// ============================================================
// 3×16 Clang .s[N] kernel — 12 FMLAs per K, 4x K-unroll
// ============================================================
// Register budget: 12 acc (v0-v11) + 4 B (v12-v15) + 1 A (v16) = 17

void gemm_kernel_3x16_lane(int K,
                             const float* __restrict__ A, int lda,
                             const float* __restrict__ B, int ldb,
                             float* __restrict__ C, int ldc,
                             float alpha, float beta) {
    // Accumulators: 3 rows × 4 B-quads = 12
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);

    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* b_ptr = B;

    // Prefetch offset: 8*K-rows of B = 8*ldb*4 bytes
    int pf_b_offset = 8 * ldb * 4;

    // 4x K-unrolled main loop
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Prefetch B
        __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(b_ptr + pf_b_offset / 4) : "memory");

        // Load 4 B quads (16 cols)
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);
        float32x4_t b2 = vld1q_f32(b_ptr + 8);
        float32x4_t b3 = vld1q_f32(b_ptr + 12);
        b_ptr += ldb;

        // K+0: load A as scalars, FMLA with B
        {
            float32x4_t ar0 = {a0[0], a1[0], a2[0], 0}; a0++; a1++; a2++;
            c00 = vfmaq_laneq_f32(c00, b0, ar0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, ar0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, ar0, 0);
            c03 = vfmaq_laneq_f32(c03, b3, ar0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, ar0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, ar0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, ar0, 1);
            c13 = vfmaq_laneq_f32(c13, b3, ar0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, ar0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, ar0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, ar0, 2);
            c23 = vfmaq_laneq_f32(c23, b3, ar0, 2);
        }

        // K+1
        {
            float32x4_t b0 = vld1q_f32(b_ptr);
            float32x4_t b1 = vld1q_f32(b_ptr + 4);
            float32x4_t b2 = vld1q_f32(b_ptr + 8);
            float32x4_t b3 = vld1q_f32(b_ptr + 12);
            b_ptr += ldb;

            float32x4_t ar0 = {a0[0], a1[0], a2[0], 0}; a0++; a1++; a2++;
            c00 = vfmaq_laneq_f32(c00, b0, ar0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, ar0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, ar0, 0);
            c03 = vfmaq_laneq_f32(c03, b3, ar0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, ar0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, ar0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, ar0, 1);
            c13 = vfmaq_laneq_f32(c13, b3, ar0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, ar0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, ar0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, ar0, 2);
            c23 = vfmaq_laneq_f32(c23, b3, ar0, 2);
        }

        // K+2
        {
            float32x4_t b0 = vld1q_f32(b_ptr);
            float32x4_t b1 = vld1q_f32(b_ptr + 4);
            float32x4_t b2 = vld1q_f32(b_ptr + 8);
            float32x4_t b3 = vld1q_f32(b_ptr + 12);
            b_ptr += ldb;

            float32x4_t ar0 = {a0[0], a1[0], a2[0], 0}; a0++; a1++; a2++;
            c00 = vfmaq_laneq_f32(c00, b0, ar0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, ar0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, ar0, 0);
            c03 = vfmaq_laneq_f32(c03, b3, ar0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, ar0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, ar0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, ar0, 1);
            c13 = vfmaq_laneq_f32(c13, b3, ar0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, ar0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, ar0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, ar0, 2);
            c23 = vfmaq_laneq_f32(c23, b3, ar0, 2);
        }

        // K+3
        {
            float32x4_t b0 = vld1q_f32(b_ptr);
            float32x4_t b1 = vld1q_f32(b_ptr + 4);
            float32x4_t b2 = vld1q_f32(b_ptr + 8);
            float32x4_t b3 = vld1q_f32(b_ptr + 12);
            b_ptr += ldb;

            float32x4_t ar0 = {a0[0], a1[0], a2[0], 0}; a0++; a1++; a2++;
            c00 = vfmaq_laneq_f32(c00, b0, ar0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, ar0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, ar0, 0);
            c03 = vfmaq_laneq_f32(c03, b3, ar0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, ar0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, ar0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, ar0, 1);
            c13 = vfmaq_laneq_f32(c13, b3, ar0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, ar0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, ar0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, ar0, 2);
            c23 = vfmaq_laneq_f32(c23, b3, ar0, 2);
        }
    }

    // K tail: 1 at a time
    for (; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);
        float32x4_t b2 = vld1q_f32(b_ptr + 8);
        float32x4_t b3 = vld1q_f32(b_ptr + 12);
        b_ptr += ldb;

        float32x4_t ar0 = {a0[0], a1[0], a2[0], 0}; a0++; a1++; a2++;
        c00 = vfmaq_laneq_f32(c00, b0, ar0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, ar0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, ar0, 0);
        c03 = vfmaq_laneq_f32(c03, b3, ar0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, ar0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, ar0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, ar0, 1);
        c13 = vfmaq_laneq_f32(c13, b3, ar0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, ar0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, ar0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, ar0, 2);
        c23 = vfmaq_laneq_f32(c23, b3, ar0, 2);
    }

    // Epilogue
    float32x4_t av = vdupq_n_f32(alpha);
    auto store_row = [&](float* cr, float32x4_t r0, float32x4_t r1,
                          float32x4_t r2, float32x4_t r3) {
        if (beta == 0.0f) {
            vst1q_f32(cr,      vmulq_f32(av, r0));
            vst1q_f32(cr + 4,  vmulq_f32(av, r1));
            vst1q_f32(cr + 8,  vmulq_f32(av, r2));
            vst1q_f32(cr + 12, vmulq_f32(av, r3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(cr,      vfmaq_f32(vmulq_f32(av, r0), bv, vld1q_f32(cr)));
            vst1q_f32(cr + 4,  vfmaq_f32(vmulq_f32(av, r1), bv, vld1q_f32(cr + 4)));
            vst1q_f32(cr + 8,  vfmaq_f32(vmulq_f32(av, r2), bv, vld1q_f32(cr + 8)));
            vst1q_f32(cr + 12, vfmaq_f32(vmulq_f32(av, r3), bv, vld1q_f32(cr + 12)));
        }
    };

    store_row(C,             c00, c01, c02, c03);
    store_row(C + ldc,       c10, c11, c12, c13);
    store_row(C + 2 * ldc,   c20, c21, c22, c23);
}

// ============================================================
// 5×16 Clang .s[N] kernel — 20 FMLAs per K, 4x K-unroll
// ============================================================
// Register budget: 20 acc + 4 B + 1 A = 25 regs (fits in 32)

void gemm_kernel_5x16_lane(int K,
                             const float* __restrict__ A, int lda,
                             const float* __restrict__ B, int ldb,
                             float* __restrict__ C, int ldc,
                             float alpha, float beta) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c42 = vdupq_n_f32(0), c43 = vdupq_n_f32(0);

    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;
    const float* a4 = A + 4 * lda;
    const float* b_ptr = B;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        // 4x unrolled K loop — each iteration processes 4 K values
        #define K_ITER_5x16() do { \
            float32x4_t b0 = vld1q_f32(b_ptr); \
            float32x4_t b1 = vld1q_f32(b_ptr + 4); \
            float32x4_t b2 = vld1q_f32(b_ptr + 8); \
            float32x4_t b3 = vld1q_f32(b_ptr + 12); \
            b_ptr += ldb; \
            float32x4_t ar = {a0[0], a1[0], a2[0], a3[0]}; \
            float a4v = a4[0]; \
            a0++; a1++; a2++; a3++; a4++; \
            c00 = vfmaq_laneq_f32(c00, b0, ar, 0); \
            c01 = vfmaq_laneq_f32(c01, b1, ar, 0); \
            c02 = vfmaq_laneq_f32(c02, b2, ar, 0); \
            c03 = vfmaq_laneq_f32(c03, b3, ar, 0); \
            c10 = vfmaq_laneq_f32(c10, b0, ar, 1); \
            c11 = vfmaq_laneq_f32(c11, b1, ar, 1); \
            c12 = vfmaq_laneq_f32(c12, b2, ar, 1); \
            c13 = vfmaq_laneq_f32(c13, b3, ar, 1); \
            c20 = vfmaq_laneq_f32(c20, b0, ar, 2); \
            c21 = vfmaq_laneq_f32(c21, b1, ar, 2); \
            c22 = vfmaq_laneq_f32(c22, b2, ar, 2); \
            c23 = vfmaq_laneq_f32(c23, b3, ar, 2); \
            c30 = vfmaq_laneq_f32(c30, b0, ar, 3); \
            c31 = vfmaq_laneq_f32(c31, b1, ar, 3); \
            c32 = vfmaq_laneq_f32(c32, b2, ar, 3); \
            c33 = vfmaq_laneq_f32(c33, b3, ar, 3); \
            c40 = vfmaq_n_f32(c40, b0, a4v); \
            c41 = vfmaq_n_f32(c41, b1, a4v); \
            c42 = vfmaq_n_f32(c42, b2, a4v); \
            c43 = vfmaq_n_f32(c43, b3, a4v); \
        } while(0)

        K_ITER_5x16();
        K_ITER_5x16();
        K_ITER_5x16();
        K_ITER_5x16();
        #undef K_ITER_5x16
    }

    for (; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);
        float32x4_t b2 = vld1q_f32(b_ptr + 8);
        float32x4_t b3 = vld1q_f32(b_ptr + 12);
        b_ptr += ldb;

        float32x4_t ar = {a0[0], a1[0], a2[0], a3[0]};
        float a4v = a4[0];
        a0++; a1++; a2++; a3++; a4++;

        c00 = vfmaq_laneq_f32(c00, b0, ar, 0);
        c01 = vfmaq_laneq_f32(c01, b1, ar, 0);
        c02 = vfmaq_laneq_f32(c02, b2, ar, 0);
        c03 = vfmaq_laneq_f32(c03, b3, ar, 0);
        c10 = vfmaq_laneq_f32(c10, b0, ar, 1);
        c11 = vfmaq_laneq_f32(c11, b1, ar, 1);
        c12 = vfmaq_laneq_f32(c12, b2, ar, 1);
        c13 = vfmaq_laneq_f32(c13, b3, ar, 1);
        c20 = vfmaq_laneq_f32(c20, b0, ar, 2);
        c21 = vfmaq_laneq_f32(c21, b1, ar, 2);
        c22 = vfmaq_laneq_f32(c22, b2, ar, 2);
        c23 = vfmaq_laneq_f32(c23, b3, ar, 2);
        c30 = vfmaq_laneq_f32(c30, b0, ar, 3);
        c31 = vfmaq_laneq_f32(c31, b1, ar, 3);
        c32 = vfmaq_laneq_f32(c32, b2, ar, 3);
        c33 = vfmaq_laneq_f32(c33, b3, ar, 3);
        c40 = vfmaq_n_f32(c40, b0, a4v);
        c41 = vfmaq_n_f32(c41, b1, a4v);
        c42 = vfmaq_n_f32(c42, b2, a4v);
        c43 = vfmaq_n_f32(c43, b3, a4v);
    }

    float32x4_t av = vdupq_n_f32(alpha);
    auto store_row = [&](float* cr, float32x4_t r0, float32x4_t r1,
                          float32x4_t r2, float32x4_t r3) {
        if (beta == 0.0f) {
            vst1q_f32(cr,      vmulq_f32(av, r0));
            vst1q_f32(cr + 4,  vmulq_f32(av, r1));
            vst1q_f32(cr + 8,  vmulq_f32(av, r2));
            vst1q_f32(cr + 12, vmulq_f32(av, r3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(cr,      vfmaq_f32(vmulq_f32(av, r0), bv, vld1q_f32(cr)));
            vst1q_f32(cr + 4,  vfmaq_f32(vmulq_f32(av, r1), bv, vld1q_f32(cr + 4)));
            vst1q_f32(cr + 8,  vfmaq_f32(vmulq_f32(av, r2), bv, vld1q_f32(cr + 8)));
            vst1q_f32(cr + 12, vfmaq_f32(vmulq_f32(av, r3), bv, vld1q_f32(cr + 12)));
        }
    };

    store_row(C,             c00, c01, c02, c03);
    store_row(C + ldc,       c10, c11, c12, c13);
    store_row(C + 2 * ldc,   c20, c21, c22, c23);
    store_row(C + 3 * ldc,   c30, c31, c32, c33);
    store_row(C + 4 * ldc,   c40, c41, c42, c43);
}

// ============================================================
// 7×16 Clang .s[N] kernel — 28 FMLAs per K
// 2-pass approach: rows 0-3 (16 acc) + rows 4-6 (12 acc)
// Register budget per pass: 16 acc + 4 B + 1 A = 21 (pass 1)
//                           12 acc + 4 B + 1 A = 17 (pass 2)
// ============================================================

void gemm_kernel_7x16_lane(int K,
                             const float* __restrict__ A, int lda,
                             const float* __restrict__ B, int ldb,
                             float* __restrict__ C, int ldc,
                             float alpha, float beta) {
    // Pass 1: rows 0-3 (16 acc)
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);

    // Pass 2: rows 4-6 (12 acc)
    float32x4_t d40 = vdupq_n_f32(0), d41 = vdupq_n_f32(0);
    float32x4_t d42 = vdupq_n_f32(0), d43 = vdupq_n_f32(0);
    float32x4_t d50 = vdupq_n_f32(0), d51 = vdupq_n_f32(0);
    float32x4_t d52 = vdupq_n_f32(0), d53 = vdupq_n_f32(0);
    float32x4_t d60 = vdupq_n_f32(0), d61 = vdupq_n_f32(0);
    float32x4_t d62 = vdupq_n_f32(0), d63 = vdupq_n_f32(0);

    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;
    const float* a4 = A + 4 * lda;
    const float* a5 = A + 5 * lda;
    const float* a6 = A + 6 * lda;

    for (int k = 0; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(&B[k * ldb]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);
        float32x4_t b2 = vld1q_f32(&B[k * ldb + 8]);
        float32x4_t b3 = vld1q_f32(&B[k * ldb + 12]);

        // Pack rows 0-3 into a vector, rows 4-6 as scalars
        float32x4_t ar0123 = {a0[k], a1[k], a2[k], a3[k]};

        // Rows 0-3: .s[0..3]
        c00 = vfmaq_laneq_f32(c00, b0, ar0123, 0);
        c01 = vfmaq_laneq_f32(c01, b1, ar0123, 0);
        c02 = vfmaq_laneq_f32(c02, b2, ar0123, 0);
        c03 = vfmaq_laneq_f32(c03, b3, ar0123, 0);
        c10 = vfmaq_laneq_f32(c10, b0, ar0123, 1);
        c11 = vfmaq_laneq_f32(c11, b1, ar0123, 1);
        c12 = vfmaq_laneq_f32(c12, b2, ar0123, 1);
        c13 = vfmaq_laneq_f32(c13, b3, ar0123, 1);
        c20 = vfmaq_laneq_f32(c20, b0, ar0123, 2);
        c21 = vfmaq_laneq_f32(c21, b1, ar0123, 2);
        c22 = vfmaq_laneq_f32(c22, b2, ar0123, 2);
        c23 = vfmaq_laneq_f32(c23, b3, ar0123, 2);
        c30 = vfmaq_laneq_f32(c30, b0, ar0123, 3);
        c31 = vfmaq_laneq_f32(c31, b1, ar0123, 3);
        c32 = vfmaq_laneq_f32(c32, b2, ar0123, 3);
        c33 = vfmaq_laneq_f32(c33, b3, ar0123, 3);

        // Rows 4-6: scalar broadcast
        float a4v = a4[k], a5v = a5[k], a6v = a6[k];
        d40 = vfmaq_n_f32(d40, b0, a4v);
        d41 = vfmaq_n_f32(d41, b1, a4v);
        d42 = vfmaq_n_f32(d42, b2, a4v);
        d43 = vfmaq_n_f32(d43, b3, a4v);
        d50 = vfmaq_n_f32(d50, b0, a5v);
        d51 = vfmaq_n_f32(d51, b1, a5v);
        d52 = vfmaq_n_f32(d52, b2, a5v);
        d53 = vfmaq_n_f32(d53, b3, a5v);
        d60 = vfmaq_n_f32(d60, b0, a6v);
        d61 = vfmaq_n_f32(d61, b1, a6v);
        d62 = vfmaq_n_f32(d62, b2, a6v);
        d63 = vfmaq_n_f32(d63, b3, a6v);
    }

    float32x4_t av = vdupq_n_f32(alpha);
    auto store_row = [&](float* cr, float32x4_t r0, float32x4_t r1,
                          float32x4_t r2, float32x4_t r3) {
        if (beta == 0.0f) {
            vst1q_f32(cr,      vmulq_f32(av, r0));
            vst1q_f32(cr + 4,  vmulq_f32(av, r1));
            vst1q_f32(cr + 8,  vmulq_f32(av, r2));
            vst1q_f32(cr + 12, vmulq_f32(av, r3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(cr,      vfmaq_f32(vmulq_f32(av, r0), bv, vld1q_f32(cr)));
            vst1q_f32(cr + 4,  vfmaq_f32(vmulq_f32(av, r1), bv, vld1q_f32(cr + 4)));
            vst1q_f32(cr + 8,  vfmaq_f32(vmulq_f32(av, r2), bv, vld1q_f32(cr + 8)));
            vst1q_f32(cr + 12, vfmaq_f32(vmulq_f32(av, r3), bv, vld1q_f32(cr + 12)));
        }
    };

    store_row(C,             c00, c01, c02, c03);
    store_row(C + ldc,       c10, c11, c12, c13);
    store_row(C + 2 * ldc,   c20, c21, c22, c23);
    store_row(C + 3 * ldc,   c30, c31, c32, c33);
    store_row(C + 4 * ldc,   d40, d41, d42, d43);
    store_row(C + 5 * ldc,   d50, d51, d52, d53);
    store_row(C + 6 * ldc,   d60, d61, d62, d63);
}

}  // namespace dnnopt

#endif  // __aarch64__
