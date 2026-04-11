/// @file gemm_ukernel_fp32_8x16.cpp
/// Packed 8x16 FP32 NEON intrinsics micro-kernel for BLIS-style GEMM registry.
///
/// Register budget: 8x16 = 32 accumulators → fills all 32 NEON SIMD registers.
/// Strategy: 2-pass approach within the K loop:
///   - Load 4 B quads (v0-v3) once per K iteration
///   - Process rows 0-3 (acc in stack-backed array), then rows 4-7
///   - Each pass uses 16 acc + 4 B = 20 regs, no spill
///
/// Packed memory layout (same as 4x16):
///   packed_A: for each k, 8 contiguous floats (Mr=8 rows) → scalar loads
///   packed_B: for each k, 16 contiguous floats (Nr=16 cols) → 4 × ldr q
///
/// GCC-compatible: uses vfmaq_n_f32 (scalar broadcast + FMLA) instead of
/// the Clang-only .s[N] subscript syntax on float32x4_t.
///
/// Performance: packed sequential access eliminates stride calculations,
/// prefetch-friendly, and B is loaded once per K step for all 8 rows.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __aarch64__

namespace dnnopt {

// ============================================================
// Pack functions for Mr=8, Nr=16
// ============================================================

static void pack_a_fp32_8x16(int m_len, int k_len,
                              const float* A, int lda,
                              float* packed_A) {
    constexpr int Mr = 8;
    for (int i = 0; i < m_len; i += Mr) {
        int m_rem = std::min(Mr, m_len - i);
        if (m_rem == Mr) {
            for (int k = 0; k < k_len; ++k) {
                // Pack 8 rows of A[k] contiguously
                packed_A[0] = A[(i + 0) * lda + k];
                packed_A[1] = A[(i + 1) * lda + k];
                packed_A[2] = A[(i + 2) * lda + k];
                packed_A[3] = A[(i + 3) * lda + k];
                packed_A[4] = A[(i + 4) * lda + k];
                packed_A[5] = A[(i + 5) * lda + k];
                packed_A[6] = A[(i + 6) * lda + k];
                packed_A[7] = A[(i + 7) * lda + k];
                packed_A += 8;
            }
        } else {
            for (int k = 0; k < k_len; ++k) {
                int r = 0;
                for (; r < m_rem; ++r)
                    packed_A[r] = A[(i + r) * lda + k];
                for (; r < Mr; ++r)
                    packed_A[r] = 0.0f;
                packed_A += 8;
            }
        }
    }
}

// Pack B is identical to 4x16 version (Nr=16)
static void pack_b_fp32_8x16(int k_len, int n_len,
                              const float* B, int ldb,
                              float* packed_B) {
    constexpr int Nr = 16;
    for (int j = 0; j < n_len; j += Nr) {
        int n_rem = std::min(Nr, n_len - j);
        if (n_rem == Nr) {
            for (int k = 0; k < k_len; ++k) {
                const float* src = &B[k * ldb + j];
                vst1q_f32(packed_B,      vld1q_f32(src));
                vst1q_f32(packed_B + 4,  vld1q_f32(src + 4));
                vst1q_f32(packed_B + 8,  vld1q_f32(src + 8));
                vst1q_f32(packed_B + 12, vld1q_f32(src + 12));
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
// Packed 8x16 NEON intrinsics micro-kernel (GCC-compatible)
// ============================================================
//
// Strategy: 2-pass within K loop to fit in 32 SIMD registers.
//   Pass 1: rows 0-3, 16 acc + 4 B = 20 regs
//   Pass 2: rows 4-7, 16 acc + 4 B = 20 regs
// B is loaded once and shared across both passes.
//
// A values loaded as scalars, applied via vfmaq_n_f32 (scalar broadcast).
// With 4x K-unrolling and PRFM prefetch for packed sequential access.

static void gemm_ukernel_fp32_packed_8x16(
        int K,
        const float* __restrict__ packed_A,
        const float* __restrict__ packed_B,
        float* __restrict__ C, int ldc,
        float alpha, float beta) {

    // 16 accumulators for rows 0-3 (pass 1)
    float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0);
    float32x4_t a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
    float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0);
    float32x4_t a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
    float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0);
    float32x4_t a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
    float32x4_t a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0);
    float32x4_t a32 = vdupq_n_f32(0), a33 = vdupq_n_f32(0);

    // 16 accumulators for rows 4-7 (pass 2)
    float32x4_t b00 = vdupq_n_f32(0), b01 = vdupq_n_f32(0);
    float32x4_t b02 = vdupq_n_f32(0), b03 = vdupq_n_f32(0);
    float32x4_t b10 = vdupq_n_f32(0), b11 = vdupq_n_f32(0);
    float32x4_t b12 = vdupq_n_f32(0), b13 = vdupq_n_f32(0);
    float32x4_t b20 = vdupq_n_f32(0), b21 = vdupq_n_f32(0);
    float32x4_t b22 = vdupq_n_f32(0), b23 = vdupq_n_f32(0);
    float32x4_t b30 = vdupq_n_f32(0), b31 = vdupq_n_f32(0);
    float32x4_t b32 = vdupq_n_f32(0), b33 = vdupq_n_f32(0);

    const float* __restrict__ pA = packed_A;
    const float* __restrict__ pB = packed_B;

    // 4x K-unrolled main loop
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Prefetch A and B ahead (packed layout: sequential access)
        if (k + 12 < K) {
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(pA + 32) : "memory");
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(pB + 64) : "memory");
        }

        // --- K iteration 0 ---
        {
            // Load 8 A scalars (one per row)
            float a0v = pA[0], a1v = pA[1], a2v = pA[2], a3v = pA[3];
            float a4v = pA[4], a5v = pA[5], a6v = pA[6], a7v = pA[7];
            pA += 8;

            // Load B panel (4 quads = 16 cols)
            float32x4_t b0 = vld1q_f32(pB);
            float32x4_t b1 = vld1q_f32(pB + 4);
            float32x4_t b2 = vld1q_f32(pB + 8);
            float32x4_t b3 = vld1q_f32(pB + 12);
            pB += 16;

            // Rows 0-3: vfmaq_n_f32 broadcasts scalar and does FMLA
            a00 = vfmaq_n_f32(a00, b0, a0v); a01 = vfmaq_n_f32(a01, b1, a0v);
            a02 = vfmaq_n_f32(a02, b2, a0v); a03 = vfmaq_n_f32(a03, b3, a0v);
            a10 = vfmaq_n_f32(a10, b0, a1v); a11 = vfmaq_n_f32(a11, b1, a1v);
            a12 = vfmaq_n_f32(a12, b2, a1v); a13 = vfmaq_n_f32(a13, b3, a1v);
            a20 = vfmaq_n_f32(a20, b0, a2v); a21 = vfmaq_n_f32(a21, b1, a2v);
            a22 = vfmaq_n_f32(a22, b2, a2v); a23 = vfmaq_n_f32(a23, b3, a2v);
            a30 = vfmaq_n_f32(a30, b0, a3v); a31 = vfmaq_n_f32(a31, b1, a3v);
            a32 = vfmaq_n_f32(a32, b2, a3v); a33 = vfmaq_n_f32(a33, b3, a3v);

            // Rows 4-7
            b00 = vfmaq_n_f32(b00, b0, a4v); b01 = vfmaq_n_f32(b01, b1, a4v);
            b02 = vfmaq_n_f32(b02, b2, a4v); b03 = vfmaq_n_f32(b03, b3, a4v);
            b10 = vfmaq_n_f32(b10, b0, a5v); b11 = vfmaq_n_f32(b11, b1, a5v);
            b12 = vfmaq_n_f32(b12, b2, a5v); b13 = vfmaq_n_f32(b13, b3, a5v);
            b20 = vfmaq_n_f32(b20, b0, a6v); b21 = vfmaq_n_f32(b21, b1, a6v);
            b22 = vfmaq_n_f32(b22, b2, a6v); b23 = vfmaq_n_f32(b23, b3, a6v);
            b30 = vfmaq_n_f32(b30, b0, a7v); b31 = vfmaq_n_f32(b31, b1, a7v);
            b32 = vfmaq_n_f32(b32, b2, a7v); b33 = vfmaq_n_f32(b33, b3, a7v);
        }

        // --- K iteration 1 ---
        {
            float a0v = pA[0], a1v = pA[1], a2v = pA[2], a3v = pA[3];
            float a4v = pA[4], a5v = pA[5], a6v = pA[6], a7v = pA[7];
            pA += 8;

            float32x4_t b0 = vld1q_f32(pB);
            float32x4_t b1 = vld1q_f32(pB + 4);
            float32x4_t b2 = vld1q_f32(pB + 8);
            float32x4_t b3 = vld1q_f32(pB + 12);
            pB += 16;

            a00 = vfmaq_n_f32(a00, b0, a0v); a01 = vfmaq_n_f32(a01, b1, a0v);
            a02 = vfmaq_n_f32(a02, b2, a0v); a03 = vfmaq_n_f32(a03, b3, a0v);
            a10 = vfmaq_n_f32(a10, b0, a1v); a11 = vfmaq_n_f32(a11, b1, a1v);
            a12 = vfmaq_n_f32(a12, b2, a1v); a13 = vfmaq_n_f32(a13, b3, a1v);
            a20 = vfmaq_n_f32(a20, b0, a2v); a21 = vfmaq_n_f32(a21, b1, a2v);
            a22 = vfmaq_n_f32(a22, b2, a2v); a23 = vfmaq_n_f32(a23, b3, a2v);
            a30 = vfmaq_n_f32(a30, b0, a3v); a31 = vfmaq_n_f32(a31, b1, a3v);
            a32 = vfmaq_n_f32(a32, b2, a3v); a33 = vfmaq_n_f32(a33, b3, a3v);

            b00 = vfmaq_n_f32(b00, b0, a4v); b01 = vfmaq_n_f32(b01, b1, a4v);
            b02 = vfmaq_n_f32(b02, b2, a4v); b03 = vfmaq_n_f32(b03, b3, a4v);
            b10 = vfmaq_n_f32(b10, b0, a5v); b11 = vfmaq_n_f32(b11, b1, a5v);
            b12 = vfmaq_n_f32(b12, b2, a5v); b13 = vfmaq_n_f32(b13, b3, a5v);
            b20 = vfmaq_n_f32(b20, b0, a6v); b21 = vfmaq_n_f32(b21, b1, a6v);
            b22 = vfmaq_n_f32(b22, b2, a6v); b23 = vfmaq_n_f32(b23, b3, a6v);
            b30 = vfmaq_n_f32(b30, b0, a7v); b31 = vfmaq_n_f32(b31, b1, a7v);
            b32 = vfmaq_n_f32(b32, b2, a7v); b33 = vfmaq_n_f32(b33, b3, a7v);
        }

        // --- K iteration 2 ---
        {
            float a0v = pA[0], a1v = pA[1], a2v = pA[2], a3v = pA[3];
            float a4v = pA[4], a5v = pA[5], a6v = pA[6], a7v = pA[7];
            pA += 8;

            float32x4_t b0 = vld1q_f32(pB);
            float32x4_t b1 = vld1q_f32(pB + 4);
            float32x4_t b2 = vld1q_f32(pB + 8);
            float32x4_t b3 = vld1q_f32(pB + 12);
            pB += 16;

            a00 = vfmaq_n_f32(a00, b0, a0v); a01 = vfmaq_n_f32(a01, b1, a0v);
            a02 = vfmaq_n_f32(a02, b2, a0v); a03 = vfmaq_n_f32(a03, b3, a0v);
            a10 = vfmaq_n_f32(a10, b0, a1v); a11 = vfmaq_n_f32(a11, b1, a1v);
            a12 = vfmaq_n_f32(a12, b2, a1v); a13 = vfmaq_n_f32(a13, b3, a1v);
            a20 = vfmaq_n_f32(a20, b0, a2v); a21 = vfmaq_n_f32(a21, b1, a2v);
            a22 = vfmaq_n_f32(a22, b2, a2v); a23 = vfmaq_n_f32(a23, b3, a2v);
            a30 = vfmaq_n_f32(a30, b0, a3v); a31 = vfmaq_n_f32(a31, b1, a3v);
            a32 = vfmaq_n_f32(a32, b2, a3v); a33 = vfmaq_n_f32(a33, b3, a3v);

            b00 = vfmaq_n_f32(b00, b0, a4v); b01 = vfmaq_n_f32(b01, b1, a4v);
            b02 = vfmaq_n_f32(b02, b2, a4v); b03 = vfmaq_n_f32(b03, b3, a4v);
            b10 = vfmaq_n_f32(b10, b0, a5v); b11 = vfmaq_n_f32(b11, b1, a5v);
            b12 = vfmaq_n_f32(b12, b2, a5v); b13 = vfmaq_n_f32(b13, b3, a5v);
            b20 = vfmaq_n_f32(b20, b0, a6v); b21 = vfmaq_n_f32(b21, b1, a6v);
            b22 = vfmaq_n_f32(b22, b2, a6v); b23 = vfmaq_n_f32(b23, b3, a6v);
            b30 = vfmaq_n_f32(b30, b0, a7v); b31 = vfmaq_n_f32(b31, b1, a7v);
            b32 = vfmaq_n_f32(b32, b2, a7v); b33 = vfmaq_n_f32(b33, b3, a7v);
        }

        // --- K iteration 3 ---
        {
            float a0v = pA[0], a1v = pA[1], a2v = pA[2], a3v = pA[3];
            float a4v = pA[4], a5v = pA[5], a6v = pA[6], a7v = pA[7];
            pA += 8;

            float32x4_t b0 = vld1q_f32(pB);
            float32x4_t b1 = vld1q_f32(pB + 4);
            float32x4_t b2 = vld1q_f32(pB + 8);
            float32x4_t b3 = vld1q_f32(pB + 12);
            pB += 16;

            a00 = vfmaq_n_f32(a00, b0, a0v); a01 = vfmaq_n_f32(a01, b1, a0v);
            a02 = vfmaq_n_f32(a02, b2, a0v); a03 = vfmaq_n_f32(a03, b3, a0v);
            a10 = vfmaq_n_f32(a10, b0, a1v); a11 = vfmaq_n_f32(a11, b1, a1v);
            a12 = vfmaq_n_f32(a12, b2, a1v); a13 = vfmaq_n_f32(a13, b3, a1v);
            a20 = vfmaq_n_f32(a20, b0, a2v); a21 = vfmaq_n_f32(a21, b1, a2v);
            a22 = vfmaq_n_f32(a22, b2, a2v); a23 = vfmaq_n_f32(a23, b3, a2v);
            a30 = vfmaq_n_f32(a30, b0, a3v); a31 = vfmaq_n_f32(a31, b1, a3v);
            a32 = vfmaq_n_f32(a32, b2, a3v); a33 = vfmaq_n_f32(a33, b3, a3v);

            b00 = vfmaq_n_f32(b00, b0, a4v); b01 = vfmaq_n_f32(b01, b1, a4v);
            b02 = vfmaq_n_f32(b02, b2, a4v); b03 = vfmaq_n_f32(b03, b3, a4v);
            b10 = vfmaq_n_f32(b10, b0, a5v); b11 = vfmaq_n_f32(b11, b1, a5v);
            b12 = vfmaq_n_f32(b12, b2, a5v); b13 = vfmaq_n_f32(b13, b3, a5v);
            b20 = vfmaq_n_f32(b20, b0, a6v); b21 = vfmaq_n_f32(b21, b1, a6v);
            b22 = vfmaq_n_f32(b22, b2, a6v); b23 = vfmaq_n_f32(b23, b3, a6v);
            b30 = vfmaq_n_f32(b30, b0, a7v); b31 = vfmaq_n_f32(b31, b1, a7v);
            b32 = vfmaq_n_f32(b32, b2, a7v); b33 = vfmaq_n_f32(b33, b3, a7v);
        }
    }

    // K tail: 1 at a time
    for (; k < K; ++k) {
        float a0v = pA[0], a1v = pA[1], a2v = pA[2], a3v = pA[3];
        float a4v = pA[4], a5v = pA[5], a6v = pA[6], a7v = pA[7];
        pA += 8;

        float32x4_t b0 = vld1q_f32(pB);
        float32x4_t b1 = vld1q_f32(pB + 4);
        float32x4_t b2 = vld1q_f32(pB + 8);
        float32x4_t b3 = vld1q_f32(pB + 12);
        pB += 16;

        // Rows 0-3
        a00 = vfmaq_n_f32(a00, b0, a0v); a01 = vfmaq_n_f32(a01, b1, a0v);
        a02 = vfmaq_n_f32(a02, b2, a0v); a03 = vfmaq_n_f32(a03, b3, a0v);
        a10 = vfmaq_n_f32(a10, b0, a1v); a11 = vfmaq_n_f32(a11, b1, a1v);
        a12 = vfmaq_n_f32(a12, b2, a1v); a13 = vfmaq_n_f32(a13, b3, a1v);
        a20 = vfmaq_n_f32(a20, b0, a2v); a21 = vfmaq_n_f32(a21, b1, a2v);
        a22 = vfmaq_n_f32(a22, b2, a2v); a23 = vfmaq_n_f32(a23, b3, a2v);
        a30 = vfmaq_n_f32(a30, b0, a3v); a31 = vfmaq_n_f32(a31, b1, a3v);
        a32 = vfmaq_n_f32(a32, b2, a3v); a33 = vfmaq_n_f32(a33, b3, a3v);

        // Rows 4-7
        b00 = vfmaq_n_f32(b00, b0, a4v); b01 = vfmaq_n_f32(b01, b1, a4v);
        b02 = vfmaq_n_f32(b02, b2, a4v); b03 = vfmaq_n_f32(b03, b3, a4v);
        b10 = vfmaq_n_f32(b10, b0, a5v); b11 = vfmaq_n_f32(b11, b1, a5v);
        b12 = vfmaq_n_f32(b12, b2, a5v); b13 = vfmaq_n_f32(b13, b3, a5v);
        b20 = vfmaq_n_f32(b20, b0, a6v); b21 = vfmaq_n_f32(b21, b1, a6v);
        b22 = vfmaq_n_f32(b22, b2, a6v); b23 = vfmaq_n_f32(b23, b3, a6v);
        b30 = vfmaq_n_f32(b30, b0, a7v); b31 = vfmaq_n_f32(b31, b1, a7v);
        b32 = vfmaq_n_f32(b32, b2, a7v); b33 = vfmaq_n_f32(b33, b3, a7v);
    }

    // ================================================================
    // Epilogue: store C = alpha * acc + beta * C
    // ================================================================
    float32x4_t av = vdupq_n_f32(alpha);

    auto store_row = [&](float* cr, float32x4_t c0, float32x4_t c1,
                          float32x4_t c2, float32x4_t c3) {
        if (beta == 0.0f) {
            vst1q_f32(cr,      vmulq_f32(av, c0));
            vst1q_f32(cr + 4,  vmulq_f32(av, c1));
            vst1q_f32(cr + 8,  vmulq_f32(av, c2));
            vst1q_f32(cr + 12, vmulq_f32(av, c3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(cr,      vfmaq_f32(vmulq_f32(av, c0), bv, vld1q_f32(cr)));
            vst1q_f32(cr + 4,  vfmaq_f32(vmulq_f32(av, c1), bv, vld1q_f32(cr + 4)));
            vst1q_f32(cr + 8,  vfmaq_f32(vmulq_f32(av, c2), bv, vld1q_f32(cr + 8)));
            vst1q_f32(cr + 12, vfmaq_f32(vmulq_f32(av, c3), bv, vld1q_f32(cr + 12)));
        }
    };

    float* c0 = C;
    float* c1 = C + ldc;
    float* c2 = C + 2 * ldc;
    float* c3 = C + 3 * ldc;
    float* c4 = C + 4 * ldc;
    float* c5 = C + 5 * ldc;
    float* c6 = C + 6 * ldc;
    float* c7 = C + 7 * ldc;

    store_row(c0, a00, a01, a02, a03);
    store_row(c1, a10, a11, a12, a13);
    store_row(c2, a20, a21, a22, a23);
    store_row(c3, a30, a31, a32, a33);
    store_row(c4, b00, b01, b02, b03);
    store_row(c5, b10, b11, b12, b13);
    store_row(c6, b20, b21, b22, b23);
    store_row(c7, b30, b31, b32, b33);
}

// ============================================================
// Registry wrappers + auto-registration
// ============================================================

namespace {

void ukernel_fp32_8x16_wrap(int K, const void* packed_A, const void* packed_B,
                             float* C, int ldc, float alpha, float beta,
                             float /*extra*/) {
    gemm_ukernel_fp32_packed_8x16(
        K,
        static_cast<const float*>(packed_A),
        static_cast<const float*>(packed_B),
        C, ldc, alpha, beta);
}

void pack_a_fp32_8x16_wrap(int m_len, int k_len, const float* A, int lda,
                            void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32_8x16(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

void pack_b_fp32_8x16_wrap(int k_len, int n_len, const float* B, int ldb,
                            void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32_8x16(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

// Priority 120: higher than 4x16 (110) and 8x12 (100).
// 8x16 processes more FLOPs per K step (128 FMLAs vs 64 for 4x16)
// and has zero N-tail for N%16==0 shapes (256, 512, 1024, etc.).
const GemmMicrokernelDesc neon_fp32_8x16_desc = {
    "neon_fp32_8x16",
    GemmDataType::kFP32,
    kNEON,                // required_hwcaps
    8,                    // Mr = 8
    16,                   // Nr = 16
    1,                    // Kgroup
    false,                // nr_is_vla
    120,                  // priority (higher than 8x12's 100, same as 4x16's 110 — prefer 8 rows)
    sizeof(float),        // packed_a_elem_bytes
    sizeof(float),        // packed_b_elem_bytes
    0,                    // min_sve_bits
    ukernel_fp32_8x16_wrap,
    pack_a_fp32_8x16_wrap,
    pack_b_fp32_8x16_wrap,
};

static RegisterKernel reg_neon_fp32_8x16(neon_fp32_8x16_desc);

}  // namespace

}  // namespace dnnopt

#endif  // __aarch64__
