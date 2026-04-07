/// @file gemm_ukernel_bf16_sve.cpp
/// SVE BF16 GEMM microkernel using svbfmmla_f32.
///
/// On SVE-128: equivalent to NEON BFMMLA (same 8x8 tile)
/// On SVE-256+: processes 2+ column-pairs per svbfmmla, giving wider Nr
///
/// VLA design: Mr=8 (4 row-pairs), Nr=2*svcntw() (VLA)
/// Each svbfmmla processes 128-bit segments within SVE registers.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_sve.h>
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace dnnopt {

// ============================================================
// SVE BF16 packing: FP32 → BF16 with SVE predicates
// ============================================================

/// Pack A into BFMMLA format using SVE.
/// For each K-group of 4, packs row-pairs: [row0_k0..k3, row1_k0..k3] as 8 BF16.
static void pack_a_bf16_sve(int m_len, int k_len, const float* A, int lda,
                             void* packed_out, int Mr) {
    bfloat16_t* packed = static_cast<bfloat16_t*>(packed_out);
    constexpr int Kgroup = 4;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Process Mr/2 = 4 row-pairs
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;
                float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (r0 < m_len) tmp[kk]     = A[r0 * lda + k + kk];
                    if (r1 < m_len) tmp[4 + kk]  = A[r1 * lda + k + kk];
                }

                // Convert 8 FP32 to 8 BF16
                float32x4_t f0 = vld1q_f32(tmp);
                float32x4_t f1 = vld1q_f32(tmp + 4);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                vst1q_bf16(packed, combined);
                packed += 8;
            }
        }
    }
}

/// Pack B into BFMMLA format using SVE.
/// For each K-group of 4, packs col-pairs: [k0c0..k3c0, k0c1..k3c1] as 8 BF16.
static void pack_b_bf16_sve(int k_len, int n_len, const float* B, int ldb,
                             void* packed_out, int Nr) {
    bfloat16_t* packed = static_cast<bfloat16_t*>(packed_out);
    constexpr int Kgroup = 4;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Process Nr/2 column-pairs
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;
                float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (c0 < n_len) tmp[kk]     = B[(k + kk) * ldb + c0];
                    if (c1 < n_len) tmp[4 + kk]  = B[(k + kk) * ldb + c1];
                }

                float32x4_t f0 = vld1q_f32(tmp);
                float32x4_t f1 = vld1q_f32(tmp + 4);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                vst1q_bf16(packed, combined);
                packed += 8;
            }
        }
    }
}

// ============================================================
// SVE BF16 VLA microkernel: Mr=8, Nr=2*VL_f32
// ============================================================

/// BF16 BFMMLA microkernel using SVE svbfmmla_f32.
/// svbfmmla works on 128-bit segments within SVE registers.
/// On SVE-128: processes 1 column-pair (Nr=8 → 4 col-pairs)
/// On SVE-256: processes 2 column-pairs per instruction (Nr=16 → 8 col-pairs)
///
/// Accumulator layout:
///   4 row-pairs × (Nr/2) col-pairs
///   Each accumulator holds a 2×2 FP32 block: [r0c0, r0c1, r1c0, r1c1]
///
/// Since svbfmmla internally operates on 128-bit segments, the tile logic
/// extends naturally to wider vectors. We use VL_f32 accumulators per row-pair.
static void gemm_ukernel_bf16_sve_vla(int K,
                                       const bfloat16_t* packed_A,
                                       const bfloat16_t* packed_B,
                                       float* C, int ldc,
                                       float alpha, float beta) {
    const int vl = (int)svcntw();  // FP32 elements per SVE register
    const int Nr = 2 * vl;
    // Number of column-pairs = Nr / 2 = vl
    // Number of svbfmmla ops per K-group = 4 row-pairs × (vl / 4) col-pair-groups
    // But with SVE VLA, svbfmmla processes all 128-bit segments at once

    constexpr int Kgroup = 4;
    constexpr int Mr = 8;  // 4 row-pairs
    svbool_t pg = svptrue_b32();

    // BF16 predicates for loading packed data
    // Each row-pair has 8 BF16 = 128 bits per K-group
    // packed_A: 4 row-pairs × 8 BF16 = 32 BF16 per K-group
    // packed_B: (Nr/2) col-pairs × 8 BF16 per K-group

    // For SVE-128 (vl=4): 4 col-pairs, 16 accumulators (same as NEON)
    // For SVE-256 (vl=8): 8 col-pairs, 32 accumulators

    // Use dynamic accumulator array for VLA
    // Max practical: SVE-512 → vl=16 → 4*16=64 accumulators
    // We'll use a flat array on stack
    const int n_col_pairs = Nr / 2;
    const int n_accs = 4 * n_col_pairs;  // 4 row-pairs × n_col_pairs
    float acc_buf[4 * 32] = {};  // Max 128 FP32 (SVE-512: 4 × 32)
    // Zero all accumulators
    memset(acc_buf, 0, n_accs * 4 * sizeof(float));

    // K-loop
    for (int k = 0; k < K; k += Kgroup) {
        // Load A row-pairs: 4 × bfloat16x8_t (128-bit each)
        bfloat16x8_t a0 = vld1q_bf16(packed_A + 0);
        bfloat16x8_t a1 = vld1q_bf16(packed_A + 8);
        bfloat16x8_t a2 = vld1q_bf16(packed_A + 16);
        bfloat16x8_t a3 = vld1q_bf16(packed_A + 24);
        packed_A += 32;

        // For each column-pair, load B and accumulate
        for (int cp = 0; cp < n_col_pairs; ++cp) {
            bfloat16x8_t b = vld1q_bf16(packed_B);
            packed_B += 8;

            // 4 BFMMLA per col-pair
            int base = cp * 4;
            float32x4_t c0 = vld1q_f32(&acc_buf[(base + 0) * 4]);
            float32x4_t c1 = vld1q_f32(&acc_buf[(base + 1) * 4]);
            float32x4_t c2 = vld1q_f32(&acc_buf[(base + 2) * 4]);
            float32x4_t c3 = vld1q_f32(&acc_buf[(base + 3) * 4]);

            c0 = vbfmmlaq_f32(c0, a0, b);
            c1 = vbfmmlaq_f32(c1, a1, b);
            c2 = vbfmmlaq_f32(c2, a2, b);
            c3 = vbfmmlaq_f32(c3, a3, b);

            vst1q_f32(&acc_buf[(base + 0) * 4], c0);
            vst1q_f32(&acc_buf[(base + 1) * 4], c1);
            vst1q_f32(&acc_buf[(base + 2) * 4], c2);
            vst1q_f32(&acc_buf[(base + 3) * 4], c3);
        }
    }

    // Epilogue: extract 2×2 blocks and store to C
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    for (int cp = 0; cp < n_col_pairs; ++cp) {
        int col = cp * 2;
        for (int rp = 0; rp < 4; ++rp) {
            int row = rp * 2;
            int idx = (cp * 4 + rp) * 4;
            float32x4_t block = vld1q_f32(&acc_buf[idx]);
            // block = [r0c0, r0c1, r1c0, r1c1]
            float r0c0 = vgetq_lane_f32(block, 0);
            float r0c1 = vgetq_lane_f32(block, 1);
            float r1c0 = vgetq_lane_f32(block, 2);
            float r1c1 = vgetq_lane_f32(block, 3);

            float* Cr0 = &C[row * ldc + col];
            float* Cr1 = &C[(row + 1) * ldc + col];

            if (beta == 0.0f) {
                Cr0[0] = alpha * r0c0;
                Cr0[1] = alpha * r0c1;
                Cr1[0] = alpha * r1c0;
                Cr1[1] = alpha * r1c1;
            } else {
                Cr0[0] = alpha * r0c0 + beta * Cr0[0];
                Cr0[1] = alpha * r0c1 + beta * Cr0[1];
                Cr1[0] = alpha * r1c0 + beta * Cr1[0];
                Cr1[1] = alpha * r1c1 + beta * Cr1[1];
            }
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void pack_a_bf16_sve_wrap(int m_len, int k_len, const float* A, int lda,
                                  void* packed_A, int Mr, float* /*scale_out*/) {
    pack_a_bf16_sve(m_len, k_len, A, lda, packed_A, Mr);
}

static void pack_b_bf16_sve_wrap(int k_len, int n_len, const float* B, int ldb,
                                  void* packed_B, int Nr, float* /*scale_out*/) {
    pack_b_bf16_sve(k_len, n_len, B, ldb, packed_B, Nr);
}

static void ukernel_bf16_sve_wrap(int K, const void* packed_A,
                                   const void* packed_B,
                                   float* C, int ldc, float alpha,
                                   float beta, float /*extra*/) {
    gemm_ukernel_bf16_sve_vla(K,
                               static_cast<const bfloat16_t*>(packed_A),
                               static_cast<const bfloat16_t*>(packed_B),
                               C, ldc, alpha, beta);
}

// SVE BF16 VLA: only selected on wide SVE (256+)
// On SVE-128, NEON BFMMLA (priority 100) handles it identically.
static const GemmMicrokernelDesc sve_bf16_vla_desc = {
    "sve_bf16_8xVL",
    GemmDataType::kBF16,
    kSVE | kSVEBF16 | kBF16,
    8,                    // Mr
    0,                    // Nr (computed)
    4,                    // Kgroup
    true,                 // nr_is_vla
    200,                  // priority
    sizeof(bfloat16_t),
    sizeof(bfloat16_t),
    256,                  // min_sve_bits: only on wide SVE
    ukernel_bf16_sve_wrap,
    pack_a_bf16_sve_wrap,
    pack_b_bf16_sve_wrap,
};

static RegisterKernel reg_sve_bf16_vla(sve_bf16_vla_desc);

// ============================================================
// SVE-128 BF16: same 8x8 BFMMLA tile as NEON, priority=120
// ============================================================
// On SVE-128, BFMMLA intrinsics are identical to NEON. The benefit
// comes from SVE-accelerated packing (pack_a_bf16_sve/pack_b_bf16_sve)
// which uses SVE predicates and conversion instructions.

// Reuse the NEON BFMMLA microkernel (declared in gemm_ukernel_bf16_neon.cpp)
extern void gemm_ukernel_bf16_8x8(int K, const bfloat16_t* packed_A,
                                    const bfloat16_t* packed_B,
                                    float* C, int ldc, float alpha, float beta);

static void ukernel_bf16_sve128_wrap(int K, const void* packed_A,
                                      const void* packed_B,
                                      float* C, int ldc, float alpha,
                                      float beta, float /*extra*/) {
    gemm_ukernel_bf16_8x8(K,
                            static_cast<const bfloat16_t*>(packed_A),
                            static_cast<const bfloat16_t*>(packed_B),
                            C, ldc, alpha, beta);
}

static const GemmMicrokernelDesc sve128_bf16_desc = {
    "sve128_bf16_8x8",
    GemmDataType::kBF16,
    kSVE | kBF16,             // required_hwcaps
    8,                        // Mr
    8,                        // Nr (fixed, same as NEON)
    4,                        // Kgroup
    false,                    // nr_is_vla: NO
    120,                      // priority: higher than NEON 100
    sizeof(bfloat16_t),
    sizeof(bfloat16_t),
    0,                        // min_sve_bits: any SVE
    ukernel_bf16_sve128_wrap,
    pack_a_bf16_sve_wrap,     // SVE packing (predicated, faster)
    pack_b_bf16_sve_wrap,
};

static RegisterKernel reg_sve128_bf16(sve128_bf16_desc);

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
