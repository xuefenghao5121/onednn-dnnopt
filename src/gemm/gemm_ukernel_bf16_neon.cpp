/// @file gemm_ukernel_bf16_neon.cpp
/// 8×8 BF16 BFMMLA microkernel using NEON intrinsics.
///
/// Computes an 8×8 tile of C from packed BF16 A and B:
///   C[8×8] = alpha * packed_A[8×K] * packed_B[K×8] + beta * C[8×8]
///
/// packed_A layout: per K-group of 4, 4 row-pairs × 8 BF16 = 32 BF16
/// packed_B layout: per K-group of 4, 4 col-pairs × 8 BF16 = 32 BF16
/// K must be a multiple of 4 (guaranteed by packing).
///
/// Accumulator layout: 16 regs (v16-v31), each holds a 2×2 FP32 sub-block.
///   v16 = C[0:1, 0:1]  v17 = C[0:1, 2:3]  v18 = C[0:1, 4:5]  v19 = C[0:1, 6:7]
///   v20 = C[2:3, 0:1]  v21 = C[2:3, 2:3]  v22 = C[2:3, 4:5]  v23 = C[2:3, 6:7]
///   v24 = C[4:5, 0:1]  v25 = C[4:5, 2:3]  v26 = C[4:5, 4:5]  v27 = C[4:5, 6:7]
///   v28 = C[6:7, 0:1]  v29 = C[6:7, 2:3]  v30 = C[6:7, 4:5]  v31 = C[6:7, 6:7]
///
/// Each 2×2 block in a register is laid out as [r0c0, r0c1, r1c0, r1c1].

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

void gemm_ukernel_bf16_8x8(int K,
                            const bfloat16_t* packed_A,
                            const bfloat16_t* packed_B,
                            float* C, int ldc,
                            float alpha, float beta) {
    // Cast to __bf16* for GCC NEON intrinsics
    const __bf16* pa = reinterpret_cast<const __bf16*>(packed_A);
    const __bf16* pb = reinterpret_cast<const __bf16*>(packed_B);

    // 16 accumulators for 8×8 tile (each 2×2 FP32)
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);

    // K-loop: process 4 K-elements per iteration
    // Each iteration: 4 A loads (32 BF16) + 4 B loads (32 BF16) + 16 BFMMLA
    int k4 = K / 4;
    for (int ki = 0; ki < k4; ++ki) {
        // Load A row-pairs: v0=rows[0:1], v1=rows[2:3], v2=rows[4:5], v3=rows[6:7]
        bfloat16x8_t a0 = vld1q_bf16(pa);
        bfloat16x8_t a1 = vld1q_bf16(pa + 8);
        bfloat16x8_t a2 = vld1q_bf16(pa + 16);
        bfloat16x8_t a3 = vld1q_bf16(pa + 24);

        // Load B col-pairs: v4=cols[0:1], v5=cols[2:3], v6=cols[4:5], v7=cols[6:7]
        bfloat16x8_t b0 = vld1q_bf16(pb);
        bfloat16x8_t b1 = vld1q_bf16(pb + 8);
        bfloat16x8_t b2 = vld1q_bf16(pb + 16);
        bfloat16x8_t b3 = vld1q_bf16(pb + 24);

        // 16 BFMMLA instructions: rows[i:i+1] × cols[j:j+1]
        // Row pair 0 (rows 0-1)
        c00 = vbfmmlaq_f32(c00, a0, b0);  // C[0:1, 0:1]
        c01 = vbfmmlaq_f32(c01, a0, b1);  // C[0:1, 2:3]
        c02 = vbfmmlaq_f32(c02, a0, b2);  // C[0:1, 4:5]
        c03 = vbfmmlaq_f32(c03, a0, b3);  // C[0:1, 6:7]

        // Row pair 1 (rows 2-3)
        c10 = vbfmmlaq_f32(c10, a1, b0);
        c11 = vbfmmlaq_f32(c11, a1, b1);
        c12 = vbfmmlaq_f32(c12, a1, b2);
        c13 = vbfmmlaq_f32(c13, a1, b3);

        // Row pair 2 (rows 4-5)
        c20 = vbfmmlaq_f32(c20, a2, b0);
        c21 = vbfmmlaq_f32(c21, a2, b1);
        c22 = vbfmmlaq_f32(c22, a2, b2);
        c23 = vbfmmlaq_f32(c23, a2, b3);

        // Row pair 3 (rows 6-7)
        c30 = vbfmmlaq_f32(c30, a3, b0);
        c31 = vbfmmlaq_f32(c31, a3, b1);
        c32 = vbfmmlaq_f32(c32, a3, b2);
        c33 = vbfmmlaq_f32(c33, a3, b3);

        pa += 32;  // 4 row-pairs × 8 BF16
        pb += 32;  // 4 col-pairs × 8 BF16
    }

    // Epilogue: write 8×8 FP32 accumulators to row-major C.
    // Each accumulator holds a 2×2 block: [r0c0, r0c1, r1c0, r1c1]
    // We need to extract rows and assemble 8-wide row vectors.
    //
    // For row pair (r, r+1) with accumulators acc_j (j=0..3):
    //   acc_0 = [C[r,0], C[r,1], C[r+1,0], C[r+1,1]]
    //   acc_1 = [C[r,2], C[r,3], C[r+1,2], C[r+1,3]]
    //   acc_2 = [C[r,4], C[r,5], C[r+1,4], C[r+1,5]]
    //   acc_3 = [C[r,6], C[r,7], C[r+1,6], C[r+1,7]]
    // Row r:   [acc_0[0], acc_0[1], acc_1[0], acc_1[1], acc_2[0], acc_2[1], acc_3[0], acc_3[1]]
    // Row r+1: [acc_0[2], acc_0[3], acc_1[2], acc_1[3], acc_2[2], acc_2[3], acc_3[2], acc_3[3]]
    //
    // Use UZP1/UZP2 (vuzp1q/vuzp2q) to deinterleave even/odd pairs:
    //   uzp1(acc_0, acc_1) = [acc_0[0], acc_0[2], acc_1[0], acc_1[2]] = [r0c0, r1c0, r0c2, r1c2]
    // That's not quite right. Let's use TRN instead, or just extract with vget_lane.
    //
    // Simpler approach: extract via vzip to form row vectors.
    // Actually, the 2×2 layout [r0c0, r0c1, r1c0, r1c1] means:
    //   low 64-bit  = [r0c0, r0c1] = row 0's contribution
    //   high 64-bit = [r1c0, r1c1] = row 1's contribution
    // So we can use vget_low_f32 / vget_high_f32 (treating as 2×float64)
    // and vcombine to form the full row.

    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    // Helper: extract row 0 and row 1 from 4 accumulators, scale, and store
#define STORE_ROW_PAIR(row, a0, a1, a2, a3) do {                       \
    /* Row 'row': low halves of a0,a1,a2,a3 */                         \
    float32x2_t lo0 = vget_low_f32(a0);                               \
    float32x2_t lo1 = vget_low_f32(a1);                               \
    float32x2_t lo2 = vget_low_f32(a2);                               \
    float32x2_t lo3 = vget_low_f32(a3);                               \
    float32x4_t row0_lo = vcombine_f32(lo0, lo1);  /* cols 0-3 */     \
    float32x4_t row0_hi = vcombine_f32(lo2, lo3);  /* cols 4-7 */     \
                                                                       \
    /* Row 'row+1': high halves */                                     \
    float32x2_t hi0 = vget_high_f32(a0);                              \
    float32x2_t hi1 = vget_high_f32(a1);                              \
    float32x2_t hi2 = vget_high_f32(a2);                              \
    float32x2_t hi3 = vget_high_f32(a3);                              \
    float32x4_t row1_lo = vcombine_f32(hi0, hi1);                     \
    float32x4_t row1_hi = vcombine_f32(hi2, hi3);                     \
                                                                       \
    float* Cr0 = C + (row) * ldc;                                     \
    float* Cr1 = C + ((row) + 1) * ldc;                               \
                                                                       \
    if (beta == 0.0f) {                                                \
        vst1q_f32(Cr0,     vmulq_f32(alpha_v, row0_lo));             \
        vst1q_f32(Cr0 + 4, vmulq_f32(alpha_v, row0_hi));             \
        vst1q_f32(Cr1,     vmulq_f32(alpha_v, row1_lo));             \
        vst1q_f32(Cr1 + 4, vmulq_f32(alpha_v, row1_hi));             \
    } else {                                                           \
        vst1q_f32(Cr0,     vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr0)),     alpha_v, row0_lo)); \
        vst1q_f32(Cr0 + 4, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr0 + 4)), alpha_v, row0_hi)); \
        vst1q_f32(Cr1,     vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr1)),     alpha_v, row1_lo)); \
        vst1q_f32(Cr1 + 4, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr1 + 4)), alpha_v, row1_hi)); \
    }                                                                  \
} while(0)

    STORE_ROW_PAIR(0, c00, c01, c02, c03);
    STORE_ROW_PAIR(2, c10, c11, c12, c13);
    STORE_ROW_PAIR(4, c20, c21, c22, c23);
    STORE_ROW_PAIR(6, c30, c31, c32, c33);

#undef STORE_ROW_PAIR
}

#endif  // __ARM_NEON

// ============================================================
// Registry wrappers + auto-registration
// ============================================================

// Original packing functions (defined in gemm_pack_bf16.cpp)
void pack_a_bf16(int m_len, int k_len, const float* A, int lda, bfloat16_t* packed_A);
void pack_b_bf16(int k_len, int n_len, const float* B, int ldb, bfloat16_t* packed_B);

namespace {

void ukernel_bf16_neon_wrap(int K, const void* packed_A, const void* packed_B,
                            float* C, int ldc, float alpha, float beta,
                            float /*extra*/) {
#ifdef __ARM_NEON
    gemm_ukernel_bf16_8x8(K,
                           static_cast<const bfloat16_t*>(packed_A),
                           static_cast<const bfloat16_t*>(packed_B),
                           C, ldc, alpha, beta);
#else
    (void)K; (void)packed_A; (void)packed_B; (void)C; (void)ldc; (void)alpha; (void)beta;
#endif
}

void pack_a_bf16_wrap(int m_len, int k_len, const float* A, int lda,
                      void* packed_A, int /*Mr*/, float* /*scale_out*/) {
#ifdef __ARM_NEON
    pack_a_bf16(m_len, k_len, A, lda, static_cast<bfloat16_t*>(packed_A));
#else
    (void)m_len; (void)k_len; (void)A; (void)lda; (void)packed_A;
#endif
}

void pack_b_bf16_wrap(int k_len, int n_len, const float* B, int ldb,
                      void* packed_B, int /*Nr*/, float* /*scale_out*/) {
#ifdef __ARM_NEON
    pack_b_bf16(k_len, n_len, B, ldb, static_cast<bfloat16_t*>(packed_B));
#else
    (void)k_len; (void)n_len; (void)B; (void)ldb; (void)packed_B;
#endif
}

const GemmMicrokernelDesc neon_bf16_desc = {
    "neon_bf16_8x8",
    GemmDataType::kBF16,
    kNEON | kBF16,        // required_hwcaps
    kGemmMrBf16,          // Mr = 8
    kGemmNrBf16,          // Nr = 8
    4,                    // Kgroup
    false,                // nr_is_vla
    100,                  // priority
    sizeof(bfloat16_t),   // packed_a_elem_bytes
    sizeof(bfloat16_t),   // packed_b_elem_bytes
    0,                    // min_sve_bits
    ukernel_bf16_neon_wrap,
    pack_a_bf16_wrap,
    pack_b_bf16_wrap,
};

static RegisterKernel reg_neon_bf16(neon_bf16_desc);

}  // namespace

}  // namespace dnnopt
