/// @file gemm_ukernel_fp32_neon.cpp
/// 8×12 FP32 NEON microkernel using intrinsics.
///
/// Computes a Mr×Nr (8×12) tile of C from packed A (Mr×K) and packed B (K×Nr):
///   C[8×12] = alpha * packed_A[8×K]^T * packed_B[K×12] + beta * C[8×12]
///
/// packed_A layout: for each k, 8 contiguous floats (one Mr-panel column).
/// packed_B layout: for each k, 12 contiguous floats (one Nr-panel row).

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

void gemm_ukernel_fp32_8x12(int K,
                             const float* packed_A,
                             const float* packed_B,
                             float* C, int ldc,
                             float alpha, float beta) {
    // 24 accumulator registers: 8 rows × 3 groups of 4 columns
    // acc[row][col_group], row in [0,8), col_group in [0,3)
    float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0);
    float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0);
    float32x4_t acc20 = vdupq_n_f32(0), acc21 = vdupq_n_f32(0), acc22 = vdupq_n_f32(0);
    float32x4_t acc30 = vdupq_n_f32(0), acc31 = vdupq_n_f32(0), acc32 = vdupq_n_f32(0);
    float32x4_t acc40 = vdupq_n_f32(0), acc41 = vdupq_n_f32(0), acc42 = vdupq_n_f32(0);
    float32x4_t acc50 = vdupq_n_f32(0), acc51 = vdupq_n_f32(0), acc52 = vdupq_n_f32(0);
    float32x4_t acc60 = vdupq_n_f32(0), acc61 = vdupq_n_f32(0), acc62 = vdupq_n_f32(0);
    float32x4_t acc70 = vdupq_n_f32(0), acc71 = vdupq_n_f32(0), acc72 = vdupq_n_f32(0);

    // Main K-loop, unrolled 4x
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Iteration 0
        {
            float32x4_t a_lo = vld1q_f32(packed_A);       // rows 0-3
            float32x4_t a_hi = vld1q_f32(packed_A + 4);   // rows 4-7
            float32x4_t b0 = vld1q_f32(packed_B);         // cols 0-3
            float32x4_t b1 = vld1q_f32(packed_B + 4);     // cols 4-7
            float32x4_t b2 = vld1q_f32(packed_B + 8);     // cols 8-11

            acc00 = vfmaq_laneq_f32(acc00, b0, a_lo, 0);
            acc01 = vfmaq_laneq_f32(acc01, b1, a_lo, 0);
            acc02 = vfmaq_laneq_f32(acc02, b2, a_lo, 0);
            acc10 = vfmaq_laneq_f32(acc10, b0, a_lo, 1);
            acc11 = vfmaq_laneq_f32(acc11, b1, a_lo, 1);
            acc12 = vfmaq_laneq_f32(acc12, b2, a_lo, 1);
            acc20 = vfmaq_laneq_f32(acc20, b0, a_lo, 2);
            acc21 = vfmaq_laneq_f32(acc21, b1, a_lo, 2);
            acc22 = vfmaq_laneq_f32(acc22, b2, a_lo, 2);
            acc30 = vfmaq_laneq_f32(acc30, b0, a_lo, 3);
            acc31 = vfmaq_laneq_f32(acc31, b1, a_lo, 3);
            acc32 = vfmaq_laneq_f32(acc32, b2, a_lo, 3);

            acc40 = vfmaq_laneq_f32(acc40, b0, a_hi, 0);
            acc41 = vfmaq_laneq_f32(acc41, b1, a_hi, 0);
            acc42 = vfmaq_laneq_f32(acc42, b2, a_hi, 0);
            acc50 = vfmaq_laneq_f32(acc50, b0, a_hi, 1);
            acc51 = vfmaq_laneq_f32(acc51, b1, a_hi, 1);
            acc52 = vfmaq_laneq_f32(acc52, b2, a_hi, 1);
            acc60 = vfmaq_laneq_f32(acc60, b0, a_hi, 2);
            acc61 = vfmaq_laneq_f32(acc61, b1, a_hi, 2);
            acc62 = vfmaq_laneq_f32(acc62, b2, a_hi, 2);
            acc70 = vfmaq_laneq_f32(acc70, b0, a_hi, 3);
            acc71 = vfmaq_laneq_f32(acc71, b1, a_hi, 3);
            acc72 = vfmaq_laneq_f32(acc72, b2, a_hi, 3);

            packed_A += 8;
            packed_B += 12;
        }
        // Iteration 1
        {
            float32x4_t a_lo = vld1q_f32(packed_A);
            float32x4_t a_hi = vld1q_f32(packed_A + 4);
            float32x4_t b0 = vld1q_f32(packed_B);
            float32x4_t b1 = vld1q_f32(packed_B + 4);
            float32x4_t b2 = vld1q_f32(packed_B + 8);

            acc00 = vfmaq_laneq_f32(acc00, b0, a_lo, 0);
            acc01 = vfmaq_laneq_f32(acc01, b1, a_lo, 0);
            acc02 = vfmaq_laneq_f32(acc02, b2, a_lo, 0);
            acc10 = vfmaq_laneq_f32(acc10, b0, a_lo, 1);
            acc11 = vfmaq_laneq_f32(acc11, b1, a_lo, 1);
            acc12 = vfmaq_laneq_f32(acc12, b2, a_lo, 1);
            acc20 = vfmaq_laneq_f32(acc20, b0, a_lo, 2);
            acc21 = vfmaq_laneq_f32(acc21, b1, a_lo, 2);
            acc22 = vfmaq_laneq_f32(acc22, b2, a_lo, 2);
            acc30 = vfmaq_laneq_f32(acc30, b0, a_lo, 3);
            acc31 = vfmaq_laneq_f32(acc31, b1, a_lo, 3);
            acc32 = vfmaq_laneq_f32(acc32, b2, a_lo, 3);

            acc40 = vfmaq_laneq_f32(acc40, b0, a_hi, 0);
            acc41 = vfmaq_laneq_f32(acc41, b1, a_hi, 0);
            acc42 = vfmaq_laneq_f32(acc42, b2, a_hi, 0);
            acc50 = vfmaq_laneq_f32(acc50, b0, a_hi, 1);
            acc51 = vfmaq_laneq_f32(acc51, b1, a_hi, 1);
            acc52 = vfmaq_laneq_f32(acc52, b2, a_hi, 1);
            acc60 = vfmaq_laneq_f32(acc60, b0, a_hi, 2);
            acc61 = vfmaq_laneq_f32(acc61, b1, a_hi, 2);
            acc62 = vfmaq_laneq_f32(acc62, b2, a_hi, 2);
            acc70 = vfmaq_laneq_f32(acc70, b0, a_hi, 3);
            acc71 = vfmaq_laneq_f32(acc71, b1, a_hi, 3);
            acc72 = vfmaq_laneq_f32(acc72, b2, a_hi, 3);

            packed_A += 8;
            packed_B += 12;
        }
        // Iteration 2
        {
            float32x4_t a_lo = vld1q_f32(packed_A);
            float32x4_t a_hi = vld1q_f32(packed_A + 4);
            float32x4_t b0 = vld1q_f32(packed_B);
            float32x4_t b1 = vld1q_f32(packed_B + 4);
            float32x4_t b2 = vld1q_f32(packed_B + 8);

            acc00 = vfmaq_laneq_f32(acc00, b0, a_lo, 0);
            acc01 = vfmaq_laneq_f32(acc01, b1, a_lo, 0);
            acc02 = vfmaq_laneq_f32(acc02, b2, a_lo, 0);
            acc10 = vfmaq_laneq_f32(acc10, b0, a_lo, 1);
            acc11 = vfmaq_laneq_f32(acc11, b1, a_lo, 1);
            acc12 = vfmaq_laneq_f32(acc12, b2, a_lo, 1);
            acc20 = vfmaq_laneq_f32(acc20, b0, a_lo, 2);
            acc21 = vfmaq_laneq_f32(acc21, b1, a_lo, 2);
            acc22 = vfmaq_laneq_f32(acc22, b2, a_lo, 2);
            acc30 = vfmaq_laneq_f32(acc30, b0, a_lo, 3);
            acc31 = vfmaq_laneq_f32(acc31, b1, a_lo, 3);
            acc32 = vfmaq_laneq_f32(acc32, b2, a_lo, 3);

            acc40 = vfmaq_laneq_f32(acc40, b0, a_hi, 0);
            acc41 = vfmaq_laneq_f32(acc41, b1, a_hi, 0);
            acc42 = vfmaq_laneq_f32(acc42, b2, a_hi, 0);
            acc50 = vfmaq_laneq_f32(acc50, b0, a_hi, 1);
            acc51 = vfmaq_laneq_f32(acc51, b1, a_hi, 1);
            acc52 = vfmaq_laneq_f32(acc52, b2, a_hi, 1);
            acc60 = vfmaq_laneq_f32(acc60, b0, a_hi, 2);
            acc61 = vfmaq_laneq_f32(acc61, b1, a_hi, 2);
            acc62 = vfmaq_laneq_f32(acc62, b2, a_hi, 2);
            acc70 = vfmaq_laneq_f32(acc70, b0, a_hi, 3);
            acc71 = vfmaq_laneq_f32(acc71, b1, a_hi, 3);
            acc72 = vfmaq_laneq_f32(acc72, b2, a_hi, 3);

            packed_A += 8;
            packed_B += 12;
        }
        // Iteration 3
        {
            float32x4_t a_lo = vld1q_f32(packed_A);
            float32x4_t a_hi = vld1q_f32(packed_A + 4);
            float32x4_t b0 = vld1q_f32(packed_B);
            float32x4_t b1 = vld1q_f32(packed_B + 4);
            float32x4_t b2 = vld1q_f32(packed_B + 8);

            acc00 = vfmaq_laneq_f32(acc00, b0, a_lo, 0);
            acc01 = vfmaq_laneq_f32(acc01, b1, a_lo, 0);
            acc02 = vfmaq_laneq_f32(acc02, b2, a_lo, 0);
            acc10 = vfmaq_laneq_f32(acc10, b0, a_lo, 1);
            acc11 = vfmaq_laneq_f32(acc11, b1, a_lo, 1);
            acc12 = vfmaq_laneq_f32(acc12, b2, a_lo, 1);
            acc20 = vfmaq_laneq_f32(acc20, b0, a_lo, 2);
            acc21 = vfmaq_laneq_f32(acc21, b1, a_lo, 2);
            acc22 = vfmaq_laneq_f32(acc22, b2, a_lo, 2);
            acc30 = vfmaq_laneq_f32(acc30, b0, a_lo, 3);
            acc31 = vfmaq_laneq_f32(acc31, b1, a_lo, 3);
            acc32 = vfmaq_laneq_f32(acc32, b2, a_lo, 3);

            acc40 = vfmaq_laneq_f32(acc40, b0, a_hi, 0);
            acc41 = vfmaq_laneq_f32(acc41, b1, a_hi, 0);
            acc42 = vfmaq_laneq_f32(acc42, b2, a_hi, 0);
            acc50 = vfmaq_laneq_f32(acc50, b0, a_hi, 1);
            acc51 = vfmaq_laneq_f32(acc51, b1, a_hi, 1);
            acc52 = vfmaq_laneq_f32(acc52, b2, a_hi, 1);
            acc60 = vfmaq_laneq_f32(acc60, b0, a_hi, 2);
            acc61 = vfmaq_laneq_f32(acc61, b1, a_hi, 2);
            acc62 = vfmaq_laneq_f32(acc62, b2, a_hi, 2);
            acc70 = vfmaq_laneq_f32(acc70, b0, a_hi, 3);
            acc71 = vfmaq_laneq_f32(acc71, b1, a_hi, 3);
            acc72 = vfmaq_laneq_f32(acc72, b2, a_hi, 3);

            packed_A += 8;
            packed_B += 12;
        }
    }

    // K-tail (remaining 0-3 iterations)
    for (; k < K; ++k) {
        float32x4_t a_lo = vld1q_f32(packed_A);
        float32x4_t a_hi = vld1q_f32(packed_A + 4);
        float32x4_t b0 = vld1q_f32(packed_B);
        float32x4_t b1 = vld1q_f32(packed_B + 4);
        float32x4_t b2 = vld1q_f32(packed_B + 8);

        acc00 = vfmaq_laneq_f32(acc00, b0, a_lo, 0);
        acc01 = vfmaq_laneq_f32(acc01, b1, a_lo, 0);
        acc02 = vfmaq_laneq_f32(acc02, b2, a_lo, 0);
        acc10 = vfmaq_laneq_f32(acc10, b0, a_lo, 1);
        acc11 = vfmaq_laneq_f32(acc11, b1, a_lo, 1);
        acc12 = vfmaq_laneq_f32(acc12, b2, a_lo, 1);
        acc20 = vfmaq_laneq_f32(acc20, b0, a_lo, 2);
        acc21 = vfmaq_laneq_f32(acc21, b1, a_lo, 2);
        acc22 = vfmaq_laneq_f32(acc22, b2, a_lo, 2);
        acc30 = vfmaq_laneq_f32(acc30, b0, a_lo, 3);
        acc31 = vfmaq_laneq_f32(acc31, b1, a_lo, 3);
        acc32 = vfmaq_laneq_f32(acc32, b2, a_lo, 3);

        acc40 = vfmaq_laneq_f32(acc40, b0, a_hi, 0);
        acc41 = vfmaq_laneq_f32(acc41, b1, a_hi, 0);
        acc42 = vfmaq_laneq_f32(acc42, b2, a_hi, 0);
        acc50 = vfmaq_laneq_f32(acc50, b0, a_hi, 1);
        acc51 = vfmaq_laneq_f32(acc51, b1, a_hi, 1);
        acc52 = vfmaq_laneq_f32(acc52, b2, a_hi, 1);
        acc60 = vfmaq_laneq_f32(acc60, b0, a_hi, 2);
        acc61 = vfmaq_laneq_f32(acc61, b1, a_hi, 2);
        acc62 = vfmaq_laneq_f32(acc62, b2, a_hi, 2);
        acc70 = vfmaq_laneq_f32(acc70, b0, a_hi, 3);
        acc71 = vfmaq_laneq_f32(acc71, b1, a_hi, 3);
        acc72 = vfmaq_laneq_f32(acc72, b2, a_hi, 3);

        packed_A += 8;
        packed_B += 12;
    }

    // Epilogue: C = alpha * acc + beta * C
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    // Helper macro: scale accumulator and write one row
#define STORE_ROW(row, c0, c1, c2) do {                            \
    float* Crow = C + (row) * ldc;                                 \
    if (beta == 0.0f) {                                            \
        vst1q_f32(Crow,     vmulq_f32(alpha_v, c0));              \
        vst1q_f32(Crow + 4, vmulq_f32(alpha_v, c1));              \
        vst1q_f32(Crow + 8, vmulq_f32(alpha_v, c2));              \
    } else {                                                       \
        vst1q_f32(Crow,     vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Crow)),     alpha_v, c0)); \
        vst1q_f32(Crow + 4, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Crow + 4)), alpha_v, c1)); \
        vst1q_f32(Crow + 8, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Crow + 8)), alpha_v, c2)); \
    }                                                              \
} while(0)

    STORE_ROW(0, acc00, acc01, acc02);
    STORE_ROW(1, acc10, acc11, acc12);
    STORE_ROW(2, acc20, acc21, acc22);
    STORE_ROW(3, acc30, acc31, acc32);
    STORE_ROW(4, acc40, acc41, acc42);
    STORE_ROW(5, acc50, acc51, acc52);
    STORE_ROW(6, acc60, acc61, acc62);
    STORE_ROW(7, acc70, acc71, acc72);

#undef STORE_ROW
}

#endif  // __ARM_NEON

// ============================================================
// Registry wrappers + auto-registration
// ============================================================

// Original packing functions (defined in gemm_pack_fp32.cpp)
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

namespace {

void ukernel_fp32_neon_wrap(int K, const void* packed_A, const void* packed_B,
                            float* C, int ldc, float alpha, float beta,
                            float /*extra*/) {
#ifdef __ARM_NEON
    gemm_ukernel_fp32_8x12(K,
                            static_cast<const float*>(packed_A),
                            static_cast<const float*>(packed_B),
                            C, ldc, alpha, beta);
#else
    (void)K; (void)packed_A; (void)packed_B; (void)C; (void)ldc; (void)alpha; (void)beta;
#endif
}

void pack_a_fp32_wrap(int m_len, int k_len, const float* A, int lda,
                      void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

void pack_b_fp32_wrap(int k_len, int n_len, const float* B, int ldb,
                      void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

const GemmMicrokernelDesc neon_fp32_desc = {
    "neon_fp32_8x12",
    GemmDataType::kFP32,
    kNEON,                // required_hwcaps
    kGemmMrFp32,          // Mr = 8
    kGemmNrFp32,          // Nr = 12
    1,                    // Kgroup
    false,                // nr_is_vla
    100,                  // priority
    sizeof(float),        // packed_a_elem_bytes
    sizeof(float),        // packed_b_elem_bytes
    0,                    // min_sve_bits
    ukernel_fp32_neon_wrap,
    pack_a_fp32_wrap,
    pack_b_fp32_wrap,
};

static RegisterKernel reg_neon_fp32(neon_fp32_desc);

}  // namespace

}  // namespace dnnopt
