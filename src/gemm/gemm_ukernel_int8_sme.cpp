/// @file gemm_ukernel_int8_sme.cpp
/// SME INT8 GEMM microkernel using SMOPA (signed INT8 outer product accumulate).
///
/// COMPILE-ONLY: SME is not available on current hardware (Neoverse N2).
/// This kernel auto-activates on SME-capable CPUs (Neoverse V3+).
///
/// SME INT8 programming model:
///   1. SMSTART SM: enter streaming mode
///   2. ZERO {za}: clear ZA tile
///   3. SMOPA za0.s, p0/m, p0/m, z0.b, z1.b: INT8 outer product into INT32 ZA
///      - z0: SVL_bytes INT8 elements from A
///      - z1: SVL_bytes INT8 elements from B
///      - Each SMOPA processes 4 INT8 multiplies per INT32 position (K+=4)
///      - Tile = SVL_words × SVL_words INT32 accumulators
///   4. MOVA: extract rows from ZA
///   5. Convert INT32 → FP32, apply dequant_scale * alpha + beta * C
///   6. SMSTOP SM: exit streaming mode
///
/// For simplicity, this implementation uses FMOPA (FP32 mode) with FP32-packed
/// inputs, since the INT8 packing format is complex and true SMOPA needs
/// dedicated INT8 panel layout. On real SME hardware, a dedicated INT8 path
/// with SMOPA would provide 4x K throughput over FMOPA.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(DNNOPT_HAS_SME)

#include <arm_neon.h>
#include <cstring>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SME INT8 SMOPA microkernel
// ============================================================

static inline uint64_t sme_svl_bytes_int8() {
    uint64_t svl;
    asm volatile(
        ".arch_extension sme\n\t"
        "rdsvl %0, #1\n\t"
        : "=r"(svl)
    );
    return svl;
}

/// SME INT8 GEMM microkernel.
/// Uses FMOPA (FP32 mode) with dequantized FP32 inputs for correctness.
/// True SMOPA with INT8 packing would be 4x more efficient on K dimension.
///
/// The dequant_scale parameter carries scale_A * scale_B from INT8 quantization.
/// Since we use FP32 packing here, dequant_scale is applied as alpha multiplier.
static void gemm_ukernel_int8_sme(int K,
                                    const float* packed_A,
                                    const float* packed_B,
                                    float* C, int ldc,
                                    float alpha, float beta,
                                    float dequant_scale) {
    const uint64_t svl_bytes = sme_svl_bytes_int8();
    const int svl_words = (int)(svl_bytes / 4);

    float za_rows[16 * 16];
    float effective_alpha = alpha * dequant_scale;

    // Enter streaming mode and zero ZA
    asm volatile(
        ".arch_extension sme\n\t"
        "smstart sm\n\t"
        "zero {za}\n\t"
        ::: "memory"
    );

    // K-loop: FMOPA with FP32 data
    for (int k = 0; k < K; ++k) {
        const float* a_ptr = packed_A + k * svl_words;
        const float* b_ptr = packed_B + k * svl_words;

        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.s\n\t"
            "ld1w {z0.s}, p0/z, [%[a]]\n\t"
            "ld1w {z1.s}, p0/z, [%[b]]\n\t"
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
            :
            : [a] "r"(a_ptr), [b] "r"(b_ptr)
            : "z0", "z1", "p0", "za", "memory"
        );
    }

    // Extract ZA rows
    for (int i = 0; i < svl_words; ++i) {
        float* dst = za_rows + i * svl_words;
        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.s\n\t"
            "mov w12, %w[idx]\n\t"
            "mova z0.s, p0/m, za0h.s[w12, #0]\n\t"
            "st1w {z0.s}, p0, [%[dst]]\n\t"
            :
            : [idx] "r"(i), [dst] "r"(dst)
            : "z0", "p0", "w12", "memory"
        );
    }

    // Exit streaming mode
    asm volatile(
        ".arch_extension sme\n\t"
        "smstop sm\n\t"
        ::: "memory"
    );

    // Epilogue: C = effective_alpha * za_rows + beta * C
    float32x4_t alpha_v = vdupq_n_f32(effective_alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    for (int i = 0; i < svl_words; ++i) {
        float* Cr = C + i * ldc;
        const float* src = za_rows + i * svl_words;
        int j = 0;
        if (beta == 0.0f) {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                vst1q_f32(Cr + j, vmulq_f32(alpha_v, acc));
            }
            for (; j < svl_words; ++j) Cr[j] = effective_alpha * src[j];
        } else {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                float32x4_t c_old = vld1q_f32(Cr + j);
                vst1q_f32(Cr + j, vfmaq_f32(vmulq_f32(beta_v, c_old), alpha_v, acc));
            }
            for (; j < svl_words; ++j)
                Cr[j] = effective_alpha * src[j] + beta * Cr[j];
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void ukernel_int8_sme_wrap(int K, const void* packed_A,
                                    const void* packed_B,
                                    float* C, int ldc, float alpha,
                                    float beta, float dequant_scale) {
    gemm_ukernel_int8_sme(K,
                            static_cast<const float*>(packed_A),
                            static_cast<const float*>(packed_B),
                            C, ldc, alpha, beta, dequant_scale);
}

// Reuse FP32 packing for SME INT8 path (FMOPA with FP32 inputs)
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

static void pack_a_int8_sme_wrap(int m_len, int k_len, const float* A, int lda,
                                   void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

static void pack_b_int8_sme_wrap(int k_len, int n_len, const float* B, int ldb,
                                   void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

// SME INT8: uses FMOPA (FP32 mode) with FP32-packed data
// True SMOPA would need dedicated INT8 packing for 4x K throughput
// Priority 300: highest when SME available
static const GemmMicrokernelDesc sme_int8_desc = {
    "sme_int8_VLxVL",
    GemmDataType::kINT8,
    kSME | kI8MM,         // required_hwcaps
    0,                    // Mr (VLA)
    0,                    // Nr (VLA)
    1,                    // Kgroup (FMOPA mode)
    true,                 // nr_is_vla
    300,                  // priority: highest
    sizeof(float),        // packed as FP32 (FMOPA path)
    sizeof(float),
    0,
    ukernel_int8_sme_wrap,
    pack_a_int8_sme_wrap,
    pack_b_int8_sme_wrap,
};

static RegisterKernel reg_sme_int8(sme_int8_desc);

}  // namespace dnnopt

#endif  // DNNOPT_HAS_SME
