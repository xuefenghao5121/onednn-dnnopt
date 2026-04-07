/// @file gemm_ukernel_bf16_sme.cpp
/// SME BF16 GEMM microkernel using BFMOPA (BF16 outer product accumulate).
///
/// COMPILE-ONLY: SME is not available on current hardware (Neoverse N2).
/// This kernel auto-activates on SME-capable CPUs (Neoverse V3+).
///
/// SME BF16 programming model:
///   1. SMSTART SM: enter streaming mode
///   2. ZERO {za}: clear ZA tile
///   3. BFMOPA za0.s, p0/m, p0/m, z0.h, z1.h: BF16 outer product into FP32 ZA
///      - z0: SVL_half BF16 elements from A (2x more than FP32)
///      - z1: SVL_half BF16 elements from B
///      - Accumulates into FP32 ZA tile
///      - SVL=512: 32 BF16 per vec → 32x32 element pairs, 16x16 FP32 output
///      - Actually: BFMOPA processes 2 BF16 elements per FP32 position
///      - Tile = SVL_words × SVL_words (same as FP32), K processed 2 at a time
///   4. MOVA: extract rows from ZA
///   5. SMSTOP SM: exit streaming mode
///
/// BFMOPA operates like FMOPA but consumes BF16 inputs, processing K+=2 per
/// instruction (each BF16 pair is multiplied and accumulated into FP32).
/// The ZA tile size in FP32 words is SVL/32 × SVL/32 (same as FP32 FMOPA).
/// But A/B are packed as BF16, so the packing layout provides 2K per BFMOPA.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(DNNOPT_HAS_SME)

#include <arm_neon.h>
#include <cstring>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SME BF16 BFMOPA microkernel
// ============================================================

/// Query streaming SVE vector length (SVL) in bytes.
static inline uint64_t sme_svl_bytes_bf16() {
    uint64_t svl;
    asm volatile(
        ".arch_extension sme\n\t"
        "rdsvl %0, #1\n\t"
        : "=r"(svl)
    );
    return svl;
}

/// SME BFMOPA-based BF16 GEMM microkernel.
/// Tile: SVL_words x SVL_words FP32 accumulators.
/// Input: BF16 packed panels. BFMOPA processes 2 K-elements per instruction.
///
/// packed_A: BF16, layout: for each K-pair, SVL_half contiguous BF16
///           (SVL_half = SVL_bytes/2 = 2*SVL_words BF16 elements per load)
///           But we're processing K in groups of 2, so each load carries
///           SVL_words rows × 2 K-elements as interleaved BF16.
///
/// For simplicity, we use the FP32 packing format and convert in-register.
/// The outer driver packs A/B as FP32, and we convert to BF16 at load time.
/// This avoids a separate BF16 packing format for SME.
static void gemm_ukernel_bf16_sme(int K,
                                    const float* packed_A,
                                    const float* packed_B,
                                    float* C, int ldc,
                                    float alpha, float beta) {
    const uint64_t svl_bytes = sme_svl_bytes_bf16();
    const int svl_words = (int)(svl_bytes / 4);

    float za_rows[16 * 16];

    // Enter streaming mode and zero ZA
    asm volatile(
        ".arch_extension sme\n\t"
        "smstart sm\n\t"
        "zero {za}\n\t"
        ::: "memory"
    );

    // K-loop using FMOPA (FP32 mode) since our packing is FP32.
    // True BF16 BFMOPA would need BF16-packed data; we use FMOPA here
    // for correctness with FP32-packed inputs. On real SME hardware,
    // a dedicated BF16 packing path would enable BFMOPA's 2x K throughput.
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

    // Epilogue: C = alpha * za_rows + beta * C
    float32x4_t alpha_v = vdupq_n_f32(alpha);
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
            for (; j < svl_words; ++j) Cr[j] = alpha * src[j];
        } else {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                float32x4_t c_old = vld1q_f32(Cr + j);
                vst1q_f32(Cr + j, vfmaq_f32(vmulq_f32(beta_v, c_old), alpha_v, acc));
            }
            for (; j < svl_words; ++j)
                Cr[j] = alpha * src[j] + beta * Cr[j];
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void ukernel_bf16_sme_wrap(int K, const void* packed_A,
                                    const void* packed_B,
                                    float* C, int ldc, float alpha,
                                    float beta, float /*extra*/) {
    // BF16 SME uses FP32 packing + FMOPA (see comment in kernel above)
    gemm_ukernel_bf16_sme(K,
                            static_cast<const float*>(packed_A),
                            static_cast<const float*>(packed_B),
                            C, ldc, alpha, beta);
}

// Reuse FP32 packing for SME BF16 path (FMOPA with FP32 inputs)
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

static void pack_a_bf16_sme_wrap(int m_len, int k_len, const float* A, int lda,
                                   void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

static void pack_b_bf16_sme_wrap(int k_len, int n_len, const float* B, int ldb,
                                   void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

// SME BF16: uses FMOPA (FP32 mode) with FP32-packed data
// True BFMOPA would need dedicated BF16 packing for 2x K throughput
// Priority 300: highest when SME available
static const GemmMicrokernelDesc sme_bf16_desc = {
    "sme_bf16_VLxVL",
    GemmDataType::kBF16,
    kSME | kBF16,         // required_hwcaps
    0,                    // Mr (VLA)
    0,                    // Nr (VLA)
    1,                    // Kgroup (FMOPA mode, K=1 per instruction)
    true,                 // nr_is_vla
    300,                  // priority: highest
    sizeof(float),        // packed as FP32 (FMOPA path)
    sizeof(float),
    0,
    ukernel_bf16_sme_wrap,
    pack_a_bf16_sme_wrap,
    pack_b_bf16_sme_wrap,
};

static RegisterKernel reg_sme_bf16(sme_bf16_desc);

}  // namespace dnnopt

#endif  // DNNOPT_HAS_SME
