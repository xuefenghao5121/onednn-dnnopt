/// @file gemm_pack_bf16.cpp
/// BF16 matrix packing for BFMMLA GEMM.
/// Converts FP32 input to BF16 and rearranges into BFMMLA-friendly layout.
///
/// BFMMLA Vd.4S, Vn.8H, Vm.8H:
///   Vn = 2×4 BF16 matrix (2 rows, 4 K-elements)
///   Vm = 4×2 BF16 matrix (4 K-elements, 2 columns)
///   Vd = 2×2 FP32 accumulated result

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================================
// FP32 input versions (convert to BF16 during packing)
// ============================================================================

/// Pack A (FP32, row-major) into BF16 BFMMLA format.
///
/// For each K-group of 4, pack row pairs (2 rows × 4 K-elements = 8 BF16):
///   v0: [A[0,k], A[0,k+1], A[0,k+2], A[0,k+3], A[1,k], ..., A[1,k+3]]
///   v1: [A[2,k], ..., A[3,k+3]]
///   v2: [A[4,k], ..., A[5,k+3]]
///   v3: [A[6,k], ..., A[7,k+3]]
///
/// Total per K-group: Mr/2 * 8 BF16 = 4 * 8 = 32 BF16 = 64 bytes
/// K is rounded up to multiple of 4 (zero-padded).
/// m_len is rounded up to multiple of Mr (8, zero-padded).
void pack_a_bf16(int m_len, int k_len,
                 const float* A, int lda,
                 bfloat16_t* packed_A) {
    constexpr int Mr = kGemmMrBf16;  // 8
    constexpr int Kgroup = 4;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 row-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;

                // Load 4 FP32 from each row, convert to BF16
                float tmp0[4] = {0, 0, 0, 0};
                float tmp1[4] = {0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (r0 < m_len) tmp0[kk] = A[r0 * lda + k + kk];
                    if (r1 < m_len) tmp1[kk] = A[r1 * lda + k + kk];
                }

                // Convert to BF16: 4 FP32 from row0, 4 FP32 from row1 → 8 BF16
                float32x4_t f0 = vld1q_f32(tmp0);
                float32x4_t f1 = vld1q_f32(tmp1);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                // GCC's bfloat16_t is __bf16, need cast from dnnopt::bfloat16_t*
                vst1q_bf16(reinterpret_cast<__bf16*>(packed_A), combined);
                packed_A += 8;
            }
        }
    }
}

/// Pack B (FP32, row-major) into BF16 BFMMLA format.
///
/// For each K-group of 4, pack column pairs (4 K-elements × 2 columns = 8 BF16):
///   v4: [B[k,0], B[k,1], B[k+1,0], B[k+1,1], B[k+2,0], B[k+2,1], B[k+3,0], B[k+3,1]]
///   v5: [B[k,2], B[k,3], ...]
///   v6: [B[k,4], B[k,5], ...]
///   v7: [B[k,6], B[k,7], ...]
///
/// Total per K-group: Nr/2 * 8 BF16 = 4 * 8 = 32 BF16 = 64 bytes
void pack_b_bf16(int k_len, int n_len,
                 const float* B, int ldb,
                 bfloat16_t* packed_B) {
    constexpr int Nr = kGemmNrBf16;  // 8
    constexpr int Kgroup = 4;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 column-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;

                // BFMMLA expects Vm layout: [k0c0, k1c0, k2c0, k3c0, k0c1, k1c1, k2c1, k3c1]
                // First 4 = all K-elements for col 0, next 4 = all K-elements for col 1
                float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (c0 < n_len) tmp[kk]     = B[(k + kk) * ldb + c0];
                    if (c1 < n_len) tmp[4 + kk] = B[(k + kk) * ldb + c1];
                }

                // Convert 8 FP32 to 8 BF16
                float32x4_t f0 = vld1q_f32(tmp);
                float32x4_t f1 = vld1q_f32(tmp + 4);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                // GCC's bfloat16_t is __bf16, need cast from dnnopt::bfloat16_t*
                vst1q_bf16(reinterpret_cast<__bf16*>(packed_B), combined);
                packed_B += 8;
            }
        }
    }
}

// ============================================================================
// Direct BF16 input versions (no conversion needed)
// ============================================================================

/// Pack A (BF16, row-major) into BF16 BFMMLA format - direct copy.
///
/// Same layout as FP32 version, but input is already BF16.
/// For each K-group of 4, pack row pairs (2 rows × 4 K-elements = 8 BF16):
///   v0: [A[0,k], A[0,k+1], A[0,k+2], A[0,k+3], A[1,k], ..., A[1,k+3]]
void pack_a_bf16_direct(int m_len, int k_len,
                        const bfloat16_t* A, int lda,
                        bfloat16_t* packed_A) {
    constexpr int Mr = kGemmMrBf16;  // 8
    constexpr int Kgroup = 4;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 row-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;

                // Load 4 BF16 from each row
                uint16_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (r0 < m_len) tmp[kk]     = A[r0 * lda + k + kk].raw_bits;
                    if (r1 < m_len) tmp[4 + kk] = A[r1 * lda + k + kk].raw_bits;
                }

                // Store 8 BF16 directly (already in correct layout)
                uint16x8_t v = vld1q_u16(tmp);
                vst1q_u16(reinterpret_cast<uint16_t*>(packed_A), v);
                packed_A += 8;
            }
        }
    }
}

/// Pack B (BF16, row-major) into BF16 BFMMLA format - direct copy.
///
/// BFMMLA expects Vm layout: [k0c0, k1c0, k2c0, k3c0, k0c1, k1c1, k2c1, k3c1]
/// For each K-group of 4, pack column pairs (4 K-elements × 2 columns = 8 BF16).
void pack_b_bf16_direct(int k_len, int n_len,
                        const bfloat16_t* B, int ldb,
                        bfloat16_t* packed_B) {
    constexpr int Nr = kGemmNrBf16;  // 8
    constexpr int Kgroup = 4;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 column-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;

                // BFMMLA expects: [k0c0, k1c0, k2c0, k3c0, k0c1, k1c1, k2c1, k3c1]
                // First 4 = all K-elements for col 0, next 4 = all K-elements for col 1
                uint16_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (c0 < n_len) tmp[kk]     = B[(k + kk) * ldb + c0].raw_bits;
                    if (c1 < n_len) tmp[4 + kk] = B[(k + kk) * ldb + c1].raw_bits;
                }

                // Store 8 BF16 directly
                uint16x8_t v = vld1q_u16(tmp);
                vst1q_u16(reinterpret_cast<uint16_t*>(packed_B), v);
                packed_B += 8;
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
