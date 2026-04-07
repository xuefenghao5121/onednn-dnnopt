/// @file gemm_pack_int8.cpp
/// INT8 matrix packing with FP32→INT8 symmetric quantization for SMMLA GEMM.
///
/// SMMLA Vd.4S, Vn.16B, Vm.16B:
///   Vn = 2×8 INT8 matrix (2 rows, 8 K-elements)
///   Vm = 8×2 INT8 matrix (8 K-elements, 2 columns)
///   Vd = 2×2 INT32 accumulated result

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

/// Compute symmetric quantization scale: scale = max(|matrix|) / 127.0f
/// SVE-accelerated when available: uses svmaxv for fast horizontal reduction.
static float compute_quant_scale(const float* data, int rows, int cols, int ld) {
#ifdef __ARM_FEATURE_SVE
    // SVE path: wider vectors + svmaxv for fast abs-max
    svfloat32_t vmax_sve = svdup_f32(0);
    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * ld;
        int j = 0;
        for (; j < cols; ) {
            svbool_t pg = svwhilelt_b32(j, cols);
            svfloat32_t v = svld1_f32(pg, row + j);
            vmax_sve = svmax_f32_m(svptrue_b32(), vmax_sve, svabs_f32_x(pg, v));
            j += (int)svcntw();
        }
    }
    float amax = svmaxv_f32(svptrue_b32(), vmax_sve);
#else
    // NEON path
    float32x4_t vmax = vdupq_n_f32(0);
    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * ld;
        int j = 0;
        for (; j + 3 < cols; j += 4) {
            float32x4_t v = vld1q_f32(row + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        for (; j < cols; ++j) {
            float av = std::fabs(row[j]);
            vmax = vmaxq_f32(vmax, vdupq_n_f32(av));
        }
    }
    float amax = vmaxvq_f32(vmax);
#endif
    if (amax == 0.0f) return 1.0f;
    return amax / 127.0f;
}

/// Quantize a single FP32 value to INT8 with given scale.
static inline int8_t quantize_s8(float val, float inv_scale) {
    int32_t q = (int32_t)lrintf(val * inv_scale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

/// Pack A (FP32, row-major) into INT8 SMMLA format.
///
/// For each K-group of 8, pack row pairs (2 rows × 8 K-elements = 16 INT8):
///   [A_q[r0,k0..k7], A_q[r1,k0..k7]]
///
/// Total per K-group: Mr/2 * 16 INT8 = 4 * 16 = 64 bytes
void pack_a_int8(int m_len, int k_len,
                 const float* A, int lda,
                 int8_t* packed_A, float* scale_A) {
    constexpr int Mr = kGemmMrInt8;  // 8
    constexpr int Kgroup = 8;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    *scale_A = compute_quant_scale(A, m_len, k_len, lda);
    float inv_scale = 1.0f / *scale_A;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 row-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;

                for (int kk = 0; kk < Kgroup; ++kk) {
                    float v0 = 0, v1 = 0;
                    if (r0 < m_len && (k + kk) < k_len) v0 = A[r0 * lda + k + kk];
                    if (r1 < m_len && (k + kk) < k_len) v1 = A[r1 * lda + k + kk];
                    packed_A[kk]          = quantize_s8(v0, inv_scale);
                    packed_A[Kgroup + kk] = quantize_s8(v1, inv_scale);
                }
                packed_A += 16;
            }
        }
    }
}

/// Pack B (FP32, row-major) into INT8 SMMLA format.
///
/// For each K-group of 8, pack column pairs:
///   SMMLA expects Vm: [k0c0..k7c0, k0c1..k7c1] (column-major within col-pair)
///
/// Total per K-group: Nr/2 * 16 INT8 = 4 * 16 = 64 bytes
void pack_b_int8(int k_len, int n_len,
                 const float* B, int ldb,
                 int8_t* packed_B, float* scale_B) {
    constexpr int Nr = kGemmNrInt8;  // 8
    constexpr int Kgroup = 8;
    int k_padded = (k_len + Kgroup - 1) / Kgroup * Kgroup;

    *scale_B = compute_quant_scale(B, k_len, n_len, ldb);
    float inv_scale = 1.0f / *scale_B;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Pack 4 column-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;

                // SMMLA expects: [k0c0..k7c0, k0c1..k7c1]
                for (int kk = 0; kk < Kgroup; ++kk) {
                    float v0 = 0, v1 = 0;
                    if (c0 < n_len && (k + kk) < k_len) v0 = B[(k + kk) * ldb + c0];
                    if (c1 < n_len && (k + kk) < k_len) v1 = B[(k + kk) * ldb + c1];
                    packed_B[kk]          = quantize_s8(v0, inv_scale);
                    packed_B[Kgroup + kk] = quantize_s8(v1, inv_scale);
                }
                packed_B += 16;
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
