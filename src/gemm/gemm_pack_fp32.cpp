/// @file gemm_pack_fp32.cpp
/// FP32 matrix packing for BLIS-style GEMM.
/// Packs A and B into microkernel-friendly contiguous layouts.

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace dnnopt {

/// Pack a block of A (m_len x k_len) into Mr-wide column panels.
///
/// Input:  A row-major, stride lda
/// Output: packed_A layout — for each k in [0, k_len):
///           Mr contiguous floats from column k of the block.
///         Total size: m_panels * Mr * k_len floats
///         where m_panels = ceil(m_len / Mr).
///
/// If m_len is not a multiple of Mr, the last panel is zero-padded.
void pack_a_fp32(int m_len, int k_len,
                 const float* A, int lda,
                 float* packed_A) {
    constexpr int Mr = kGemmMrFp32;  // 8

    for (int i = 0; i < m_len; i += Mr) {
        int m_rem = std::min(Mr, m_len - i);
        for (int k = 0; k < k_len; ++k) {
            int r = 0;
#ifdef __ARM_NEON
            // Full Mr=8 case: 2 NEON loads
            if (m_rem == Mr) {
                // Gather Mr elements from column k: A[i+r][k] for r in [0, Mr)
                // These are NOT contiguous in memory (stride = lda), so we gather
                float tmp[Mr];
                for (int rr = 0; rr < Mr; ++rr)
                    tmp[rr] = A[(i + rr) * lda + k];
                vst1q_f32(packed_A, vld1q_f32(tmp));
                vst1q_f32(packed_A + 4, vld1q_f32(tmp + 4));
                packed_A += Mr;
                continue;
            }
#endif
            // Scalar path (edge case or no NEON)
            for (; r < m_rem; ++r)
                *packed_A++ = A[(i + r) * lda + k];
            for (; r < Mr; ++r)
                *packed_A++ = 0.0f;  // zero-pad
        }
    }
}

/// Pack a block of B (k_len x n_len) into Nr-wide row panels.
///
/// Input:  B row-major, stride ldb
/// Output: packed_B layout — for each k in [0, k_len):
///           Nr contiguous floats from row k of the block.
///         Total size: n_panels * Nr * k_len floats
///         where n_panels = ceil(n_len / Nr).
///
/// If n_len is not a multiple of Nr, the last panel is zero-padded.
void pack_b_fp32(int k_len, int n_len,
                 const float* B, int ldb,
                 float* packed_B) {
    constexpr int Nr = kGemmNrFp32;  // 12

    for (int j = 0; j < n_len; j += Nr) {
        int n_rem = std::min(Nr, n_len - j);
        for (int k = 0; k < k_len; ++k) {
            const float* src = &B[k * ldb + j];
            if (n_rem == Nr) {
#ifdef __ARM_NEON
                // Nr=12 = 3 NEON registers, contiguous in source
                vst1q_f32(packed_B,     vld1q_f32(src));
                vst1q_f32(packed_B + 4, vld1q_f32(src + 4));
                vst1q_f32(packed_B + 8, vld1q_f32(src + 8));
#else
                memcpy(packed_B, src, Nr * sizeof(float));
#endif
                packed_B += Nr;
            } else {
                // Edge case: predicated copy + zero-pad
#ifdef __ARM_FEATURE_SVE
                // SVE predicated load handles edge cleanly
                int done = 0;
                while (done < n_rem) {
                    svbool_t pg = svwhilelt_b32(done, n_rem);
                    svfloat32_t v = svld1_f32(pg, src + done);
                    svst1_f32(pg, packed_B + done, v);
                    done += (int)svcntw();
                }
                for (int c = n_rem; c < Nr; ++c)
                    packed_B[c] = 0.0f;
#else
                int c = 0;
                for (; c < n_rem; ++c)
                    packed_B[c] = src[c];
                for (; c < Nr; ++c)
                    packed_B[c] = 0.0f;
#endif
                packed_B += Nr;
            }
        }
    }
}

}  // namespace dnnopt
