/// @file gemm_driver_bf16.cpp
/// BLIS-style BF16 GEMM driver with cache blocking and packing.
/// Input/output are FP32; internal computation uses BF16 BFMMLA.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// BF16 packing (defined in gemm_pack_bf16.cpp)
void pack_a_bf16(int m_len, int k_len, const float* A, int lda, bfloat16_t* packed_A);
void pack_b_bf16(int k_len, int n_len, const float* B, int ldb, bfloat16_t* packed_B);
void pack_a_bf16_direct(int m_len, int k_len, const bfloat16_t* A, int lda, bfloat16_t* packed_A);
void pack_b_bf16_direct(int k_len, int n_len, const bfloat16_t* B, int ldb, bfloat16_t* packed_B);

// BF16 microkernel (defined in gemm_ukernel_bf16_neon.cpp)
void gemm_ukernel_bf16_8x8(int K, const bfloat16_t* packed_A, const bfloat16_t* packed_B,
                            float* C, int ldc, float alpha, float beta);

void gemm_driver_bf16(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc) {
    constexpr int Mr = kGemmMrBf16;  // 8
    constexpr int Nr = kGemmNrBf16;  // 8
    constexpr int Kgroup = 4;

    auto bp = get_gemm_blocking_params();
    int Mc = bp.Mc;
    int Nc = bp.Nc;
    // BF16 is 2x denser, allow larger Kc for better compute/pack ratio
    int Kc = std::min(bp.Kc * 2, K);
    // Round Kc up to multiple of 4
    Kc = (Kc + Kgroup - 1) / Kgroup * Kgroup;

    // Packing buffers (BF16)
    // A: ceil(Mc/Mr) * Mr * ceil(Kc/4) * 4 BF16 elements per Mr-panel
    // Packed A size per Mr-panel per K-group: Mr/2 * 8 BF16 = 32 BF16
    // Total: ceil(Mc/Mr) * ceil(Kc/4) * 32 BF16
    int m_panels_max = (Mc + Mr - 1) / Mr;
    int k_groups_max = (Kc + Kgroup - 1) / Kgroup;
    size_t packed_a_size = (size_t)m_panels_max * k_groups_max * 32;

    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_b_size = (size_t)n_panels_max * k_groups_max * 32;

    auto packed_A = aligned_array<bfloat16_t>(packed_a_size);
    auto packed_B = aligned_array<bfloat16_t>(packed_b_size);

    // Edge buffer for partial tiles
    float edge_buf[Mr * Nr];

    // BLIS 5-loop
    for (int jc = 0; jc < N; jc += Nc) {
        int nc = std::min(Nc, N - jc);

        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            int kc_padded = (kc + Kgroup - 1) / Kgroup * Kgroup;

            float beta_eff = (pc == 0) ? beta : 1.0f;
            // Alpha must scale ALL K-chunks equally. Apply alpha on every
            // chunk and adjust beta accordingly for the first iteration:
            //   pc==0: C = alpha*A0*B0 + beta*C
            //   pc >0: C = alpha*Ai*Bi + 1.0*C  (accumulate into already-scaled C)
            float alpha_eff = alpha;

            // Pack B panel
            pack_b_bf16(kc, nc, &B[pc * ldb + jc], ldb, packed_B.get());

            for (int ic = 0; ic < M; ic += Mc) {
                int mc = std::min(Mc, M - ic);

                // Pack A block
                pack_a_bf16(mc, kc, &A[ic * lda + pc], lda, packed_A.get());

                int m_panels = (mc + Mr - 1) / Mr;
                int n_panels = (nc + Nr - 1) / Nr;

                // Packed panel sizes in BF16 elements
                // Per Mr-panel: (kc_padded / 4) * 32 BF16
                size_t a_panel_stride = (size_t)(kc_padded / Kgroup) * 32;
                size_t b_panel_stride = (size_t)(kc_padded / Kgroup) * 32;

                for (int jr = 0; jr < n_panels; jr++) {
                    int n_start = jr * Nr;
                    int n_rem = std::min(Nr, nc - n_start);
                    const bfloat16_t* B_panel = packed_B.get() + jr * b_panel_stride;

                    for (int ir = 0; ir < m_panels; ir++) {
                        int m_start = ir * Mr;
                        int m_rem = std::min(Mr, mc - m_start);
                        const bfloat16_t* A_panel = packed_A.get() + ir * a_panel_stride;

                        float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                        if (m_rem == Mr && n_rem == Nr) {
                            gemm_ukernel_bf16_8x8(kc_padded, A_panel, B_panel,
                                                  C_ptr, ldc, alpha_eff, beta_eff);
                        } else {
                            // Edge tile
                            memset(edge_buf, 0, sizeof(edge_buf));
                            if (beta_eff != 0.0f) {
                                for (int i = 0; i < m_rem; i++)
                                    memcpy(&edge_buf[i * Nr], &C_ptr[i * ldc],
                                           n_rem * sizeof(float));
                            }
                            gemm_ukernel_bf16_8x8(kc_padded, A_panel, B_panel,
                                                  edge_buf, Nr, alpha_eff, beta_eff);
                            for (int i = 0; i < m_rem; i++)
                                memcpy(&C_ptr[i * ldc], &edge_buf[i * Nr],
                                       n_rem * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

/// BF16 GEMM with native bfloat16 input (for oneDNN integration).
/// A: M×K bfloat16 row-major, B: K×N bfloat16 row-major, C: M×N float row-major.
/// Computes: C = alpha * A * B + beta * C
void gemm_bf16_bf16bf16f32(int M, int N, int K,
                           float alpha, const bfloat16_t* A, int lda,
                           const bfloat16_t* B, int ldb,
                           float beta, float* C, int ldc) {
    constexpr int Mr = kGemmMrBf16;  // 8
    constexpr int Nr = kGemmNrBf16;  // 8
    constexpr int Kgroup = 4;

    auto bp = get_gemm_blocking_params();
    int Mc = bp.Mc;
    int Nc = bp.Nc;
    int Kc = std::min(bp.Kc * 2, K);
    Kc = (Kc + Kgroup - 1) / Kgroup * Kgroup;

    int m_panels_max = (Mc + Mr - 1) / Mr;
    int k_groups_max = (Kc + Kgroup - 1) / Kgroup;
    size_t packed_a_size = (size_t)m_panels_max * k_groups_max * 32;

    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_b_size = (size_t)n_panels_max * k_groups_max * 32;

    auto packed_A = aligned_array<bfloat16_t>(packed_a_size);
    auto packed_B = aligned_array<bfloat16_t>(packed_b_size);

    float edge_buf[Mr * Nr];

    for (int jc = 0; jc < N; jc += Nc) {
        int nc = std::min(Nc, N - jc);

        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            int kc_padded = (kc + Kgroup - 1) / Kgroup * Kgroup;

            float beta_eff = (pc == 0) ? beta : 1.0f;
            float alpha_eff = alpha;

            // Pack B panel (direct BF16 copy)
            pack_b_bf16_direct(kc, nc, &B[pc * ldb + jc], ldb, packed_B.get());

            for (int ic = 0; ic < M; ic += Mc) {
                int mc = std::min(Mc, M - ic);

                // Pack A block (direct BF16 copy)
                pack_a_bf16_direct(mc, kc, &A[ic * lda + pc], lda, packed_A.get());

                int m_panels = (mc + Mr - 1) / Mr;
                int n_panels = (nc + Nr - 1) / Nr;

                size_t a_panel_stride = (size_t)(kc_padded / Kgroup) * 32;
                size_t b_panel_stride = (size_t)(kc_padded / Kgroup) * 32;

                for (int jr = 0; jr < n_panels; jr++) {
                    int n_start = jr * Nr;
                    int n_rem = std::min(Nr, nc - n_start);
                    const bfloat16_t* B_panel = packed_B.get() + jr * b_panel_stride;

                    for (int ir = 0; ir < m_panels; ir++) {
                        int m_start = ir * Mr;
                        int m_rem = std::min(Mr, mc - m_start);
                        const bfloat16_t* A_panel = packed_A.get() + ir * a_panel_stride;

                        float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                        if (m_rem == Mr && n_rem == Nr) {
                            gemm_ukernel_bf16_8x8(kc_padded, A_panel, B_panel,
                                                  C_ptr, ldc, alpha_eff, beta_eff);
                        } else {
                            memset(edge_buf, 0, sizeof(edge_buf));
                            if (beta_eff != 0.0f) {
                                for (int i = 0; i < m_rem; i++)
                                    memcpy(&edge_buf[i * Nr], &C_ptr[i * ldc],
                                           n_rem * sizeof(float));
                            }
                            gemm_ukernel_bf16_8x8(kc_padded, A_panel, B_panel,
                                                  edge_buf, Nr, alpha_eff, beta_eff);
                            for (int i = 0; i < m_rem; i++)
                                memcpy(&C_ptr[i * ldc], &edge_buf[i * Nr],
                                       n_rem * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
