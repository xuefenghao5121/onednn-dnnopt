/// @file gemm_driver_generic.cpp
/// Generic parameterized BLIS-style GEMM driver with OpenMP parallelization.
/// Replaces the three per-type drivers with a single implementation
/// that works with any microkernel via function pointers.

#include "dnnopt/gemm/gemm_driver_generic.h"
#include "dnnopt/gemm/gemm_threading.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dnnopt {

void gemm_driver_generic(int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc,
                         const GemmDriverConfig& cfg) {
    const int Mr = cfg.Mr;
    const int Nr = cfg.Nr;
    const int Kgroup = cfg.Kgroup;
    const int Mc = cfg.Mc;
    const int Nc = cfg.Nc;
    int Kc = cfg.Kc;
    if (Kgroup > 1) Kc = (Kc / Kgroup) * Kgroup;
    if (Kc < Kgroup) Kc = Kgroup;

    const int a_elem = cfg.packed_a_elem_bytes;
    const int b_elem = cfg.packed_b_elem_bytes;

    // Panel stride lambdas
    auto a_panel_stride = [&](int kc_padded) -> size_t {
        if (cfg.dtype == GemmDataType::kFP32)
            return (size_t)Mr * kc_padded * a_elem;
        return (size_t)(kc_padded / Kgroup) * Mr * Kgroup * a_elem;
    };
    auto b_panel_stride = [&](int kc_padded) -> size_t {
        if (cfg.dtype == GemmDataType::kFP32)
            return (size_t)Nr * kc_padded * b_elem;
        return (size_t)(kc_padded / Kgroup) * Nr * Kgroup * b_elem;
    };

    // Determine thread count
    int num_threads = gemm_get_num_threads();
    // Small problems: skip threading to avoid fork/join overhead
    int64_t flops = (int64_t)2 * M * N * K;
    if (flops < 200000) num_threads = 1;

    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_b_size = (size_t)n_panels_max * b_panel_stride(Kc);

    // Shared B packing buffer
    auto packed_B = aligned_array<uint8_t>(packed_b_size);

    // Per-thread A packing buffers and edge buffers
    int m_panels_max = (Mc + Mr - 1) / Mr;
    size_t packed_a_size = (size_t)m_panels_max * a_panel_stride(Kc);

    std::vector<AlignedPtr<uint8_t>> packed_A_bufs(num_threads);
    for (int t = 0; t < num_threads; ++t)
        packed_A_bufs[t] = aligned_array<uint8_t>(packed_a_size);

    // Per-thread edge buffers
    std::vector<std::vector<float>> edge_bufs(num_threads);
    for (int t = 0; t < num_threads; ++t)
        edge_bufs[t].resize(Mr * Nr, 0.0f);

    // Loop 1: N dimension (L3 blocking)
    for (int jc = 0; jc < N; jc += Nc) {
        int nc = std::min(Nc, N - jc);

        // Loop 2: K dimension (L2 blocking)
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            int kc_padded = (Kgroup > 1)
                ? ((kc + Kgroup - 1) / Kgroup) * Kgroup
                : kc;

            float beta_eff  = (pc == 0) ? beta : 1.0f;
            float alpha_eff = (pc + kc >= K) ? alpha : 1.0f;

            // Pack B panel (shared across threads)
            float scale_B = 1.0f;
            cfg.pack_b(kc, nc, &B[pc * ldb + jc], ldb,
                       packed_B.get(), Nr, &scale_B);

            size_t a_stride = a_panel_stride(kc_padded);
            size_t b_stride = b_panel_stride(kc_padded);
            int n_panels = (nc + Nr - 1) / Nr;

            // Loop 3: M dimension — parallelized with OpenMP
            // Each thread handles independent Mc-blocks: different A rows, different C rows
            int n_mc_blocks = (M + Mc - 1) / Mc;

#ifdef _OPENMP
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic) \
                if(num_threads > 1)
#endif
            for (int mc_idx = 0; mc_idx < n_mc_blocks; mc_idx++) {
                int ic = mc_idx * Mc;
                int mc = std::min(Mc, M - ic);

#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                uint8_t* my_packed_A = packed_A_bufs[tid].get();
                auto& my_edge_buf = edge_bufs[tid];

                // Pack A block (per-thread)
                float scale_A = 1.0f;
                cfg.pack_a(mc, kc, &A[ic * lda + pc], lda,
                           my_packed_A, Mr, &scale_A);

                float extra = (cfg.dtype == GemmDataType::kINT8)
                    ? (scale_A * scale_B) : 0.0f;

                int m_panels = (mc + Mr - 1) / Mr;

                // Loop 4+5: micro-tiles
                for (int jr = 0; jr < n_panels; jr++) {
                    int n_start = jr * Nr;
                    int n_rem = std::min(Nr, nc - n_start);
                    const void* B_panel = packed_B.get() + jr * b_stride;

                    for (int ir = 0; ir < m_panels; ir++) {
                        int m_start = ir * Mr;
                        int m_rem = std::min(Mr, mc - m_start);
                        const void* A_panel = my_packed_A + ir * a_stride;

                        float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                        if (m_rem == Mr && n_rem == Nr) {
                            cfg.ukernel(kc_padded, A_panel, B_panel,
                                       C_ptr, ldc, alpha_eff, beta_eff, extra);
                        } else {
                            std::fill(my_edge_buf.begin(), my_edge_buf.end(), 0.0f);
                            if (beta_eff != 0.0f) {
                                for (int i = 0; i < m_rem; i++)
                                    memcpy(&my_edge_buf[i * Nr], &C_ptr[i * ldc],
                                           n_rem * sizeof(float));
                            }
                            cfg.ukernel(kc_padded, A_panel, B_panel,
                                       my_edge_buf.data(), Nr, alpha_eff, beta_eff,
                                       extra);
                            for (int i = 0; i < m_rem; i++)
                                memcpy(&C_ptr[i * ldc], &my_edge_buf[i * Nr],
                                       n_rem * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace dnnopt
