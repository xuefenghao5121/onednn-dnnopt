/// @file bench_gemm.cpp
/// GEMM (General Matrix Multiply) benchmark suite.
/// Tests naive and reference implementations across multiple precisions and shapes.

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ============================================================
// Naive GEMM implementations (baseline for correctness & comparison)
// ============================================================

/// Naive FP32 GEMM: C = alpha * A * B + beta * C
/// A: M×K, B: K×N, C: M×N, all row-major.
void gemm_naive_fp32(int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}

#ifdef __ARM_NEON
#include <arm_neon.h>

/// NEON-optimized FP32 GEMM with FMLA.
/// Basic vectorization over N dimension, no blocking.
void gemm_neon_fp32(int M, int N, int K,
                    float alpha, const float* A, int lda,
                    const float* B, int ldb,
                    float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        int j = 0;
        // Process 16 columns at a time (4 NEON registers)
        for (; j + 15 < N; j += 16) {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);

            for (int k = 0; k < K; ++k) {
                float32x4_t a_val = vdupq_n_f32(A[i * lda + k]);
                float32x4_t b0 = vld1q_f32(&B[k * ldb + j]);
                float32x4_t b1 = vld1q_f32(&B[k * ldb + j + 4]);
                float32x4_t b2 = vld1q_f32(&B[k * ldb + j + 8]);
                float32x4_t b3 = vld1q_f32(&B[k * ldb + j + 12]);
                c0 = vfmaq_f32(c0, a_val, b0);
                c1 = vfmaq_f32(c1, a_val, b1);
                c2 = vfmaq_f32(c2, a_val, b2);
                c3 = vfmaq_f32(c3, a_val, b3);
            }

            float32x4_t alpha_v = vdupq_n_f32(alpha);
            float32x4_t beta_v  = vdupq_n_f32(beta);

            float32x4_t out0 = vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(&C[i*ldc+j])),     alpha_v, c0);
            float32x4_t out1 = vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(&C[i*ldc+j+4])),   alpha_v, c1);
            float32x4_t out2 = vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(&C[i*ldc+j+8])),   alpha_v, c2);
            float32x4_t out3 = vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(&C[i*ldc+j+12])),  alpha_v, c3);

            vst1q_f32(&C[i*ldc+j],    out0);
            vst1q_f32(&C[i*ldc+j+4],  out1);
            vst1q_f32(&C[i*ldc+j+8],  out2);
            vst1q_f32(&C[i*ldc+j+12], out3);
        }
        // Process 4 columns at a time
        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0);
            for (int k = 0; k < K; ++k) {
                float32x4_t a_val = vdupq_n_f32(A[i * lda + k]);
                float32x4_t b0 = vld1q_f32(&B[k * ldb + j]);
                c0 = vfmaq_f32(c0, a_val, b0);
            }
            float32x4_t alpha_v = vdupq_n_f32(alpha);
            float32x4_t beta_v  = vdupq_n_f32(beta);
            float32x4_t out = vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(&C[i*ldc+j])), alpha_v, c0);
            vst1q_f32(&C[i*ldc+j], out);
        }
        // Scalar tail
        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}
#endif

// ============================================================
// Random initialization
// ============================================================
void fill_random(float* data, size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

// ============================================================
// GEMM shapes for benchmarking
// ============================================================
struct GemmShape {
    int M, N, K;
    const char* label;
};

const GemmShape gemm_shapes[] = {
    // Square matrices
    {128,  128,  128,  "Square-128"},
    {256,  256,  256,  "Square-256"},
    {512,  512,  512,  "Square-512"},
    {1024, 1024, 1024, "Square-1024"},
    {2048, 2048, 2048, "Square-2048"},

    // BERT-base shapes (seq=128, hidden=768)
    {128, 768,  768,   "BERT-QKV"},
    {128, 3072, 768,   "BERT-FFN1"},
    {128, 768,  3072,  "BERT-FFN2"},

    // Attention shapes (batch*heads=12, seq=128, head_dim=64)
    {128, 128, 64,     "Attn-QK"},
    {128, 64,  128,    "Attn-V"},

    // ResNet FC layer
    {1,   1000, 2048,  "ResNet-FC"},

    // GPT-2 shapes (seq=1024, hidden=768)
    {1024, 768,  768,  "GPT2-QKV"},
    {1024, 3072, 768,  "GPT2-FFN1"},
};

}  // namespace

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  GEMM Benchmark Suite\n");
    printf("==========================================================\n\n");

    // Print hardware info
    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);
    printf("FP32 peak: %.2f GFLOPS/core\n\n", hw.fp32_gflops_per_core);

    int warmup = 3;
    int runs   = 10;
    if (argc > 1) runs = atoi(argv[1]);

    std::vector<dnnopt::BenchStats> all_results;

    for (const auto& shape : gemm_shapes) {
        int M = shape.M, N = shape.N, K = shape.K;
        size_t a_size = (size_t)M * K;
        size_t b_size = (size_t)K * N;
        size_t c_size = (size_t)M * N;

        auto A = dnnopt::aligned_array<float>(a_size);
        auto B = dnnopt::aligned_array<float>(b_size);
        auto C = dnnopt::aligned_array<float>(c_size);

        fill_random(A.get(), a_size);
        fill_random(B.get(), b_size);
        memset(C.get(), 0, c_size * sizeof(float));

        double flops = 2.0 * M * N * K;  // MxNxK MACs = 2*M*N*K FLOP
        double bytes = (a_size + b_size + c_size) * sizeof(float);

        // Naive GEMM
        if (M * N * K <= 512 * 512 * 512) {  // Skip very large for naive
            char name[128];
            snprintf(name, sizeof(name), "%s [%dx%dx%d] naive", shape.label, M, N, K);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                memset(C.get(), 0, c_size * sizeof(float));
                gemm_naive_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

#ifdef __ARM_NEON
        {
            char name[128];
            snprintf(name, sizeof(name), "%s [%dx%dx%d] neon", shape.label, M, N, K);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                memset(C.get(), 0, c_size * sizeof(float));
                gemm_neon_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }
#endif
        printf("\n");
    }

    // Write CSV
    dnnopt::write_csv("bench_gemm_results.csv", all_results);

    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}
