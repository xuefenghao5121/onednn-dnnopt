/// @file bench_small_gemm.cpp
/// Small and irregular GEMM shapes benchmark for autoGEMM validation.
/// Tests edge cases: tiny matrices, non-power-of-2, boundary conditions.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

// Naive GEMM for correctness verification
void gemm_naive(int M, int N, int K,
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

bool verify_correctness(int M, int N, int K,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C_dnnopt, int ldc) {
    size_t c_size = (size_t)M * N;
    auto C_ref = dnnopt::aligned_array<float>(c_size);
    memcpy(C_ref.get(), C_dnnopt, c_size * sizeof(float));

    gemm_naive(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C_ref.get(), ldc);

    float max_err = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float err = std::abs(C_dnnopt[i * ldc + j] - C_ref.get()[i * ldc + j]);
            max_err = std::max(max_err, err);
        }
    }
    // For larger K, accumulated error can be higher
    // Use relative tolerance: max(|err|) < max(1e-4, 1e-5 * K)
    float tolerance = std::max(1e-4f, 1e-5f * K);
    return max_err < tolerance;
}

struct GemmShape {
    int M, N, K;
    const char* label;
    double flops() const { return 2.0 * M * N * K; }
};

// Comprehensive small/irregular shapes
const GemmShape shapes[] = {
    // === Extremely tiny (GEMV-like) ===
    {1, 1, 1,        "1x1x1-GEMV"},
    {1, 4, 4,        "1x4x4"},
    {1, 8, 8,        "1x8x8"},
    {1, 16, 16,      "1x16x16"},
    {1, 32, 32,      "1x32x32"},
    {1, 64, 64,      "1x64x64"},
    {1, 128, 128,    "1x128x128"},
    {1, 256, 256,    "1x256x256"},
    {1, 512, 512,    "1x512x512"},

    // === Very small square ===
    {2, 2, 2,        "2x2x2"},
    {3, 3, 3,        "3x3x3"},
    {4, 4, 4,        "4x4x4"},
    {5, 5, 5,        "5x5x5"},
    {6, 6, 6,        "6x6x6"},
    {7, 7, 7,        "7x7x7"},
    {8, 8, 8,        "8x8x8"},
    {9, 9, 9,        "9x9x9"},
    {10, 10, 10,     "10x10x10"},
    {12, 12, 12,     "12x12x12"},
    {15, 15, 15,     "15x15x15"},
    {16, 16, 16,     "16x16x16"},
    {20, 20, 20,     "20x20x20"},
    {24, 24, 24,     "24x24x24"},
    {32, 32, 32,     "32x32x32"},

    // === Non-power-of-2 squares ===
    {3, 3, 64,       "3x3x64"},
    {7, 7, 7,        "7x7x7"},
    {13, 13, 13,     "13x13x13"},
    {17, 17, 17,     "17x17x17"},
    {19, 19, 19,     "19x19x19"},
    {23, 23, 23,     "23x23x23"},
    {31, 31, 31,     "31x31x31"},
    {33, 33, 33,     "33x33x33"},
    {47, 47, 47,     "47x47x47"},
    {63, 63, 63,     "63x63x63"},
    {65, 65, 65,     "65x65x65"},

    // === Asymmetric: M << N ===
    {2, 64, 64,      "2x64x64"},
    {2, 128, 128,    "2x128x128"},
    {2, 256, 256,    "2x256x256"},
    {4, 64, 64,      "4x64x64"},
    {4, 128, 128,    "4x128x128"},
    {4, 256, 256,    "4x256x256"},
    {8, 64, 64,      "8x64x64"},
    {8, 128, 128,    "8x128x128"},
    {8, 256, 256,    "8x256x256"},
    {16, 64, 64,     "16x64x64"},
    {16, 128, 128,   "16x128x128"},
    {16, 256, 256,   "16x256x256"},

    // === Asymmetric: N << M ===
    {64, 2, 64,      "64x2x64"},
    {128, 2, 128,    "128x2x128"},
    {256, 2, 256,    "256x2x256"},
    {64, 4, 64,      "64x4x64"},
    {128, 4, 128,    "128x4x128"},
    {256, 4, 256,    "256x4x256"},
    {64, 8, 64,      "64x8x64"},
    {128, 8, 128,    "128x8x128"},
    {256, 8, 256,    "256x8x256"},

    // === Asymmetric: K dimension ===
    {16, 16, 4,      "16x16x4"},
    {16, 16, 8,      "16x16x8"},
    {16, 16, 12,     "16x16x12"},
    {16, 16, 24,     "16x16x24"},
    {16, 16, 48,     "16x16x48"},
    {16, 16, 96,     "16x16x96"},
    {32, 32, 4,      "32x32x4"},
    {32, 32, 8,      "32x32x8"},
    {32, 32, 16,     "32x32x16"},
    {32, 32, 24,     "32x32x24"},

    // === Tall-skinny ===
    {128, 16, 16,    "128x16x16"},
    {256, 16, 16,    "256x16x16"},
    {512, 16, 16,    "512x16x16"},
    {1024, 16, 16,   "1024x16x16"},
    {128, 8, 8,      "128x8x8"},
    {256, 8, 8,      "256x8x8"},
    {512, 8, 8,      "512x8x8"},

    // === Wide-short ===
    {16, 128, 16,    "16x128x16"},
    {16, 256, 16,    "16x256x16"},
    {16, 512, 16,    "16x512x16"},
    {8, 128, 8,      "8x128x8"},
    {8, 256, 8,      "8x256x8"},
    {8, 512, 8,      "8x512x8"},

    // === DL-specific: attention-like ===
    {128, 64, 64,    "QK-128x64"},
    {128, 64, 128,   "QK-128x64x128"},
    {64, 128, 64,    "V-64x128"},
    {64, 128, 128,   "V-64x128x128"},
    {32, 64, 64,     "small-attn"},
    {256, 64, 64,    "large-attn"},
    {512, 64, 64,    "xl-attn"},

    // === Conv-like ===
    {28*28, 32, 3*3*3,   "conv-28x32"},   // MNIST first layer
    {14*14, 64, 3*3*32,  "conv-14x64"},  // MNIST second
    {56*56, 64, 3*3*3,   "conv-56x64"},  // ResNet stem
    {28*28, 128, 3*3*64,"conv-28x128"}, // ResNet stage 2
    {14*14, 256, 3*3*128,"conv-14x256"},// ResNet stage 3
    {7*7, 512, 3*3*256,  "conv-7x512"}, // ResNet stage 4

    // === K=1 edge cases (scaling) ===
    {1, 1, 1,        "K1-1x1"},
    {4, 4, 1,        "K1-4x4"},
    {8, 8, 1,        "K1-8x8"},
    {16, 16, 1,      "K1-16x16"},
    {32, 32, 1,      "K1-32x32"},
    {64, 64, 1,      "K1-64x64"},
    {128, 128, 1,    "K1-128x128"},
    {256, 256, 1,    "K1-256x256"},

    // === N=1 edge cases (dot product) ===
    {1, 1, 4,        "N1-dot-4"},
    {1, 1, 8,        "N1-dot-8"},
    {1, 1, 16,       "N1-dot-16"},
    {1, 1, 32,       "N1-dot-32"},
    {1, 1, 64,       "N1-dot-64"},
    {1, 1, 128,      "N1-dot-128"},
    {4, 1, 4,        "N1-4x1x4"},
    {8, 1, 8,        "N1-8x1x8"},
    {16, 1, 16,      "N1-16x1x16"},
    {32, 1, 32,      "N1-32x1x32"},
    {64, 1, 64,      "N1-64x1x64"},

    // === M=1 edge cases (row vector) ===
    {1, 4, 4,        "M1-1x4x4"},
    {1, 8, 8,        "M1-1x8x8"},
    {1, 16, 16,      "M1-1x16x16"},
    {1, 32, 32,      "M1-1x32x32"},
    {1, 64, 64,      "M1-1x64x64"},
    {1, 128, 128,    "M1-1x128x128"},

    // === Primes (stress cache blocking) ===
    {17, 17, 17,     "prime-17"},
    {23, 23, 23,     "prime-23"},
    {29, 29, 29,     "prime-29"},
    {37, 37, 37,     "prime-37"},
    {41, 41, 41,     "prime-41"},
    {43, 43, 43,     "prime-43"},
    {47, 47, 47,     "prime-47"},
    {53, 53, 53,     "prime-53"},
    {59, 59, 59,     "prime-59"},
    {61, 61, 61,     "prime-61"},
    {67, 67, 67,     "prime-67"},
    {71, 71, 71,     "prime-71"},
    {73, 73, 73,     "prime-73"},
    {79, 79, 79,     "prime-79"},
    {83, 83, 83,     "prime-83"},
    {89, 89, 89,     "prime-89"},
    {97, 97, 97,     "prime-97"},

    // === Not square, not divisible by 8 ===
    {3, 5, 7,        "3x5x7"},
    {5, 3, 7,        "5x3x7"},
    {7, 5, 3,        "7x5x3"},
    {11, 13, 17,     "prime-mix-11x13x17"},
    {13, 11, 19,     "prime-mix-13x11x19"},
    {17, 19, 23,     "prime-mix-17x19x23"},
    {19, 17, 23,     "prime-mix-19x17x23"},

    // === Larger irregular ===
    {100, 100, 100,  "100x100x100"},
    {150, 150, 150,  "150x150x150"},
    {200, 200, 200,  "200x200x200"},
    {300, 300, 300,  "300x300x300"},
    {400, 400, 400,  "400x400x400"},
    {600, 600, 600,  "600x600x600"},
    {700, 700, 700,  "700x700x700"},
    {800, 800, 800,  "800x800x800"},
    {900, 900, 900,  "900x900x900"},
    {1000, 1000, 1000, "1000x1000x1000"},
};

}  // namespace

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  Small & Irregular GEMM Benchmark Suite\n");
    printf("  Testing autoGEMM edge case handling\n");
    printf("================================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);
    printf("FP32 peak: %.2f GFLOPS/core\n\n", hw.fp32_gflops_per_core);

    int warmup = 2;
    int runs = 5;
    if (argc > 1) runs = atoi(argv[1]);

    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);
    int passed = 0, failed = 0;
    double total_gflops = 0.0;
    double min_gflops = 1e9, max_gflops = 0.0;

    printf("%-20s %6s %6s %6s %10s %8s %s\n",
           "Shape", "M", "N", "K", "GFLOPS", "ms", "status");
    printf("%-20s %6s %6s %6s %10s %8s %s\n",
           "-----", "-", "-", "-", "------", "--", "------");

    for (int s = 0; s < num_shapes; ++s) {
        int M = shapes[s].M;
        int N = shapes[s].N;
        int K = shapes[s].K;

        size_t a_sz = (size_t)M * K;
        size_t b_sz = (size_t)K * N;
        size_t c_sz = (size_t)M * N;

        auto A = dnnopt::aligned_array<float>(a_sz);
        auto B = dnnopt::aligned_array<float>(b_sz);
        auto C = dnnopt::aligned_array<float>(c_sz);

        fill_random(A.get(), a_sz);
        fill_random(B.get(), b_sz);
        memset(C.get(), 0, c_sz * sizeof(float));

        double flops = shapes[s].flops();
        double bytes = (a_sz + b_sz + c_sz) * sizeof(float);

        // Benchmark dnnopt
        auto stats = dnnopt::benchmark(shapes[s].label, flops, bytes, warmup, runs, [&]() {
            memset(C.get(), 0, c_sz * sizeof(float));
            dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        });

        // Verify correctness
        bool ok = verify_correctness(M, N, K, A.get(), K, B.get(), N, C.get(), N);

        if (ok) {
            passed++;
            printf("%-20s %6d %6d %6d %9.2f %7.3f OK\n",
                   shapes[s].label, M, N, K, stats.gflops, stats.median_ms);
        } else {
            failed++;
            printf("%-20s %6d %6d %6d %9s %7s FAIL\n",
                   shapes[s].label, M, N, K, "N/A", "N/A");
        }

        total_gflops += stats.gflops;
        if (stats.gflops > 0 && stats.gflops < min_gflops) min_gflops = stats.gflops;
        if (stats.gflops > max_gflops) max_gflops = stats.gflops;
    }

    printf("\n================================================================\n");
    printf("  Summary\n");
    printf("================================================================\n");
    printf("Total shapes:    %d\n", num_shapes);
    printf("Passed:          %d (%.1f%%)\n", passed, 100.0 * passed / num_shapes);
    printf("Failed:          %d (%.1f%%)\n", failed, 100.0 * failed / num_shapes);
    printf("Avg GFLOPS:      %.2f\n", total_gflops / num_shapes);
    printf("Min GFLOPS:      %.2f\n", min_gflops);
    printf("Max GFLOPS:      %.2f\n", max_gflops);

    if (hw.fp32_gflops_per_core > 0) {
        printf("Peak utilization: %.1f%%\n", 100.0 * max_gflops / hw.fp32_gflops_per_core);
    }

    return failed > 0 ? 1 : 0;
}