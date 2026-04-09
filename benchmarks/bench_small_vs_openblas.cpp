/// @file bench_small_vs_openblas.cpp
/// Compare dnnopt vs OpenBLAS on small and irregular GEMM shapes.
/// Focus on shapes where autoGEMM should show advantage.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/timer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

namespace {

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

struct GemmShape {
    int M, N, K;
    const char* label;
    double flops() const { return 2.0 * M * N * K; }
};

// Shapes where dnnopt should have advantage (small M, irregular, DL-specific)
const GemmShape shapes[] = {
    // === Small batch inference (M=1,2,4,8) ===
    {1, 1000, 2048,   "batch1-FC"},
    {2, 1000, 2048,   "batch2-FC"},
    {4, 1000, 2048,   "batch4-FC"},
    {8, 1000, 2048,   "batch8-FC"},
    {16, 1000, 2048,  "batch16-FC"},
    {1, 4096, 4096,   "batch1-LLM"},
    {2, 4096, 4096,   "batch2-LLM"},
    {4, 4096, 4096,   "batch4-LLM"},

    // === Attention-like ===
    {128, 64, 64,     "attn-128x64"},
    {256, 64, 64,     "attn-256x64"},
    {512, 64, 64,     "attn-512x64"},
    {64, 128, 64,     "attn-64x128"},
    {128, 128, 64,    "attn-128x128"},
    {256, 128, 64,    "attn-256x128"},

    // === Conv-like (im2col GEMM) ===
    {784, 32, 27,     "conv-MNIST-1"},
    {196, 64, 288,    "conv-MNIST-2"},
    {3136, 64, 27,    "conv-ResNet-1"},
    {784, 128, 576,   "conv-ResNet-2"},
    {196, 256, 1152,  "conv-ResNet-3"},
    {49, 512, 2304,   "conv-ResNet-4"},

    // === Tall-skinny (common in DL) ===
    {256, 16, 16,     "tall-256x16"},
    {512, 16, 16,     "tall-512x16"},
    {1024, 16, 16,    "tall-1024x16"},
    {256, 8, 8,       "tall-256x8"},
    {512, 8, 8,       "tall-512x8"},

    // === Wide-short ===
    {16, 256, 16,     "wide-16x256"},
    {16, 512, 16,     "wide-16x512"},
    {8, 256, 8,       "wide-8x256"},
    {8, 512, 8,       "wide-8x512"},

    // === Non-power-of-2 ===
    {3, 64, 64,       "npo2-3x64"},
    {5, 64, 64,       "npo2-5x64"},
    {7, 64, 64,       "npo2-7x64"},
    {11, 64, 64,      "npo2-11x64"},
    {13, 64, 64,      "npo2-13x64"},
    {17, 64, 64,      "npo2-17x64"},
    {23, 64, 64,      "npo2-23x64"},
    {31, 64, 64,      "npo2-31x64"},
    {47, 64, 64,      "npo2-47x64"},
    {63, 64, 64,      "npo2-63x64"},

    // === K-small (limited reduction dimension) ===
    {64, 64, 4,       "Ksmall-64x64x4"},
    {64, 64, 8,       "Ksmall-64x64x8"},
    {64, 64, 12,      "Ksmall-64x64x12"},
    {128, 128, 4,     "Ksmall-128x128x4"},
    {128, 128, 8,     "Ksmall-128x128x8"},

    // === Medium squares ===
    {64, 64, 64,      "square-64"},
    {128, 128, 128,   "square-128"},
    {192, 192, 192,   "square-192"},
    {256, 256, 256,   "square-256"},
    {384, 384, 384,   "square-384"},
    {512, 512, 512,   "square-512"},

    // === Large (for reference) ===
    {1024, 1024, 1024, "large-1024"},
    {2048, 2048, 2048, "large-2048"},
};

typedef void (*cblas_sgemm_fn)(int, int, int, int, int, int,
                                float, const float*, int,
                                const float*, int,
                                float, float*, int);

}  // namespace

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  dnnopt vs OpenBLAS: Small & Irregular GEMM Shapes\n");
    printf("================================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);
    printf("FP32 peak: %.2f GFLOPS/core\n\n", hw.fp32_gflops_per_core);

    int warmup = 2;
    int runs = 5;
    if (argc > 1) runs = atoi(argv[1]);

    // Load OpenBLAS
    void* openblas_handle = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
    cblas_sgemm_fn openblas_sgemm = nullptr;
    if (openblas_handle) {
        openblas_sgemm = (cblas_sgemm_fn)dlsym(openblas_handle, "cblas_sgemm");
    }

    if (openblas_sgemm) {
        printf("OpenBLAS: loaded for comparison\n\n");
    } else {
        printf("OpenBLAS: not found (dnnopt only)\n\n");
    }

    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);
    double dnnopt_total = 0, openblas_total = 0;
    int dnnopt_wins = 0, openblas_wins = 0;
    double best_speedup = 0, worst_speedup = 1e9;
    const char* best_shape = "", *worst_shape = "";

    printf("%-20s %10s %10s %8s %10s\n",
           "Shape", "dnnopt", "OpenBLAS", "Speedup", "dnnopt GF");
    printf("%-20s %10s %10s %8s %10s\n",
           "-----", "------", "--------", "-------", "---------");

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

        double flops = shapes[s].flops();
        double bytes = (a_sz + b_sz + c_sz) * sizeof(float);

        // Benchmark dnnopt
        auto dnnopt_stats = dnnopt::benchmark(shapes[s].label, flops, bytes, warmup, runs, [&]() {
            memset(C.get(), 0, c_sz * sizeof(float));
            dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        });

        // Benchmark OpenBLAS
        double openblas_ms = 0;
        if (openblas_sgemm) {
            auto ob_stats = dnnopt::benchmark(shapes[s].label, flops, bytes, warmup, runs, [&]() {
                memset(C.get(), 0, c_sz * sizeof(float));
                openblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                               M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
            });
            openblas_ms = ob_stats.median_ms;
        }

        if (openblas_sgemm && openblas_ms > 0) {
            double speedup = openblas_ms / dnnopt_stats.median_ms;
            printf("%-20s %9.3f  %9.3f  %7.2fx  %9.2f\n",
                   shapes[s].label, dnnopt_stats.median_ms, openblas_ms, speedup, dnnopt_stats.gflops);

            dnnopt_total += dnnopt_stats.median_ms;
            openblas_total += openblas_ms;

            if (speedup > 1.0) dnnopt_wins++;
            else openblas_wins++;

            if (speedup > best_speedup) {
                best_speedup = speedup;
                best_shape = shapes[s].label;
            }
            if (speedup < worst_speedup) {
                worst_speedup = speedup;
                worst_shape = shapes[s].label;
            }
        } else {
            printf("%-20s %9.3f  %10s  %8s  %9.2f\n",
                   shapes[s].label, dnnopt_stats.median_ms, "N/A", "N/A", dnnopt_stats.gflops);
        }
    }

    printf("\n================================================================\n");
    printf("  Summary\n");
    printf("================================================================\n");
    if (openblas_sgemm) {
        printf("Total time: dnnopt=%.3f ms, OpenBLAS=%.3f ms\n", dnnopt_total, openblas_total);
        printf("Overall speedup: %.2fx\n", openblas_total / dnnopt_total);
        printf("dnnopt wins: %d, OpenBLAS wins: %d\n", dnnopt_wins, openblas_wins);
        printf("Best speedup: %.2fx (%s)\n", best_speedup, best_shape);
        printf("Worst speedup: %.2fx (%s)\n", worst_speedup, worst_shape);
    }

    if (openblas_handle) dlclose(openblas_handle);
    return 0;
}