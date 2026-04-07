/// @file bench_blas.cpp
/// CBLAS sgemm benchmark: dnnopt vs OpenBLAS.
/// Links against system OpenBLAS at runtime for comparison.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
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

const GemmShape shapes[] = {
    {64,   64,   64,   "tiny-64"},
    {128,  128,  128,  "small-128"},
    {256,  256,  256,  "medium-256"},
    {512,  512,  512,  "large-512"},
    {1024, 1024, 1024, "xlarge-1024"},
    {2048, 2048, 2048, "xxlarge-2048"},
    // Non-square shapes common in DL
    {1,    1024, 1024, "batch1-1024"},
    {32,   4096, 1024, "bert-proj"},
    {128,  512,  2048, "fc-layer"},
    {3136, 64,   576,  "conv-like"},    // N*OH*OW × OC × IC*KH*KW
};

// OpenBLAS function pointer type
typedef void (*openblas_sgemm_t)(int, int, int, int, int, int,
                                 float, const float*, int,
                                 const float*, int,
                                 float, float*, int);

}  // namespace

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  CBLAS sgemm Benchmark: dnnopt vs OpenBLAS\n");
    printf("==========================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);

    int warmup = 2;
    int runs = 5;
    if (argc > 1) runs = atoi(argv[1]);

    // Try to load OpenBLAS for comparison
    void* openblas_handle = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
    openblas_sgemm_t openblas_sgemm = nullptr;
    if (openblas_handle) {
        openblas_sgemm = (openblas_sgemm_t)dlsym(openblas_handle, "cblas_sgemm");
        if (openblas_sgemm) {
            printf("OpenBLAS: loaded for comparison\n\n");
        } else {
            printf("OpenBLAS: loaded but cblas_sgemm not found\n\n");
        }
    } else {
        printf("OpenBLAS: not found (skipping comparison)\n\n");
    }

    std::vector<dnnopt::BenchStats> all_results;

    printf("%-25s %12s %12s %12s\n", "Shape", "dnnopt(ms)", "OpenBLAS(ms)", "Speedup");
    printf("%-25s %12s %12s %12s\n", "-----", "----------", "------------", "-------");

    for (const auto& shape : shapes) {
        size_t a_sz = (size_t)shape.M * shape.K;
        size_t b_sz = (size_t)shape.K * shape.N;
        size_t c_sz = (size_t)shape.M * shape.N;

        auto A = dnnopt::aligned_array<float>(a_sz);
        auto B = dnnopt::aligned_array<float>(b_sz);
        auto C = dnnopt::aligned_array<float>(c_sz);

        fill_random(A.get(), a_sz);
        fill_random(B.get(), b_sz);

        double flops = shape.flops();
        double bytes = (a_sz + b_sz + c_sz) * sizeof(float);

        // Benchmark dnnopt cblas_sgemm
        char dnnopt_name[128];
        snprintf(dnnopt_name, sizeof(dnnopt_name), "%s dnnopt", shape.label);
        auto dnnopt_stats = dnnopt::benchmark(dnnopt_name, flops, bytes, warmup, runs, [&]() {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        shape.M, shape.N, shape.K,
                        1.0f, A.get(), shape.K, B.get(), shape.N,
                        0.0f, C.get(), shape.N);
        });
        all_results.push_back(dnnopt_stats);

        // Benchmark OpenBLAS (if available)
        double openblas_median = 0.0;
        if (openblas_sgemm) {
            char ob_name[128];
            snprintf(ob_name, sizeof(ob_name), "%s OpenBLAS", shape.label);
            auto ob_stats = dnnopt::benchmark(ob_name, flops, bytes, warmup, runs, [&]() {
                openblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                               shape.M, shape.N, shape.K,
                               1.0f, A.get(), shape.K, B.get(), shape.N,
                               0.0f, C.get(), shape.N);
            });
            all_results.push_back(ob_stats);
            openblas_median = ob_stats.median_ms;
        }

        // Print comparison
        if (openblas_sgemm && openblas_median > 0) {
            double speedup = openblas_median / dnnopt_stats.median_ms;
            printf("%-25s %10.3f   %10.3f   %10.2fx\n",
                   shape.label, dnnopt_stats.median_ms, openblas_median, speedup);
        } else {
            printf("%-25s %10.3f   %12s   %12s  %.1f GFLOPS\n",
                   shape.label, dnnopt_stats.median_ms, "N/A", "N/A", dnnopt_stats.gflops);
        }
    }

    // Also test ColMajor + Trans to verify overhead
    printf("\n--- ColMajor + Trans overhead ---\n");
    {
        int M = 512, N = 512, K = 512;
        auto A = dnnopt::aligned_array<float>((size_t)M * K);
        auto B = dnnopt::aligned_array<float>((size_t)K * N);
        auto C = dnnopt::aligned_array<float>((size_t)M * N);
        fill_random(A.get(), (size_t)M * K);
        fill_random(B.get(), (size_t)K * N);
        double flops = 2.0 * M * N * K;
        double bytes = ((size_t)M * K + (size_t)K * N + (size_t)M * N) * sizeof(float);

        auto s_nn = dnnopt::benchmark("Row+NN 512", flops, bytes, warmup, runs, [&]() {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        });
        auto s_tn = dnnopt::benchmark("Row+TN 512", flops, bytes, warmup, runs, [&]() {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.get(), M, B.get(), N, 0.0f, C.get(), N);
        });
        auto s_col = dnnopt::benchmark("Col+NN 512", flops, bytes, warmup, runs, [&]() {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.get(), M, B.get(), K, 0.0f, C.get(), M);
        });
        printf("  Row+NN: %.3f ms (%.1f GFLOPS)\n", s_nn.median_ms, s_nn.gflops);
        printf("  Row+TN: %.3f ms (%.1f GFLOPS) — transpose overhead: %.1f%%\n",
               s_tn.median_ms, s_tn.gflops,
               (s_tn.median_ms / s_nn.median_ms - 1.0) * 100.0);
        printf("  Col+NN: %.3f ms (%.1f GFLOPS) — col-major overhead: %.1f%%\n",
               s_col.median_ms, s_col.gflops,
               (s_col.median_ms / s_nn.median_ms - 1.0) * 100.0);
    }

    dnnopt::write_csv("bench_blas_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());

    if (openblas_handle) dlclose(openblas_handle);
    return 0;
}
