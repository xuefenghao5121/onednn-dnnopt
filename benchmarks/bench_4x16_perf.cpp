#include "dnnopt/gemm/gemm.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"
#include <cstdio>
#include <cstring>
#include <random>

static void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

int main() {
    const int M = 4, N = 16, K = 64;
    const int lda = K, ldb = N, ldc = N;

    auto A = dnnopt::aligned_array<float>(M * lda);
    auto B = dnnopt::aligned_array<float>(K * ldb);
    auto C = dnnopt::aligned_array<float>(M * ldc);

    fill_random(A.get(), M * lda);
    fill_random(B.get(), K * ldb);

    const int warmup = 100;
    const int iterations = 10000;

    for (int i = 0; i < warmup; i++) {
        memset(C.get(), 0, M * ldc * sizeof(float));
        dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, C.get(), ldc);
    }

    dnnopt::Timer t;
    t.start();
    for (int i = 0; i < iterations; i++) {
        memset(C.get(), 0, M * ldc * sizeof(float));
        dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, C.get(), ldc);
    }
    t.stop();

    double elapsed_ms = t.elapsed_ms();
    double gflops = 2.0 * M * N * K * iterations / elapsed_ms / 1e6;

    printf("dnnopt 4x16 kernel (M=%d,N=%d,K=%d): %.2f GFLOPS (%.1f%% of 48 GFLOPS peak)\\n",
           M, N, K, gflops, gflops / 48.0 * 100);
    printf("For comparison: autoGEMM assembly achieves 47.08 GFLOPS (98%% peak)\\n");

    return 0;
}
