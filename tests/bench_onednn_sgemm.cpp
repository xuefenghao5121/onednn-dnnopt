/// @file bench_onednn_sgemm.cpp
/// Standalone oneDNN sgemm benchmark for small/irregular GEMM shapes.
///
/// Usage:
///   # With upstream oneDNN:
///   LD_LIBRARY_PATH=/root/onednn-upstream/build/src ./bench_onednn_sgemm
///   # With dnnopt-patched oneDNN:
///   LD_LIBRARY_PATH=/root/onednn-dnnopt/build/src ./bench_onednn_sgemm
///
/// Outputs TSV for easy comparison.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <dnnl.h>

namespace {

struct Shape {
    int M, N, K;
    const char* label;
};

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

double bench_sgemm(int M, int N, int K,
                   const float* A, const float* B, float* C,
                   int warmup = 3, int iters = 15) {
    float alpha = 1.0f, beta = 0.0f;
    for (int w = 0; w < warmup; w++)
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];  // median microseconds
}

}  // namespace

int main() {
    srand(42);

    Shape shapes[] = {
        // Small M (GEMV-like)
        { 1,  128,  128, "M1_N128_K128"},
        { 1,  256,  512, "M1_N256_K512"},
        { 1, 1024, 1024, "M1_N1024_K1024"},
        { 1, 4096, 4096, "M1_N4096_K4096"},

        // Small M, moderate K
        { 2,   64,   64, "M2_N64_K64"},
        { 3,   64,   64, "M3_N64_K64"},
        { 4,   64,   64, "M4_N64_K64"},
        { 5,   64,   64, "M5_N64_K64"},
        { 6,   64,   64, "M6_N64_K64"},
        { 7,   64,   64, "M7_N64_K64"},
        { 8,   64,   64, "M8_N64_K64"},

        // BERT-like
        { 4,  768,  768, "M4_N768_K768"},
        { 6,  768,  768, "M6_N768_K768"},
        { 4, 3072,  768, "M4_N3072_K768"},
        { 4,  768, 3072, "M4_N768_K3072"},

        // LLM-like
        { 4, 4096, 4096, "M4_N4096_K4096"},
        { 6, 4096, 4096, "M6_N4096_K4096"},

        // Prime/irregular N
        { 8,   17,   64, "M8_N17_K64"},
        { 8,   37,   64, "M8_N37_K64"},
        { 8,   53,  128, "M8_N53_K128"},
        { 8,   97,  128, "M8_N97_K128"},
        {16,   23,  128, "M16_N23_K128"},
        {16,   47,  128, "M16_N47_K128"},

        // Non-power-of-2 boundaries
        { 3,   48,   48, "M3_N48_K48"},
        { 3,   49,   49, "M3_N49_K49"},
        { 3,   50,   50, "M3_N50_K50"},
        { 5,   31,   31, "M5_N31_K31"},
        { 5,   32,   32, "M5_N32_K32"},
        { 5,   33,   33, "M5_N33_K33"},
        { 7,   63,   63, "M7_N63_K63"},
        { 7,   64,   64, "M7_N64_K64"},
        { 7,   65,   65, "M7_N65_K65"},

        // Tiny
        { 3,    8,    8, "M3_N8_K8"},
        { 3,   16,   16, "M3_N16_K16"},
        { 5,   12,   12, "M5_N12_K12"},
        { 5,   16,   16, "M5_N16_K16"},
        { 7,   16,   16, "M7_N16_K16"},

        // Medium M, irregular N
        {16,   33,  256, "M16_N33_K256"},
        {16,   65,  256, "M16_N65_K256"},
        {32,   17,  256, "M32_N17_K256"},
        {32,   47,  256, "M32_N47_K256"},

        // Tall-skinny (N << M)
        {128,   1,  128, "M128_N1_K128"},
        {128,   2,  128, "M128_N2_K128"},
        {128,   3,  128, "M128_N3_K128"},
        {128,   4,  128, "M128_N4_K128"},
        {128,   7,  128, "M128_N7_K128"},

        // Short-wide (M << N)
        {  2, 128,  128, "M2_N128_K128"},
        {  3, 128,  128, "M3_N128_K128"},
        {  4, 128,  128, "M4_N128_K128"},
        {  7, 128,  128, "M7_N128_K128"},

        // Inference shapes
        {  1, 1024, 4096, "M1_N1024_K4096"},
        {  4, 1024, 4096, "M4_N1024_K4096"},
        {  8, 1024, 4096, "M8_N1024_K4096"},
        { 16, 1024, 4096, "M16_N1024_K4096"},

        // Batch inference
        { 4,  128, 4096, "batch4_M128"},
        { 4, 2048, 4096, "batch4_M2048"},
    };

    int n = sizeof(shapes) / sizeof(shapes[0]);

    // TSV header
    printf("shape\tM\tN\tK\ttime_us\tgflops\n");

    for (int i = 0; i < n; i++) {
        const auto& s = shapes[i];
        size_t a_sz = (size_t)s.M * s.K;
        size_t b_sz = (size_t)s.K * s.N;
        size_t c_sz = (size_t)s.M * s.N;

        float* A = (float*)aligned_alloc(64, a_sz * sizeof(float));
        float* B = (float*)aligned_alloc(64, b_sz * sizeof(float));
        float* C = (float*)aligned_alloc(64, c_sz * sizeof(float));

        fill_random(A, a_sz);
        fill_random(B, b_sz);

        double us = bench_sgemm(s.M, s.N, s.K, A, B, C);
        double gflops = 2.0 * (double)s.M * (double)s.N * (double)s.K / (us * 1e3);

        printf("%s\t%d\t%d\t%d\t%.1f\t%.2f\n", s.label, s.M, s.N, s.K, us, gflops);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
