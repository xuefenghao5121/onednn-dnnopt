/// @file test_gemm_correctness.cpp
/// Correctness tests for GEMM implementations.
/// Compares optimized implementations against naive reference.

#include "dnnopt/aligned_alloc.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace {

void fill_random(float* data, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Reference GEMM: C = A * B (row-major)
void gemm_ref(int M, int N, int K,
              const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

#ifdef __ARM_NEON
/// Simple NEON GEMM (same as bench_gemm.cpp)
void gemm_neon(int M, int N, int K,
               const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        int j = 0;
        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0);
            for (int k = 0; k < K; ++k) {
                float32x4_t a_val = vdupq_n_f32(A[i * K + k]);
                float32x4_t b0 = vld1q_f32(&B[k * N + j]);
                c0 = vfmaq_f32(c0, a_val, b0);
            }
            vst1q_f32(&C[i * N + j], c0);
        }
        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}
#endif

/// Compare two matrices element-wise.
/// Returns max absolute difference.
float max_diff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

/// Compare with relative tolerance based on K (accumulation length).
void test_gemm_shape(int M, int N, int K, const char* label) {
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;

    auto A    = dnnopt::aligned_array<float>(a_sz);
    auto B    = dnnopt::aligned_array<float>(b_sz);
    auto C_ref = dnnopt::aligned_array<float>(c_sz);

    fill_random(A.get(), a_sz);
    fill_random(B.get(), b_sz);

    // Reference
    gemm_ref(M, N, K, A.get(), B.get(), C_ref.get());

    // Tolerance: FP32 accumulation of K products
    // Each product has ~1e-7 relative error, accumulated K times
    float tol = K * 2e-5f;

#ifdef __ARM_NEON
    {
        auto C_neon = dnnopt::aligned_array<float>(c_sz);
        gemm_neon(M, N, K, A.get(), B.get(), C_neon.get());
        float md = max_diff(C_ref.get(), C_neon.get(), c_sz);

        char msg[128];
        snprintf(msg, sizeof(msg), "NEON GEMM %s [%dx%dx%d] max_diff=%.6e tol=%.6e",
                 label, M, N, K, md, tol);
        TEST_ASSERT(md < tol, msg);
    }
#endif
}

}  // namespace

int main() {
    printf("=== test_gemm_correctness ===\n");

    // Square shapes
    test_gemm_shape(1, 1, 1, "1x1x1");
    test_gemm_shape(4, 4, 4, "4x4x4");
    test_gemm_shape(8, 8, 8, "8x8x8");
    test_gemm_shape(16, 16, 16, "16x16x16");
    test_gemm_shape(32, 32, 32, "32x32x32");
    test_gemm_shape(64, 64, 64, "64x64x64");
    test_gemm_shape(128, 128, 128, "128x128x128");
    test_gemm_shape(256, 256, 256, "256x256x256");

    // Non-square / edge cases
    test_gemm_shape(1, 128, 64, "1x128x64");
    test_gemm_shape(7, 13, 5, "7x13x5");     // Prime dimensions
    test_gemm_shape(3, 3, 1000, "3x3x1000");  // Long K
    test_gemm_shape(128, 1, 768, "128x1x768"); // Column vector output
    test_gemm_shape(1, 1000, 2048, "FC-like");

    // BERT shapes
    test_gemm_shape(128, 768, 768, "BERT-QKV");
    test_gemm_shape(128, 3072, 768, "BERT-FFN1");

    // Non-aligned (not multiple of 4)
    test_gemm_shape(5, 7, 11, "5x7x11");
    test_gemm_shape(13, 17, 19, "13x17x19");
    test_gemm_shape(127, 127, 127, "127x127x127");

    TEST_SUMMARY();
}
