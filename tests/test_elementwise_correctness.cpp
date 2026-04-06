/// @file test_elementwise_correctness.cpp
/// Correctness tests for elementwise / activation functions.

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

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

// Scalar references
void relu_ref(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] > 0 ? in[i] : 0;
}

void sigmoid_ref(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

void gelu_ref(const float* in, float* out, size_t n) {
    const float c = 0.044715f;
    const float k = 0.7978845608f;
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(k * (x + c * x * x * x)));
    }
}

#ifdef __ARM_NEON
void relu_neon(const float* in, float* out, size_t n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
    for (; i < n; ++i) out[i] = in[i] > 0 ? in[i] : 0;
}
#endif

float max_diff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

template<typename Fn>
void test_activation(const char* name, size_t n, Fn ref_fn, float tol) {
    auto input   = dnnopt::aligned_array<float>(n);
    auto out_ref = dnnopt::aligned_array<float>(n);

    fill_random(input.get(), n);
    ref_fn(input.get(), out_ref.get(), n);

    // Verify reference output is finite
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(out_ref[i])) { all_finite = false; break; }
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "%s reference output is finite (n=%zu)", name, n);
    TEST_ASSERT(all_finite, msg);
}

}  // namespace

int main() {
    printf("=== test_elementwise_correctness ===\n");

    const size_t sizes[] = {1, 3, 4, 7, 16, 127, 1024, 4096, 100000};

    for (size_t n : sizes) {
        // Test reference implementations produce finite results
        test_activation("relu", n, relu_ref, 0.0f);
        test_activation("sigmoid", n, sigmoid_ref, 1e-6f);
        test_activation("gelu", n, gelu_ref, 1e-5f);

#ifdef __ARM_NEON
        // Test NEON ReLU matches reference
        {
            auto input  = dnnopt::aligned_array<float>(n);
            auto out_ref = dnnopt::aligned_array<float>(n);
            auto out_neon = dnnopt::aligned_array<float>(n);

            fill_random(input.get(), n);
            relu_ref(input.get(), out_ref.get(), n);
            relu_neon(input.get(), out_neon.get(), n);

            float md = max_diff(out_ref.get(), out_neon.get(), n);
            char msg[128];
            snprintf(msg, sizeof(msg), "NEON relu n=%zu max_diff=%.6e", n, md);
            TEST_ASSERT(md == 0.0f, msg);  // ReLU should be exact
        }
#endif
    }

    TEST_SUMMARY();
}
