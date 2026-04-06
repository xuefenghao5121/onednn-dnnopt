/// @file bench_elementwise.cpp
/// Elementwise / activation function benchmark suite.

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace {

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

// ============================================================
// Scalar implementations
// ============================================================
void relu_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] > 0 ? in[i] : 0;
}

void sigmoid_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

void gelu_tanh_scalar(const float* in, float* out, size_t n) {
    const float c = 0.044715f;
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/pi)
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        float t = tanhf(sqrt_2_pi * (x + c * x * x * x));
        out[i] = 0.5f * x * (1.0f + t);
    }
}

void silu_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

// ============================================================
// NEON implementations
// ============================================================
#ifdef __ARM_NEON
void relu_neon(const float* in, float* out, size_t n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t v0 = vld1q_f32(in + i);
        float32x4_t v1 = vld1q_f32(in + i + 4);
        float32x4_t v2 = vld1q_f32(in + i + 8);
        float32x4_t v3 = vld1q_f32(in + i + 12);
        vst1q_f32(out + i,      vmaxq_f32(v0, zero));
        vst1q_f32(out + i + 4,  vmaxq_f32(v1, zero));
        vst1q_f32(out + i + 8,  vmaxq_f32(v2, zero));
        vst1q_f32(out + i + 12, vmaxq_f32(v3, zero));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
    for (; i < n; ++i) out[i] = in[i] > 0 ? in[i] : 0;
}
#endif

// ============================================================
// Shapes to benchmark
// ============================================================
struct EltShape {
    size_t n;
    const char* label;
};

const EltShape shapes[] = {
    {64 * 112 * 112,    "ResNet-ReLU-64x112x112"},
    {256 * 56 * 56,     "ResNet-ReLU-256x56x56"},
    {512 * 28 * 28,     "ResNet-ReLU-512x28x28"},
    {2048 * 7 * 7,      "ResNet-ReLU-2048x7x7"},
    {128 * 768,         "BERT-Act-128x768"},
    {128 * 3072,        "BERT-Act-128x3072"},
    {1024 * 768,        "GPT2-Act-1024x768"},
    {1024 * 3072,       "GPT2-Act-1024x3072"},
};

}  // namespace

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  Elementwise / Activation Benchmark Suite\n");
    printf("==========================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz\n\n", hw.cpu_name.c_str(), hw.freq_mhz);

    int warmup = 5;
    int runs   = 20;
    if (argc > 1) runs = atoi(argv[1]);

    std::vector<dnnopt::BenchStats> all_results;

    for (const auto& shape : shapes) {
        size_t n = shape.n;
        auto input  = dnnopt::aligned_array<float>(n);
        auto output = dnnopt::aligned_array<float>(n);
        fill_random(input.get(), n);

        double bytes = n * sizeof(float) * 2;  // read + write

        // ReLU scalar
        {
            char name[128];
            snprintf(name, sizeof(name), "%s relu-scalar", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                relu_scalar(input.get(), output.get(), n);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

#ifdef __ARM_NEON
        // ReLU NEON
        {
            char name[128];
            snprintf(name, sizeof(name), "%s relu-neon", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                relu_neon(input.get(), output.get(), n);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }
#endif

        // Sigmoid scalar
        {
            char name[128];
            snprintf(name, sizeof(name), "%s sigmoid-scalar", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                sigmoid_scalar(input.get(), output.get(), n);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // GELU tanh scalar
        {
            char name[128];
            snprintf(name, sizeof(name), "%s gelu-scalar", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                gelu_tanh_scalar(input.get(), output.get(), n);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        printf("\n");
    }

    dnnopt::write_csv("bench_elementwise_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}
