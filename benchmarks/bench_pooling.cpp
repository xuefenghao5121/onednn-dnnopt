/// @file bench_pooling.cpp
/// Pooling benchmark suite (MaxPool, AvgPool, GlobalAvgPool).

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/timer.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace {

struct PoolShape {
    int N, C, IH, IW;
    int KH, KW, stride, pad;
    const char* label;

    int OH() const { return (IH + 2 * pad - KH) / stride + 1; }
    int OW() const { return (IW + 2 * pad - KW) / stride + 1; }
};

const PoolShape pool_shapes[] = {
    {1,   64, 112, 112, 3, 3, 2, 1, "Pool-64x112-k3s2"},
    {1,  256,  56,  56, 3, 3, 2, 1, "Pool-256x56-k3s2"},
    {1,  512,  28,  28, 3, 3, 2, 1, "Pool-512x28-k3s2"},
    {1, 2048,   7,   7, 7, 7, 1, 0, "GAvgPool-2048x7"},
    {1,  768, 128,   1, 128, 1, 1, 0, "GAvgPool-BERT-768"},
};

// Naive NHWC MaxPool
void maxpool_nhwc(const PoolShape& s, const float* in, float* out) {
    int OH = s.OH(), OW = s.OW();
    for (int n = 0; n < s.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int c = 0; c < s.C; ++c) {
                    float mx = -FLT_MAX;
                    for (int kh = 0; kh < s.KH; ++kh) {
                        for (int kw = 0; kw < s.KW; ++kw) {
                            int ih = oh * s.stride - s.pad + kh;
                            int iw = ow * s.stride - s.pad + kw;
                            if (ih >= 0 && ih < s.IH && iw >= 0 && iw < s.IW) {
                                float val = in[((n * s.IH + ih) * s.IW + iw) * s.C + c];
                                if (val > mx) mx = val;
                            }
                        }
                    }
                    out[((n * OH + oh) * OW + ow) * s.C + c] = mx;
                }
            }
        }
    }
}

// Naive NHWC AvgPool
void avgpool_nhwc(const PoolShape& s, const float* in, float* out) {
    int OH = s.OH(), OW = s.OW();
    for (int n = 0; n < s.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int c = 0; c < s.C; ++c) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < s.KH; ++kh) {
                        for (int kw = 0; kw < s.KW; ++kw) {
                            int ih = oh * s.stride - s.pad + kh;
                            int iw = ow * s.stride - s.pad + kw;
                            if (ih >= 0 && ih < s.IH && iw >= 0 && iw < s.IW) {
                                sum += in[((n * s.IH + ih) * s.IW + iw) * s.C + c];
                                count++;
                            }
                        }
                    }
                    out[((n * OH + oh) * OW + ow) * s.C + c] = sum / count;
                }
            }
        }
    }
}

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

}  // namespace

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  Pooling Benchmark Suite (NHWC)\n");
    printf("==========================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz\n\n", hw.cpu_name.c_str(), hw.freq_mhz);

    int warmup = 3;
    int runs   = 10;
    if (argc > 1) runs = atoi(argv[1]);

    std::vector<dnnopt::BenchStats> all_results;

    for (const auto& shape : pool_shapes) {
        int OH = shape.OH(), OW = shape.OW();
        size_t in_size  = (size_t)shape.N * shape.IH * shape.IW * shape.C;
        size_t out_size = (size_t)shape.N * OH * OW * shape.C;

        auto input  = dnnopt::aligned_array<float>(in_size);
        auto output = dnnopt::aligned_array<float>(out_size);
        fill_random(input.get(), in_size);

        double bytes = (in_size + out_size) * sizeof(float);

        // MaxPool
        {
            char name[128];
            snprintf(name, sizeof(name), "%s maxpool", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                maxpool_nhwc(shape, input.get(), output.get());
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // AvgPool
        {
            char name[128];
            snprintf(name, sizeof(name), "%s avgpool", shape.label);
            auto stats = dnnopt::benchmark(name, 0, bytes, warmup, runs, [&]() {
                avgpool_nhwc(shape, input.get(), output.get());
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        printf("\n");
    }

    dnnopt::write_csv("bench_pooling_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}
