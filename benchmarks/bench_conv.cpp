/// @file bench_conv.cpp
/// Convolution benchmark suite.
/// Tests naive im2col+GEMM and direct convolution across common shapes.

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

// ============================================================
// Conv2D parameters
// ============================================================
struct Conv2DShape {
    int N, IC, IH, IW;       // Input: batch, channels, height, width
    int OC, KH, KW;          // Filter: out_channels, kernel_h, kernel_w
    int stride, pad;
    const char* label;

    int OH() const { return (IH + 2 * pad - KH) / stride + 1; }
    int OW() const { return (IW + 2 * pad - KW) / stride + 1; }
    double flops() const {
        return 2.0 * N * OC * OH() * OW() * IC * KH * KW;
    }
};

const Conv2DShape conv_shapes[] = {
    // ResNet-50 representative layers (NHWC)
    {1,   3,  224, 224,  64, 7, 7, 2, 3, "ResNet-Conv1"},
    {1,  64,   56,  56,  64, 3, 3, 1, 1, "ResNet-3x3-64"},
    {1,  64,   56,  56, 128, 3, 3, 2, 1, "ResNet-3x3-128-s2"},
    {1, 128,   28,  28, 128, 3, 3, 1, 1, "ResNet-3x3-128"},
    {1, 128,   28,  28, 256, 3, 3, 2, 1, "ResNet-3x3-256-s2"},
    {1, 256,   14,  14, 256, 3, 3, 1, 1, "ResNet-3x3-256"},
    {1, 256,   14,  14, 512, 3, 3, 2, 1, "ResNet-3x3-512-s2"},
    {1, 512,    7,   7, 512, 3, 3, 1, 1, "ResNet-3x3-512"},
    // 1x1 convolutions
    {1,  64,   56,  56, 256, 1, 1, 1, 0, "ResNet-1x1-256"},
    {1, 256,   56,  56,  64, 1, 1, 1, 0, "ResNet-1x1-64"},
    {1, 512,    7,   7,2048, 1, 1, 1, 0, "ResNet-1x1-2048"},
    // MobileNet depthwise-like
    {1,  32,  112, 112,  32, 3, 3, 1, 1, "MBNet-DW-32"},
    {1, 128,   56,  56, 128, 3, 3, 1, 1, "MBNet-DW-128"},
};

// ============================================================
// Naive NHWC Conv2D (reference)
// ============================================================
void conv2d_naive_nhwc(const Conv2DShape& s,
                       const float* input,   // [N, IH, IW, IC]
                       const float* filter,  // [OC, KH, KW, IC]
                       const float* bias,    // [OC]
                       float* output) {       // [N, OH, OW, OC]
    int OH = s.OH(), OW = s.OW();
    for (int n = 0; n < s.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int oc = 0; oc < s.OC; ++oc) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int kh = 0; kh < s.KH; ++kh) {
                        for (int kw = 0; kw < s.KW; ++kw) {
                            int ih = oh * s.stride - s.pad + kh;
                            int iw = ow * s.stride - s.pad + kw;
                            if (ih < 0 || ih >= s.IH || iw < 0 || iw >= s.IW)
                                continue;
                            for (int ic = 0; ic < s.IC; ++ic) {
                                float in_val  = input[((n * s.IH + ih) * s.IW + iw) * s.IC + ic];
                                float flt_val = filter[((oc * s.KH + kh) * s.KW + kw) * s.IC + ic];
                                acc += in_val * flt_val;
                            }
                        }
                    }
                    output[((n * OH + oh) * OW + ow) * s.OC + oc] = acc;
                }
            }
        }
    }
}

// ============================================================
// im2col + naive GEMM convolution
// ============================================================
void im2col_nhwc(const Conv2DShape& s, const float* input, float* col) {
    int OH = s.OH(), OW = s.OW();
    int col_w = s.IC * s.KH * s.KW;  // columns
    for (int n = 0; n < s.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int row = (n * OH + oh) * OW + ow;
                int col_idx = 0;
                for (int kh = 0; kh < s.KH; ++kh) {
                    for (int kw = 0; kw < s.KW; ++kw) {
                        int ih = oh * s.stride - s.pad + kh;
                        int iw = ow * s.stride - s.pad + kw;
                        for (int ic = 0; ic < s.IC; ++ic) {
                            if (ih >= 0 && ih < s.IH && iw >= 0 && iw < s.IW) {
                                col[row * col_w + col_idx] =
                                    input[((n * s.IH + ih) * s.IW + iw) * s.IC + ic];
                            } else {
                                col[row * col_w + col_idx] = 0.0f;
                            }
                            col_idx++;
                        }
                    }
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
    printf("  Conv2D Benchmark Suite (NHWC)\n");
    printf("==========================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);

    int warmup = 2;
    int runs   = 5;
    if (argc > 1) runs = atoi(argv[1]);

    std::vector<dnnopt::BenchStats> all_results;

    for (const auto& shape : conv_shapes) {
        int OH = shape.OH(), OW = shape.OW();
        size_t in_size  = (size_t)shape.N * shape.IH * shape.IW * shape.IC;
        size_t flt_size = (size_t)shape.OC * shape.KH * shape.KW * shape.IC;
        size_t out_size = (size_t)shape.N * OH * OW * shape.OC;

        auto input  = dnnopt::aligned_array<float>(in_size);
        auto filter = dnnopt::aligned_array<float>(flt_size);
        auto output = dnnopt::aligned_array<float>(out_size);

        fill_random(input.get(), in_size);
        fill_random(filter.get(), flt_size);

        double flops = shape.flops();
        double bytes = (in_size + flt_size + out_size) * sizeof(float);

        // Naive conv (skip very large shapes)
        if (flops < 1e10) {
            char name[128];
            snprintf(name, sizeof(name), "%s [%dx%dx%d→%d] naive",
                     shape.label, shape.IC, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                conv2d_naive_nhwc(shape, input.get(), filter.get(), nullptr, output.get());
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // im2col version
        {
            size_t col_rows = (size_t)shape.N * OH * OW;
            size_t col_cols = (size_t)shape.IC * shape.KH * shape.KW;
            size_t col_size = col_rows * col_cols;

            // Skip if im2col buffer too large (>512MB)
            if (col_size * sizeof(float) < 512ULL * 1024 * 1024) {
                auto col_buf = dnnopt::aligned_array<float>(col_size);

                char name[128];
                snprintf(name, sizeof(name), "%s [%dx%dx%d→%d] im2col",
                         shape.label, shape.IC, shape.IH, shape.IW, shape.OC);
                auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                    im2col_nhwc(shape, input.get(), col_buf.get());
                    // GEMM: [col_rows × col_cols] × [col_cols × OC] → [col_rows × OC]
                    // (naive GEMM for reference)
                    for (size_t i = 0; i < col_rows; ++i) {
                        for (int oc = 0; oc < shape.OC; ++oc) {
                            float acc = 0.0f;
                            for (size_t k = 0; k < col_cols; ++k) {
                                acc += col_buf[i * col_cols + k] *
                                       filter[oc * col_cols + k];
                            }
                            output[i * shape.OC + oc] = acc;
                        }
                    }
                });
                dnnopt::print_bench_stats(stats);
                all_results.push_back(stats);
            }
        }

        printf("\n");
    }

    dnnopt::write_csv("bench_conv_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}
