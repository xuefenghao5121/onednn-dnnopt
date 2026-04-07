/// @file bench_conv.cpp
/// Convolution benchmark suite.
/// Tests naive, im2col+naive-GEMM, and dnnopt::conv2d_fp32 (optimized).

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/conv/conv.h"
#include "dnnopt/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ============================================================
// Conv2D parameters
// ============================================================
struct Conv2DShape {
    int N, IC, IH, IW;
    int OC, KH, KW;
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
                       const float* input, const float* filter,
                       const float* bias, float* output) {
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
                            if (ih < 0 || ih >= s.IH || iw < 0 || iw >= s.IW) continue;
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
void im2col_nhwc_naive(const Conv2DShape& s, const float* input, float* col) {
    int OH = s.OH(), OW = s.OW();
    int col_w = s.IC * s.KH * s.KW;
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

dnnopt::Conv2DParams to_lib_params(const Conv2DShape& s) {
    dnnopt::Conv2DParams p;
    p.N = s.N; p.IC = s.IC; p.IH = s.IH; p.IW = s.IW;
    p.OC = s.OC; p.KH = s.KH; p.KW = s.KW;
    p.stride_h = s.stride; p.stride_w = s.stride;
    p.pad_h = s.pad; p.pad_w = s.pad;
    return p;
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

        // --- Naive conv (skip very large shapes) ---
        if (flops < 1e10) {
            char name[128];
            snprintf(name, sizeof(name), "%s [%dx%dx%d->%d] naive",
                     shape.label, shape.IC, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                conv2d_naive_nhwc(shape, input.get(), filter.get(), nullptr, output.get());
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // --- im2col + naive GEMM ---
        {
            size_t col_rows = (size_t)shape.N * OH * OW;
            size_t col_cols = (size_t)shape.IC * shape.KH * shape.KW;
            size_t col_size = col_rows * col_cols;

            if (col_size * sizeof(float) < 512ULL * 1024 * 1024) {
                auto col_buf = dnnopt::aligned_array<float>(col_size);

                char name[128];
                snprintf(name, sizeof(name), "%s [%dx%dx%d->%d] im2col+naive",
                         shape.label, shape.IC, shape.IH, shape.IW, shape.OC);
                auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                    im2col_nhwc_naive(shape, input.get(), col_buf.get());
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

        // --- dnnopt::conv2d_fp32 (optimized im2col + GEMM) ---
        {
            auto lp = to_lib_params(shape);
            char name[128];
            snprintf(name, sizeof(name), "%s [%dx%dx%d->%d] dnnopt",
                     shape.label, shape.IC, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                dnnopt::conv2d_fp32(lp, input.get(), filter.get(), nullptr,
                                    output.get(), dnnopt::ConvPostOp::kNone);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        printf("\n");
    }

    dnnopt::write_csv("bench_conv_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}
