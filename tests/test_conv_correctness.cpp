/// @file test_conv_correctness.cpp
/// Correctness tests for Conv2D implementations.

#include "dnnopt/aligned_alloc.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace {

struct Conv2DParams {
    int N, IC, IH, IW, OC, KH, KW, stride, pad;
    const char* label;
    int OH() const { return (IH + 2 * pad - KH) / stride + 1; }
    int OW() const { return (IW + 2 * pad - KW) / stride + 1; }
};

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Reference NHWC Conv2D
void conv2d_ref(const Conv2DParams& p,
                const float* input, const float* filter, float* output) {
    int OH = p.OH(), OW = p.OW();
    for (int n = 0; n < p.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int oc = 0; oc < p.OC; ++oc) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < p.KH; ++kh) {
                        for (int kw = 0; kw < p.KW; ++kw) {
                            int ih = oh * p.stride - p.pad + kh;
                            int iw = ow * p.stride - p.pad + kw;
                            if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) continue;
                            for (int ic = 0; ic < p.IC; ++ic) {
                                float in_val  = input[((n * p.IH + ih) * p.IW + iw) * p.IC + ic];
                                float flt_val = filter[((oc * p.KH + kh) * p.KW + kw) * p.IC + ic];
                                acc += in_val * flt_val;
                            }
                        }
                    }
                    output[((n * OH + oh) * OW + ow) * p.OC + oc] = acc;
                }
            }
        }
    }
}

/// im2col + GEMM Conv2D
void conv2d_im2col(const Conv2DParams& p,
                   const float* input, const float* filter, float* output) {
    int OH = p.OH(), OW = p.OW();
    size_t col_rows = (size_t)p.N * OH * OW;
    size_t col_cols = (size_t)p.IC * p.KH * p.KW;

    auto col = dnnopt::aligned_array<float>(col_rows * col_cols);
    memset(col.get(), 0, col_rows * col_cols * sizeof(float));

    // im2col
    for (int n = 0; n < p.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                size_t row = (n * OH + oh) * OW + ow;
                size_t idx = 0;
                for (int kh = 0; kh < p.KH; ++kh) {
                    for (int kw = 0; kw < p.KW; ++kw) {
                        int ih = oh * p.stride - p.pad + kh;
                        int iw = ow * p.stride - p.pad + kw;
                        for (int ic = 0; ic < p.IC; ++ic) {
                            if (ih >= 0 && ih < p.IH && iw >= 0 && iw < p.IW)
                                col[row * col_cols + idx] =
                                    input[((n * p.IH + ih) * p.IW + iw) * p.IC + ic];
                            idx++;
                        }
                    }
                }
            }
        }
    }

    // GEMM: [col_rows × col_cols] × [OC × col_cols]^T → [col_rows × OC]
    for (size_t i = 0; i < col_rows; ++i) {
        for (int oc = 0; oc < p.OC; ++oc) {
            float acc = 0.0f;
            for (size_t k = 0; k < col_cols; ++k) {
                acc += col[i * col_cols + k] * filter[oc * col_cols + k];
            }
            output[i * p.OC + oc] = acc;
        }
    }
}

float max_diff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

void test_conv(const Conv2DParams& p) {
    int OH = p.OH(), OW = p.OW();
    size_t in_sz  = (size_t)p.N * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OH * OW * p.OC;

    auto input  = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_im2col = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz);
    fill_random(filter.get(), flt_sz);

    conv2d_ref(p, input.get(), filter.get(), out_ref.get());
    conv2d_im2col(p, input.get(), filter.get(), out_im2col.get());

    float tol = (float)(p.IC * p.KH * p.KW) * 2e-5f;
    float md = max_diff(out_ref.get(), out_im2col.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s im2col vs ref max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

}  // namespace

int main() {
    printf("=== test_conv_correctness ===\n");

    const Conv2DParams tests[] = {
        {1, 1, 3, 3, 1, 3, 3, 1, 1, "minimal-3x3"},
        {1, 3, 8, 8, 4, 3, 3, 1, 1, "small-3x3"},
        {1, 3, 8, 8, 4, 1, 1, 1, 0, "small-1x1"},
        {1, 16, 16, 16, 32, 3, 3, 1, 1, "medium-3x3"},
        {1, 16, 16, 16, 32, 3, 3, 2, 1, "medium-3x3-s2"},
        {1, 3, 7, 7, 8, 3, 3, 1, 0, "no-pad"},
        {1, 3, 5, 5, 8, 5, 5, 1, 2, "5x5-kernel"},
        {1, 64, 28, 28, 64, 3, 3, 1, 1, "resnet-like"},
        {2, 16, 8, 8, 32, 3, 3, 1, 1, "batch-2"},
        {1, 3, 7, 9, 4, 3, 3, 1, 1, "non-square-input"},
    };

    for (const auto& t : tests) {
        test_conv(t);
    }

    TEST_SUMMARY();
}
