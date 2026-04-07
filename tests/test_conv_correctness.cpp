/// @file test_conv_correctness.cpp
/// Correctness tests for Conv2D implementations.

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/conv/conv.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace {

struct Conv2DTestParams {
    int N, IC, IH, IW, OC, KH, KW, stride, pad;
    const char* label;
    int OH() const { return (IH + 2 * pad - KH) / stride + 1; }
    int OW() const { return (IW + 2 * pad - KW) / stride + 1; }
};

void fill_random(float* data, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Reference NHWC Conv2D (naive)
void conv2d_ref(const Conv2DTestParams& p,
                const float* input, const float* filter,
                const float* bias, float* output) {
    int OH = p.OH(), OW = p.OW();
    for (int n = 0; n < p.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int oc = 0; oc < p.OC; ++oc) {
                    float acc = bias ? bias[oc] : 0.0f;
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

/// Apply reference post-op in-place
void apply_ref_postop(float* output, size_t n, dnnopt::ConvPostOp op) {
    for (size_t i = 0; i < n; ++i) {
        if (op == dnnopt::ConvPostOp::kRelu || op == dnnopt::ConvPostOp::kBiasRelu) {
            if (output[i] < 0.0f) output[i] = 0.0f;
        } else if (op == dnnopt::ConvPostOp::kRelu6) {
            if (output[i] < 0.0f) output[i] = 0.0f;
            if (output[i] > 6.0f) output[i] = 6.0f;
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

// ============================================================
// Test: old im2col vs ref (keep existing)
// ============================================================
void conv2d_im2col_naive(const Conv2DTestParams& p,
                         const float* input, const float* filter, float* output) {
    int OH = p.OH(), OW = p.OW();
    size_t col_rows = (size_t)p.N * OH * OW;
    size_t col_cols = (size_t)p.IC * p.KH * p.KW;

    auto col = dnnopt::aligned_array<float>(col_rows * col_cols);
    memset(col.get(), 0, col_rows * col_cols * sizeof(float));

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

dnnopt::Conv2DParams to_lib_params(const Conv2DTestParams& p) {
    dnnopt::Conv2DParams lp;
    lp.N = p.N; lp.IC = p.IC; lp.IH = p.IH; lp.IW = p.IW;
    lp.OC = p.OC; lp.KH = p.KH; lp.KW = p.KW;
    lp.stride_h = p.stride; lp.stride_w = p.stride;
    lp.pad_h = p.pad; lp.pad_w = p.pad;
    return lp;
}

// ============================================================
// Test: im2col_naive vs ref
// ============================================================
void test_conv_im2col_vs_ref(const Conv2DTestParams& p) {
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

    conv2d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());
    conv2d_im2col_naive(p, input.get(), filter.get(), out_im2col.get());

    float tol = (float)(p.IC * p.KH * p.KW) * 2e-5f;
    float md = max_diff(out_ref.get(), out_im2col.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s im2col_naive vs ref max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: dnnopt::conv2d_fp32 vs ref (no bias, no post-op)
// ============================================================
void test_conv_optimized(const Conv2DTestParams& p) {
    int OH = p.OH(), OW = p.OW();
    size_t in_sz  = (size_t)p.N * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OH * OW * p.OC;

    auto input  = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_opt = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz);
    fill_random(filter.get(), flt_sz);

    conv2d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());

    auto lp = to_lib_params(p);
    dnnopt::conv2d_fp32(lp, input.get(), filter.get(), nullptr,
                        out_opt.get(), dnnopt::ConvPostOp::kNone);

    float tol = (float)(p.IC * p.KH * p.KW) * 5e-5f;
    float md = max_diff(out_ref.get(), out_opt.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s optimized vs ref max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: dnnopt::conv2d_fp32 with bias
// ============================================================
void test_conv_bias(const Conv2DTestParams& p) {
    int OH = p.OH(), OW = p.OW();
    size_t in_sz  = (size_t)p.N * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OH * OW * p.OC;

    auto input  = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto bias   = dnnopt::aligned_array<float>(p.OC);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_opt = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);
    fill_random(bias.get(), p.OC, 77);

    conv2d_ref(p, input.get(), filter.get(), bias.get(), out_ref.get());

    auto lp = to_lib_params(p);
    dnnopt::conv2d_fp32(lp, input.get(), filter.get(), bias.get(),
                        out_opt.get(), dnnopt::ConvPostOp::kNone);

    float tol = (float)(p.IC * p.KH * p.KW) * 5e-5f;
    float md = max_diff(out_ref.get(), out_opt.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s bias max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: dnnopt::conv2d_fp32 with bias + ReLU
// ============================================================
void test_conv_bias_relu(const Conv2DTestParams& p) {
    int OH = p.OH(), OW = p.OW();
    size_t in_sz  = (size_t)p.N * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OH * OW * p.OC;

    auto input  = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto bias   = dnnopt::aligned_array<float>(p.OC);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_opt = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);
    fill_random(bias.get(), p.OC, 77);

    // Reference: bias + relu
    conv2d_ref(p, input.get(), filter.get(), bias.get(), out_ref.get());
    apply_ref_postop(out_ref.get(), out_sz, dnnopt::ConvPostOp::kBiasRelu);

    auto lp = to_lib_params(p);
    dnnopt::conv2d_fp32(lp, input.get(), filter.get(), bias.get(),
                        out_opt.get(), dnnopt::ConvPostOp::kBiasRelu);

    float tol = (float)(p.IC * p.KH * p.KW) * 5e-5f;
    float md = max_diff(out_ref.get(), out_opt.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s bias+relu max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: dnnopt::conv2d_fp32 with ReLU6 (no bias)
// ============================================================
void test_conv_relu6(const Conv2DTestParams& p) {
    int OH = p.OH(), OW = p.OW();
    size_t in_sz  = (size_t)p.N * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OH * OW * p.OC;

    auto input  = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_opt = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);

    conv2d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());
    apply_ref_postop(out_ref.get(), out_sz, dnnopt::ConvPostOp::kRelu6);

    auto lp = to_lib_params(p);
    dnnopt::conv2d_fp32(lp, input.get(), filter.get(), nullptr,
                        out_opt.get(), dnnopt::ConvPostOp::kRelu6);

    float tol = (float)(p.IC * p.KH * p.KW) * 5e-5f;
    float md = max_diff(out_ref.get(), out_opt.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv2D %s relu6 max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

}  // namespace

int main() {
    printf("=== test_conv_correctness ===\n");

    const Conv2DTestParams tests[] = {
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

    // Section 1: im2col_naive vs ref (existing tests)
    printf("\n--- im2col_naive vs ref ---\n");
    for (const auto& t : tests) {
        test_conv_im2col_vs_ref(t);
    }

    // Section 2: optimized conv2d_fp32 vs ref (no bias, no postop)
    printf("\n--- dnnopt::conv2d_fp32 vs ref ---\n");
    for (const auto& t : tests) {
        test_conv_optimized(t);
    }

    // Section 3: bias tests
    printf("\n--- conv2d_fp32 with bias ---\n");
    const Conv2DTestParams bias_tests[] = {
        {1, 3, 8, 8, 4, 3, 3, 1, 1, "bias-3x3"},
        {1, 3, 8, 8, 4, 1, 1, 1, 0, "bias-1x1"},
        {1, 64, 14, 14, 128, 3, 3, 1, 1, "bias-large"},
        {2, 16, 8, 8, 32, 3, 3, 1, 1, "bias-batch2"},
    };
    for (const auto& t : bias_tests) {
        test_conv_bias(t);
    }

    // Section 4: bias + relu
    printf("\n--- conv2d_fp32 bias+relu ---\n");
    for (const auto& t : bias_tests) {
        test_conv_bias_relu(t);
    }

    // Section 5: relu6 (no bias)
    printf("\n--- conv2d_fp32 relu6 ---\n");
    const Conv2DTestParams relu6_tests[] = {
        {1, 3, 8, 8, 4, 3, 3, 1, 1, "relu6-3x3"},
        {1, 3, 8, 8, 4, 1, 1, 1, 0, "relu6-1x1"},
        {1, 32, 16, 16, 64, 3, 3, 1, 1, "relu6-medium"},
    };
    for (const auto& t : relu6_tests) {
        test_conv_relu6(t);
    }

    TEST_SUMMARY();
}
