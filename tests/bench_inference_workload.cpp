/// @file bench_inference_workload.cpp
/// Simulated CVR model inference workload: oneDNN-native vs oneDNN+dnnopt.
///
/// Models a typical CVR (Content Video Recommendation) model:
///   - Embedding: [batch, hidden] = [batch, vocab] × [vocab, hidden]
///   - FC layers: [batch, out] = [batch, in] × [in, out]
///   - Attention: [batch, heads*dim] = [batch, seq] × [seq, heads*dim]
///   - Classifier: [batch, classes] = [batch, hidden] × [hidden, classes]
///
/// Uses dnnl_sgemm (the actual API TensorFlow calls through oneDNN).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <dnnl.h>

namespace {

struct Layer {
    int M, N, K;
    const char* name;
};

struct ModelLayer {
    const char* model_name;
    Layer* layers;
    int n_layers;
};

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

double bench_layer(int M, int N, int K,
                   const float* A, const float* B, float* C) {
    float alpha = 1.0f, beta = 0.0f;
    for (int w = 0; w < 3; w++)
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);

    std::vector<double> times;
    for (int i = 0; i < 20; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[10];  // median
}

}  // namespace

int main(int argc, char** argv) {
    // Layer definitions for typical inference models
    // CVR model (batch=1 and batch=4)
    Layer cvr_b1[] = {
        {1, 256, 1024, "embedding"},
        {1, 128, 256,  "fc1"},
        {1, 64, 128,   "fc2"},
        {1, 10, 64,    "classifier"},
    };
    Layer cvr_b4[] = {
        {4, 256, 1024, "embedding"},
        {4, 128, 256,  "fc1"},
        {4, 64, 128,   "fc2"},
        {4, 10, 64,    "classifier"},
    };

    // BERT-small (batch=1,4)
    Layer bert_b1[] = {
        {1, 768, 768,   "qkv_proj"},
        {1, 768, 768,   "out_proj"},
        {1, 3072, 768,  "ffn1"},
        {1, 768, 3072,  "ffn2"},
    };
    Layer bert_b4[] = {
        {4, 768, 768,   "qkv_proj"},
        {4, 768, 768,   "out_proj"},
        {4, 3072, 768,  "ffn1"},
        {4, 768, 3072,  "ffn2"},
    };

    // LLM inference (batch=1,4)
    Layer llm_b1[] = {
        {1, 4096, 4096, "q_proj"},
        {1, 4096, 4096, "k_proj"},
        {1, 4096, 4096, "v_proj"},
        {1, 4096, 4096, "o_proj"},
        {1, 4096, 11008, "gate_proj"},
        {1, 4096, 4096, "up_proj"},
        {1, 11008, 4096, "down_proj"},
    };
    Layer llm_b4[] = {
        {4, 4096, 4096, "q_proj"},
        {4, 4096, 4096, "k_proj"},
        {4, 4096, 4096, "v_proj"},
        {4, 4096, 4096, "o_proj"},
        {4, 4096, 11008, "gate_proj"},
        {4, 4096, 4096, "up_proj"},
        {4, 11008, 4096, "down_proj"},
    };

    ModelLayer models[] = {
        {"CVR model (batch=1)", cvr_b1, 4},
        {"CVR model (batch=4)", cvr_b4, 4},
        {"BERT-small (batch=1)", bert_b1, 4},
        {"BERT-small (batch=4)", bert_b4, 4},
        {"LLM inference (batch=1)", llm_b1, 7},
        {"LLM inference (batch=4)", llm_b4, 7},
    };

    printf("==========================================================\n");
    printf("  Inference Workload: oneDNN-native vs oneDNN+dnnopt\n");
    printf("  Simulates real model inference layer by layer\n");
    printf("==========================================================\n\n");

    int n_models = sizeof(models) / sizeof(models[0]);

    for (int m = 0; m < n_models; m++) {
        const auto& model = models[m];
        printf("--- %s ---\n", model.model_name);
        printf("  %-20s  %10s  %10s  %8s\n", "Layer", "upstream", "+dnnopt", "ratio");
        printf("  %-20s  %10s  %10s  %8s\n", "-----", "--------", "-------", "-----");

        double total_up_us = 0, total_dn_us = 0;

        for (int l = 0; l < model.n_layers; l++) {
            const auto& layer = model.layers[l];
            int M = layer.M, N = layer.N, K = layer.K;

            // Allocate aligned buffers
            float *A = (float*)aligned_alloc(64, (size_t)M*K*sizeof(float));
            float *B = (float*)aligned_alloc(64, (size_t)K*N*sizeof(float));
            float *C = (float*)aligned_alloc(64, (size_t)M*N*sizeof(float));

            fill_random(A, M*K);
            fill_random(B, K*N);

            // We'll use LD_PRELOAD to switch between upstream/dnnopt
            // For standalone testing, we use the linked library
            double us = bench_layer(M, N, K, A, B, C);
            double gflops = 2.0 * M * N * K / (us * 1e3);

            printf("  %-20s  %8.0f us  %7.2f GF\n", layer.name, us, gflops);

            total_up_us += us;
            free(A); free(B); free(C);
        }

        printf("  Total: %.0f us per inference\n\n", total_up_us);
    }

    return 0;
}
