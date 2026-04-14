/**
 * Direct test of oneDNN dispatch logic to dnnopt
 * This verifies the dispatch wrapper code works correctly
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>

// Include dnnopt headers (C++ headers, don't use extern "C")
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

// Copy of the NEON transpose from dnnopt_gemm_wrapper.hpp
inline void neon_transpose_f32(const float *src, float *dst,
        int rows, int cols, int ld_src, int ld_dst) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[c * ld_dst + r] = src[r * ld_src + c];
}

// Minimal ldc for dnnopt microkernel
static constexpr int DNNOPT_MIN_LDC = 12;

// Safe wrapper around dnnopt::gemm_fp32
inline void safe_gemm_fp32(int M, int N, int K, float alpha,
        const float *A, int lda, const float *B, int ldb,
        float beta, float *C, int ldc) {
    if (N >= DNNOPT_MIN_LDC || ldc >= DNNOPT_MIN_LDC) {
        dnnopt::gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        int safe_ldc = DNNOPT_MIN_LDC;
        float* tmp = (float*)dnnopt::aligned_malloc(sizeof(float) * M * safe_ldc, 64);

        if (beta != 0.0f) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    tmp[i * safe_ldc + j] = C[i * ldc + j];
        }

        dnnopt::gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, tmp, safe_ldc);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * ldc + j] = tmp[i * safe_ldc + j];

        dnnopt::aligned_free(tmp);
    }
}

// Row-major GEMM with transpose support
inline void dnnopt_gemm_row_major(bool transA, bool transB,
        int M, int N, int K, float alpha,
        const float *A, int lda, const float *B, int ldb,
        float beta, float *C, int ldc) {
    if (!transA && !transB) {
        safe_gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (transA && !transB) {
        float* A_T = (float*)dnnopt::aligned_malloc(sizeof(float) * M * K, 64);
        neon_transpose_f32(A, A_T, K, M, lda, K);
        safe_gemm_fp32(M, N, K, alpha, A_T, K, B, ldb, beta, C, ldc);
        dnnopt::aligned_free(A_T);
    } else if (!transA && transB) {
        float* B_T = (float*)dnnopt::aligned_malloc(sizeof(float) * K * N, 64);
        neon_transpose_f32(B, B_T, N, K, ldb, N);
        safe_gemm_fp32(M, N, K, alpha, A, lda, B_T, N, beta, C, ldc);
        dnnopt::aligned_free(B_T);
    } else {
        float* A_T = (float*)dnnopt::aligned_malloc(sizeof(float) * M * K, 64);
        float* B_T = (float*)dnnopt::aligned_malloc(sizeof(float) * K * N, 64);
        neon_transpose_f32(A, A_T, K, M, lda, K);
        neon_transpose_f32(B, B_T, N, K, ldb, N);
        safe_gemm_fp32(M, N, K, alpha, A_T, K, B_T, N, beta, C, ldc);
        dnnopt::aligned_free(A_T);
        dnnopt::aligned_free(B_T);
    }
}

// OneDNN-style sgemm wrapper (column-major to row-major)
inline bool onednn_dnnopt_sgemm(const char *transa, const char *transb,
        long M, long N, long K, float alpha,
        const float *A, long lda, const float *B, long ldb,
        float beta, float *C, long ldc) {

    if (M == 0 || N == 0) return true;
    if (K == 0) {
        if (beta == 0.0f) {
            for (long j = 0; j < N; j++)
                for (long i = 0; i < M; i++)
                    C[j * ldc + i] = 0.0f;
        } else if (beta != 1.0f) {
            for (long j = 0; j < N; j++)
                for (long i = 0; i < M; i++)
                    C[j * ldc + i] *= beta;
        }
        return true;
    }

    bool tA = (*transa == 'T' || *transa == 't');
    bool tB = (*transb == 'T' || *transb == 't');

    // Col-major to row-major duality
    int rm_M = (int)N;
    int rm_N = (int)M;
    int rm_K = (int)K;

    if (ldc >= rm_N) {
        dnnopt_gemm_row_major(tB, tA, rm_M, rm_N, rm_K, alpha,
                B, ldb, A, lda, beta, C, ldc);
    } else {
        int safe_ldc = rm_N;
        float* tmp = (float*)dnnopt::aligned_malloc(sizeof(float) * rm_M * safe_ldc, 64);

        if (beta != 0.0f) {
            for (int r = 0; r < rm_M; r++)
                for (int c = 0; c < rm_N; c++)
                    tmp[r * safe_ldc + c] = C[r * ldc + c];
        }

        dnnopt_gemm_row_major(tB, tA, rm_M, rm_N, rm_K, alpha,
                B, ldb, A, lda, beta, tmp, safe_ldc);

        for (int r = 0; r < rm_M; r++)
            for (int c = 0; c < rm_N; c++)
                C[r * ldc + c] = tmp[r * safe_ldc + c];

        dnnopt::aligned_free(tmp);
    }

    return true;
}

// Test shapes (oneDNN GEMM patterns)
struct TestShape {
    long m, n, k;
    const char* name;
    char ta, tb;  // transpose flags
};

TestShape test_shapes[] = {
    {1, 256, 1024, "CVR embedding (NN)", 'N', 'N'},
    {256, 1024, 1, "CVR embedding (TN)", 'T', 'N'},
    {4, 256, 1024, "CVR embedding b=4", 'N', 'N'},
    {32, 256, 256, "BERT QKV (NN)", 'N', 'N'},
    {32, 256, 512, "BERT FFN1", 'N', 'N'},
    {32, 512, 256, "BERT FFN2", 'N', 'N'},
    {8, 512, 512, "LLM QKV", 'N', 'N'},
    {8, 512, 1376, "LLM FFN1", 'N', 'N'},
    {256, 256, 256, "Square", 'N', 'N'},
};

// Reference GEMM (simple implementation)
void ref_sgemm(bool ta, bool tb, long M, long N, long K, float alpha,
               const float* A, long lda, const float* B, long ldb,
               float beta, float* C, long ldc, size_t B_size) {
    // Column-major reference: element (row, col) is at row + col*ld
    for (long n = 0; n < N; n++) {
        for (long m = 0; m < M; m++) {
            float sum = 0.0f;
            for (long k = 0; k < K; k++) {
                // A: M×K if not transposed, K×M if transposed
                // NOT transposed: element A(m,k) at index k*lda + m (column k, row m)
                // Transposed: element A(m,k) comes from stored A(k,m) at index m*lda + k
                size_t a_idx = ta ? (m * lda + k) : (k * lda + m);

                // B: K×N if not transposed, N×K if transposed
                // NOT transposed: element B(k,n) at index n*ldb + k (column n, row k)
                // Transposed: element B(k,n) comes from stored B(n,k) at index k*ldb + n
                size_t b_idx = tb ? (k * ldb + n) : (n * ldb + k);

                if (b_idx >= B_size) {
                    std::cerr << "B index out of bounds: " << b_idx << " >= " << B_size << std::endl;
                    std::cerr << "  M=" << M << " N=" << N << " K=" << K << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc << std::endl;
                    std::cerr << "  m=" << m << " n=" << n << " k=" << k << " ta=" << ta << " tb=" << tb << std::endl;
                    return;
                }
                float a = A[a_idx];
                float b = B[b_idx];
                sum += a * b;
            }
            // C is M×N, element C(m,n) at index n*ldc + m (column n, row m)
            C[n * ldc + m] = alpha * sum + (beta != 0.0f ? beta * C[n * ldc + m] : 0.0f);
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "oneDNN → dnnopt Dispatch Logic Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0, total = 0;

    for (const auto& shape : test_shapes) {
        total++;

        // Allocate matrices (column-major for oneDNN convention)
        // A: M×K (K×M if transposed), B: K×N (N×K if transposed), C: M×N
        std::vector<float> A(shape.m * shape.k);
        std::vector<float> B(shape.k * shape.n);
        std::vector<float> C(shape.m * shape.n);
        std::vector<float> C_ref(shape.m * shape.n);

        // Initialize with random data
        srand(42);  // Fixed seed for reproducibility
        for (size_t i = 0; i < A.size(); i++) A[i] = (float)rand() / RAND_MAX;
        for (size_t i = 0; i < B.size(); i++) B[i] = (float)rand() / RAND_MAX;
        for (size_t i = 0; i < C.size(); i++) C[i] = C_ref[i] = 0.0f;

        // Run reference
        bool ta = (shape.ta == 'T' || shape.ta == 't');
        bool tb = (shape.tb == 'T' || shape.tb == 't');

        // Column-major leading dimensions:
        // For A: if not transposed, lda >= M; if transposed, lda >= K
        // For B: if not transposed, ldb >= K; if transposed, ldb >= N
        // For C: ldc >= M
        long lda = ta ? shape.k : shape.m;
        long ldb = tb ? shape.n : shape.k;
        long ldc = shape.m;

        ref_sgemm(ta, tb, shape.m, shape.n, shape.k, 1.0f,
                  A.data(), lda, B.data(), ldb, 0.0f, C_ref.data(), ldc, B.size());

        // Run dispatch
        char ta_str[2] = {shape.ta, 0};
        char tb_str[2] = {shape.tb, 0};
        bool ok = onednn_dnnopt_sgemm(ta_str, tb_str,
                shape.m, shape.n, shape.k, 1.0f,
                A.data(), lda, B.data(), ldb, 0.0f, C.data(), ldc);

        // Check correctness
        float max_err = 0.0f;
        for (size_t i = 0; i < C.size(); i++) {
            float err = std::abs(C[i] - C_ref[i]);
            max_err = std::max(max_err, err);
        }

        float tol = shape.k * 2e-5f;  // FP32 accumulation tolerance
        bool correct = (max_err < tol);

        std::cout << std::left << std::setw(25) << shape.name
                  << " M=" << std::setw(4) << shape.m
                  << " N=" << std::setw(4) << shape.n
                  << " K=" << std::setw(4) << shape.k
                  << " [" << shape.ta << shape.tb << "]"
                  << "  err=" << std::scientific << std::setprecision(1) << max_err
                  << "  " << (correct ? "✓ PASS" : "✗ FAIL")
                  << std::fixed << std::endl;

        if (correct) passed++;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Correctness: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    if (passed == total) {
        std::cout << std::endl;
        std::cout << "✓ Dispatch logic is CORRECT!" << std::endl;
        std::cout << "✓ oneDNN → dnnopt bridge works as designed" << std::endl;
        std::cout << std::endl;
        std::cout << "Next step: Verify oneDNN library is built with" << std::endl;
        std::cout << "  DNNL_USE_DNNOPT=1 and properly linked to dnnopt." << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "✗ Dispatch logic has ERRORS!" << std::endl;
    }

    return (passed == total) ? 0 : 1;
}
