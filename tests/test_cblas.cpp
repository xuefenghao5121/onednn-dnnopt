/// @file test_cblas.cpp
/// Correctness tests for CBLAS sgemm wrapper.
/// Tests all Order × TransA × TransB combinations.

#include "dnnopt/blas/cblas.h"
#include "dnnopt/aligned_alloc.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

// Forward declaration for Fortran sgemm_
extern "C" void sgemm_(const char*, const char*, const int*, const int*, const int*,
                       const float*, const float*, const int*,
                       const float*, const int*,
                       const float*, float*, const int*);

namespace {

void fill_random(float* data, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Naive reference GEMM (row-major, no-transpose).
void ref_gemm_nn(int M, int N, int K,
                 float alpha, const float* A, int lda,
                 const float* B, int ldb,
                 float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}

/// Naive reference for ColMajor GEMM.
/// In column-major: A is m×k stored col-major (lda rows per column).
void ref_colmajor_gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                       int M, int N, int K,
                       float alpha,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float beta,
                       float* C, int ldc) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Column-major access:
                float a_val = (ta == CblasNoTrans) ? A[k * lda + i] : A[i * lda + k];
                float b_val = (tb == CblasNoTrans) ? B[j * ldb + k] : B[k * ldb + j];
                acc += a_val * b_val;
            }
            C[j * ldc + i] = alpha * acc + beta * C[j * ldc + i];
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
// Test: RowMajor, NoTrans, NoTrans
// ============================================================
void test_row_nn(int M, int N, int K) {
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    fill_random(C_ref.get(), (size_t)M * N, 77);
    memcpy(C_test.get(), C_ref.get(), (size_t)M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.5f;
    ref_gemm_nn(M, N, K, alpha, A.get(), K, B.get(), N, beta, C_ref.get(), N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A.get(), K, B.get(), N, beta, C_test.get(), N);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Row+NN %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: RowMajor, Trans, NoTrans
// ============================================================
void test_row_tn(int M, int N, int K) {
    // A is K×M row-major (transposed)
    auto A = dnnopt::aligned_array<float>((size_t)K * M);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)K * M, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    // Ref: manually transpose A then multiply
    auto A_T = dnnopt::aligned_array<float>((size_t)M * K);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < M; ++j)
            A_T[j * K + i] = A[i * M + j];

    ref_gemm_nn(M, N, K, 1.0f, A_T.get(), K, B.get(), N, 0.0f, C_ref.get(), N);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0f, A.get(), M, B.get(), N, 0.0f, C_test.get(), N);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Row+TN %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: RowMajor, NoTrans, Trans
// ============================================================
void test_row_nt(int M, int N, int K) {
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    // B is N×K row-major (transposed)
    auto B = dnnopt::aligned_array<float>((size_t)N * K);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)N * K, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    // Ref: transpose B then multiply
    auto B_T = dnnopt::aligned_array<float>((size_t)K * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B_T[j * N + i] = B[i * K + j];

    ref_gemm_nn(M, N, K, 1.0f, A.get(), K, B_T.get(), N, 0.0f, C_ref.get(), N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A.get(), K, B.get(), K, 0.0f, C_test.get(), N);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Row+NT %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: RowMajor, Trans, Trans
// ============================================================
void test_row_tt(int M, int N, int K) {
    auto A = dnnopt::aligned_array<float>((size_t)K * M);
    auto B = dnnopt::aligned_array<float>((size_t)N * K);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)K * M, 42);
    fill_random(B.get(), (size_t)N * K, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    auto A_T = dnnopt::aligned_array<float>((size_t)M * K);
    auto B_T = dnnopt::aligned_array<float>((size_t)K * N);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < M; ++j)
            A_T[j * K + i] = A[i * M + j];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B_T[j * N + i] = B[i * K + j];

    ref_gemm_nn(M, N, K, 1.0f, A_T.get(), K, B_T.get(), N, 0.0f, C_ref.get(), N);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                M, N, K, 1.0f, A.get(), M, B.get(), K, 0.0f, C_test.get(), N);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Row+TT %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: ColMajor, NoTrans, NoTrans
// ============================================================
void test_col_nn(int M, int N, int K) {
    // ColMajor: A is m×k col-major (lda=M), B is k×n col-major (ldb=K)
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    ref_colmajor_gemm(CblasNoTrans, CblasNoTrans, M, N, K,
                      1.0f, A.get(), M, B.get(), K, 0.0f, C_ref.get(), M);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A.get(), M, B.get(), K, 0.0f, C_test.get(), M);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Col+NN %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: ColMajor, Trans, NoTrans
// ============================================================
void test_col_tn(int M, int N, int K) {
    // A is k×m col-major (transposed, lda=K)
    auto A = dnnopt::aligned_array<float>((size_t)K * M);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)K * M, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    ref_colmajor_gemm(CblasTrans, CblasNoTrans, M, N, K,
                      1.0f, A.get(), K, B.get(), K, 0.0f, C_ref.get(), M);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0f, A.get(), K, B.get(), K, 0.0f, C_test.get(), M);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Col+TN %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: ColMajor, NoTrans, Trans
// ============================================================
void test_col_nt(int M, int N, int K) {
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    // B is n×k col-major (transposed, ldb=N)
    auto B = dnnopt::aligned_array<float>((size_t)N * K);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)N * K, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    ref_colmajor_gemm(CblasNoTrans, CblasTrans, M, N, K,
                      1.0f, A.get(), M, B.get(), N, 0.0f, C_ref.get(), M);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A.get(), M, B.get(), N, 0.0f, C_test.get(), M);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Col+NT %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: Fortran sgemm_
// ============================================================
void test_fortran_sgemm(int M, int N, int K) {
    // Fortran sgemm_ uses column-major
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    memset(C_ref.get(), 0, (size_t)M * N * sizeof(float));
    memset(C_test.get(), 0, (size_t)M * N * sizeof(float));

    ref_colmajor_gemm(CblasNoTrans, CblasNoTrans, M, N, K,
                      1.0f, A.get(), M, B.get(), K, 0.0f, C_ref.get(), M);

    // Call Fortran interface
    char ta = 'N', tb = 'N';
    float alpha = 1.0f, beta = 0.0f;
    sgemm_(&ta, &tb, &M, &N, &K, &alpha, A.get(), &M, B.get(), &K, &beta, C_test.get(), &M);

    float tol = K * 5e-5f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "Fortran sgemm_ %dx%dx%d max_diff=%.6e tol=%.6e", M, N, K, md, tol);
    TEST_ASSERT(md < tol, msg);
}

// ============================================================
// Test: alpha/beta scaling
// ============================================================
void test_alpha_beta(int M, int N, int K) {
    auto A = dnnopt::aligned_array<float>((size_t)M * K);
    auto B = dnnopt::aligned_array<float>((size_t)K * N);
    auto C_ref = dnnopt::aligned_array<float>((size_t)M * N);
    auto C_test = dnnopt::aligned_array<float>((size_t)M * N);

    fill_random(A.get(), (size_t)M * K, 42);
    fill_random(B.get(), (size_t)K * N, 123);
    fill_random(C_ref.get(), (size_t)M * N, 77);
    memcpy(C_test.get(), C_ref.get(), (size_t)M * N * sizeof(float));

    float alpha = 2.5f, beta = -0.3f;
    ref_gemm_nn(M, N, K, alpha, A.get(), K, B.get(), N, beta, C_ref.get(), N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A.get(), K, B.get(), N, beta, C_test.get(), N);

    float tol = K * 5e-4f;
    float md = max_diff(C_ref.get(), C_test.get(), (size_t)M * N);
    char msg[256];
    snprintf(msg, sizeof(msg), "alpha=2.5 beta=-0.3 %dx%dx%d max_diff=%.6e", M, N, K, md);
    TEST_ASSERT(md < tol, msg);
}

}  // namespace

int main() {
    printf("=== test_cblas ===\n");

    struct { int M, N, K; } sizes[] = {
        {4, 4, 4},
        {16, 16, 16},
        {32, 64, 48},
        {128, 128, 128},
        {7, 13, 11},       // Non-aligned sizes
        {1, 64, 32},       // Small M (batch=1)
        {256, 256, 256},   // Large
    };

    printf("\n--- RowMajor NoTrans NoTrans ---\n");
    for (auto& s : sizes) test_row_nn(s.M, s.N, s.K);

    printf("\n--- RowMajor Trans NoTrans ---\n");
    for (auto& s : sizes) test_row_tn(s.M, s.N, s.K);

    printf("\n--- RowMajor NoTrans Trans ---\n");
    for (auto& s : sizes) test_row_nt(s.M, s.N, s.K);

    printf("\n--- RowMajor Trans Trans ---\n");
    for (auto& s : sizes) test_row_tt(s.M, s.N, s.K);

    printf("\n--- ColMajor NoTrans NoTrans ---\n");
    for (auto& s : sizes) test_col_nn(s.M, s.N, s.K);

    printf("\n--- ColMajor Trans NoTrans ---\n");
    for (auto& s : sizes) test_col_tn(s.M, s.N, s.K);

    printf("\n--- ColMajor NoTrans Trans ---\n");
    for (auto& s : sizes) test_col_nt(s.M, s.N, s.K);

    printf("\n--- Fortran sgemm_ ---\n");
    for (auto& s : sizes) test_fortran_sgemm(s.M, s.N, s.K);

    printf("\n--- Alpha/Beta scaling ---\n");
    for (auto& s : sizes) test_alpha_beta(s.M, s.N, s.K);

    TEST_SUMMARY();
}
