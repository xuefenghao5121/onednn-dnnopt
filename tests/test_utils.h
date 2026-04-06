#pragma once
/// @file test_utils.h
/// Minimal test assertion macros (no external dependency).

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dnnopt { namespace test {

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    dnnopt::test::g_tests_run++; \
    if (!(cond)) { \
        dnnopt::test::g_tests_failed++; \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
    } else { \
        dnnopt::test::g_tests_passed++; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, tol, msg) do { \
    dnnopt::test::g_tests_run++; \
    double _diff = std::fabs((double)(a) - (double)(b)); \
    if (_diff > (tol)) { \
        dnnopt::test::g_tests_failed++; \
        fprintf(stderr, "  FAIL: %s:%d: %s (got %.6e vs %.6e, diff=%.6e, tol=%.6e)\n", \
                __FILE__, __LINE__, msg, (double)(a), (double)(b), _diff, (double)(tol)); \
    } else { \
        dnnopt::test::g_tests_passed++; \
    } \
} while(0)

#define TEST_SUMMARY() do { \
    printf("\n  Results: %d/%d passed", \
           dnnopt::test::g_tests_passed, dnnopt::test::g_tests_run); \
    if (dnnopt::test::g_tests_failed > 0) \
        printf(", %d FAILED", dnnopt::test::g_tests_failed); \
    printf("\n\n"); \
    return dnnopt::test::g_tests_failed > 0 ? 1 : 0; \
} while(0)

}}  // namespace dnnopt::test
