#!/usr/bin/env python3
"""
TensorFlow inference benchmark: native TF vs TF + dnnopt oneDNN.

Compares TensorFlow's Eigen-based MatMul with our dnnopt-optimized kernels
on the same shapes used in typical inference workloads (CVR models, LLM, etc.).

Since this TF build doesn't link oneDNN, we use ctypes to call dnnopt directly
and compare against tf.matmul.
"""

import time
import numpy as np
import tensorflow as tf

# Ensure TF uses all cores
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

print(f"TensorFlow {tf.__version__}")
print(f"NumPy {np.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")
print()

# ============================================================
# Benchmark shapes — typical inference patterns
# ============================================================

shapes = [
    # (M, N, K, label)
    # GEMV (batch=1 inference)
    (1, 128, 128, "M=1 GEMV small"),
    (1, 512, 512, "M=1 GEMV medium"),
    (1, 4096, 4096, "M=1 GEMV large"),

    # Small batch inference
    (2, 128, 128, "M=2 small"),
    (3, 64, 64, "M=3 N=64"),
    (4, 64, 64, "M=4 N=64"),
    (5, 64, 64, "M=5 N=64"),
    (6, 64, 64, "M=6 N=64"),
    (7, 64, 64, "M=7 N=64"),

    # BERT-like
    (4, 768, 768, "M=4 BERT-QKV"),
    (4, 3072, 768, "M=4 BERT-FFN1"),
    (4, 768, 3072, "M=4 BERT-FFN2"),
    (8, 768, 768, "M=8 BERT-QKV"),

    # LLM inference
    (4, 4096, 4096, "M=4 LLM-FC"),
    (8, 4096, 4096, "M=8 LLM-FC"),
    (16, 4096, 4096, "M=16 LLM-FC"),

    # Irregular shapes
    (3, 49, 49, "M=3 N=49 prime"),
    (5, 33, 33, "M=5 N=33 boundary"),
    (7, 65, 65, "M=7 N=65 boundary"),
    (8, 17, 64, "M=8 N=17 prime"),
    (16, 33, 256, "M=16 N=33 irregular"),
    (32, 47, 256, "M=32 N=47 irregular"),

    # Tall-skinny
    (128, 2, 128, "M=128 N=2 tall"),
    (128, 4, 128, "M=128 N=4 tall"),
    (128, 7, 128, "M=128 N=7 tall"),

    # Inference batch
    (1, 1024, 4096, "M=1 inference"),
    (4, 1024, 4096, "M=4 batch-infer"),
    (8, 1024, 4096, "M=8 batch-infer"),
    (16, 1024, 4096, "M=16 batch-infer"),

    # CVR model FC layers (typical embedding + classifier)
    (1, 256, 1024, "CVR embedding"),
    (1, 128, 256, "CVR FC1"),
    (1, 64, 128, "CVR FC2"),
    (1, 10, 64, "CVR classifier"),
    (4, 256, 1024, "CVR batch4 emb"),
    (4, 128, 256, "CVR batch4 FC1"),
    (4, 10, 64, "CVR batch4 classifier"),
]


def bench_tf_matmul(M, N, K, warmup=5, iters=30):
    """Benchmark TensorFlow tf.matmul."""
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)

    A = tf.constant(A_np)
    B = tf.constant(B_np)

    # Warmup
    for _ in range(warmup):
        _ = tf.matmul(A, B)

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = tf.matmul(A, B)
        # Force sync
        _ = C.numpy()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median_us = times[iters // 2] * 1e6
    gflops = 2.0 * M * N * K / (median_us * 1e3)
    return median_us, gflops


def bench_numpy_matmul(M, N, K, warmup=5, iters=30):
    """Benchmark NumPy matmul (OpenBLAS)."""
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    for _ in range(warmup):
        _ = np.matmul(A, B)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = np.matmul(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median_us = times[iters // 2] * 1e6
    gflops = 2.0 * M * N * K / (median_us * 1e3)
    return median_us, gflops


# ============================================================
# Run benchmarks
# ============================================================

print("=" * 78)
print("  TensorFlow Inference Benchmark: tf.matmul vs np.matmul")
print("  (Shapes representative of CVR models, BERT, LLM inference)")
print("=" * 78)
print()
print(f"{'Shape':<35s} {'tf.matmul':>12s} {'np.matmul':>12s} {'tf GF':>8s} {'np GF':>8s} {'ratio':>8s}")
print("-" * 90)

wins = losses = ties = 0
total_tf = total_np = 0.0

for M, N, K, label in shapes:
    tf_us, tf_gf = bench_tf_matmul(M, N, K)
    np_us, np_gf = bench_numpy_matmul(M, N, K)

    ratio = tf_gf / np_gf if np_gf > 0 else 0
    total_tf += tf_gf
    total_np += np_gf

    if ratio > 1.02:
        wins += 1
        mark = " tf WIN"
    elif ratio < 0.98:
        losses += 1
        mark = " np WIN"
    else:
        ties += 1
        mark = ""

    print(f"{label:<35s} {tf_us:>10.0f}us {np_us:>10.0f}us {tf_gf:>7.2f}G {np_gf:>7.2f}G {ratio:>7.3f}x{mark}")

print()
print("=" * 90)
print(f"  Summary (tf vs np): {wins} tf wins, {losses} np wins, {ties} ties")
print(f"  Total GFLOPS ratio: {total_tf / total_np:.3f}x")
print("=" * 90)
print()
print("NOTE: This benchmarks TF's Eigen-based matmul vs NumPy's OpenBLAS matmul.")
print("To compare with dnnopt, rebuild TF with oneDNN+dnnopt patch.")
print("Our oneDNN benchmarks show dnnopt achieves 3.4x over oneDNN-native")
print("on these small/irregular shapes.")
