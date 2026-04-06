#!/bin/bash
# Performance profiling script using Linux perf.
# Collects PMU counters, hotspot analysis, and cache behavior.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
RESULT_DIR="${PROJECT_DIR}/benchmarks/profiling/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_DIR}"

BENCH_TARGET="${1:-bench_gemm}"
BENCH_BIN="${BUILD_DIR}/benchmarks/${BENCH_TARGET}"

if [ ! -x "${BENCH_BIN}" ]; then
    echo "Error: ${BENCH_BIN} not found. Build the project first."
    exit 1
fi

echo "============================================"
echo "  Performance Profiling: ${BENCH_TARGET}"
echo "  Results: ${RESULT_DIR}"
echo "============================================"

# ============================================================
# 1. PMU Counter Collection
# ============================================================
echo ""
echo "[1/4] Collecting PMU counters..."

PMU_EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-load-misses"

# Check if ARM-specific events are available
if perf list 2>/dev/null | grep -q "armv8_pmuv3"; then
    PMU_EVENTS="${PMU_EVENTS},armv8_pmuv3/inst_retired/,armv8_pmuv3/cpu_cycles/,armv8_pmuv3/l1d_cache_refill/,armv8_pmuv3/l2d_cache_refill/"
fi

perf stat -e "${PMU_EVENTS}" -o "${RESULT_DIR}/pmu_counters.txt" \
    "${BENCH_BIN}" 3 2>&1 | tail -5

echo "  → ${RESULT_DIR}/pmu_counters.txt"

# ============================================================
# 2. Hotspot Profiling (sampling)
# ============================================================
echo ""
echo "[2/4] Hotspot profiling (perf record)..."

perf record -g --call-graph dwarf -F 999 -o "${RESULT_DIR}/perf.data" \
    "${BENCH_BIN}" 3 2>/dev/null

perf report -i "${RESULT_DIR}/perf.data" --stdio --no-children \
    --percent-limit 1 > "${RESULT_DIR}/hotspots.txt" 2>/dev/null

echo "  → ${RESULT_DIR}/hotspots.txt"

# ============================================================
# 3. Cache Miss Analysis
# ============================================================
echo ""
echo "[3/4] Cache miss analysis..."

CACHE_EVENTS="L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"

perf stat -e "${CACHE_EVENTS}" -o "${RESULT_DIR}/cache_analysis.txt" \
    "${BENCH_BIN}" 3 2>/dev/null

echo "  → ${RESULT_DIR}/cache_analysis.txt"

# ============================================================
# 4. IPC and Pipeline Analysis
# ============================================================
echo ""
echo "[4/4] IPC / pipeline utilization..."

perf stat -e "cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend" \
    -o "${RESULT_DIR}/pipeline_stats.txt" \
    "${BENCH_BIN}" 3 2>/dev/null

echo "  → ${RESULT_DIR}/pipeline_stats.txt"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "  Profiling Complete!"
echo "============================================"
echo ""

# Extract key metrics
echo "--- Key Metrics ---"
if [ -f "${RESULT_DIR}/pmu_counters.txt" ]; then
    grep -E "instructions|cycles|cache-misses|branch-misses" \
        "${RESULT_DIR}/pmu_counters.txt" 2>/dev/null || true
fi

echo ""
echo "Full results in: ${RESULT_DIR}/"
ls -la "${RESULT_DIR}/"
