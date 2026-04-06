#pragma once
/// @file timer.h
/// High-resolution timing utilities for benchmarking.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace dnnopt {

/// Simple high-resolution timer.
class Timer {
public:
    void start() { start_ = clock::now(); }
    void stop()  { end_ = clock::now(); }

    /// Elapsed time in seconds.
    double elapsed_sec() const {
        return std::chrono::duration<double>(end_ - start_).count();
    }
    /// Elapsed time in milliseconds.
    double elapsed_ms() const { return elapsed_sec() * 1e3; }
    /// Elapsed time in microseconds.
    double elapsed_us() const { return elapsed_sec() * 1e6; }

private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_{}, end_{};
};

/// Benchmark statistics from multiple runs.
struct BenchStats {
    std::string name;
    size_t      runs     = 0;
    double      min_ms   = 0;
    double      max_ms   = 0;
    double      mean_ms  = 0;
    double      median_ms = 0;
    double      stddev_ms = 0;
    double      gflops   = 0;     // 0 if not applicable
    double      gbps     = 0;     // 0 if not applicable
};

/// Run a callable `warmup` times then `runs` times, collecting timing stats.
/// @param name   Benchmark name for reporting.
/// @param flops  Total floating-point operations per call (0 to skip GFLOPS).
/// @param bytes  Total bytes accessed per call (0 to skip bandwidth).
/// @param warmup Number of warmup iterations.
/// @param runs   Number of measured iterations.
/// @param fn     Callable to benchmark.
template<typename Fn>
BenchStats benchmark(const std::string& name, double flops, double bytes,
                     int warmup, int runs, Fn&& fn);

/// Print a BenchStats result to stdout.
void print_bench_stats(const BenchStats& stats);

/// Write a vector of BenchStats to a CSV file.
void write_csv(const std::string& filepath,
               const std::vector<BenchStats>& results);

// ---- Template implementation ----

template<typename Fn>
BenchStats benchmark(const std::string& name, double flops, double bytes,
                     int warmup, int runs, Fn&& fn) {
    Timer t;

    // Warmup
    for (int i = 0; i < warmup; ++i) fn();

    // Collect timings
    std::vector<double> times(runs);
    for (int i = 0; i < runs; ++i) {
        t.start();
        fn();
        t.stop();
        times[i] = t.elapsed_ms();
    }

    // Compute stats
    std::sort(times.begin(), times.end());

    BenchStats s;
    s.name = name;
    s.runs = runs;
    s.min_ms = times.front();
    s.max_ms = times.back();
    s.median_ms = (runs % 2 == 0)
        ? (times[runs/2 - 1] + times[runs/2]) / 2.0
        : times[runs/2];

    double sum = 0;
    for (auto v : times) sum += v;
    s.mean_ms = sum / runs;

    double var = 0;
    for (auto v : times) var += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = std::sqrt(var / runs);

    if (flops > 0) s.gflops = (flops / 1e9) / (s.median_ms / 1e3);
    if (bytes > 0) s.gbps   = (bytes / 1e9) / (s.median_ms / 1e3);

    return s;
}

}  // namespace dnnopt
