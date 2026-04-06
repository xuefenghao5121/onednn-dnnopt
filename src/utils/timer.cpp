/// @file timer.cpp
/// Benchmark reporting and CSV output.

#include "dnnopt/timer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace dnnopt {

void print_bench_stats(const BenchStats& s) {
    printf("  %-40s  ", s.name.c_str());
    printf("median: %8.3f ms  ", s.median_ms);
    printf("min: %8.3f ms  ", s.min_ms);
    printf("mean: %8.3f ms  ", s.mean_ms);
    printf("stddev: %6.3f ms", s.stddev_ms);
    if (s.gflops > 0) printf("  %7.2f GFLOPS", s.gflops);
    if (s.gbps > 0)   printf("  %7.2f GB/s", s.gbps);
    printf("\n");
}

void write_csv(const std::string& filepath,
               const std::vector<BenchStats>& results) {
    std::ofstream f(filepath);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot write CSV to %s\n", filepath.c_str());
        return;
    }

    f << "name,runs,min_ms,max_ms,mean_ms,median_ms,stddev_ms,gflops,gbps\n";
    for (const auto& s : results) {
        f << s.name << ","
          << s.runs << ","
          << s.min_ms << ","
          << s.max_ms << ","
          << s.mean_ms << ","
          << s.median_ms << ","
          << s.stddev_ms << ","
          << s.gflops << ","
          << s.gbps << "\n";
    }
    printf("CSV written to: %s\n", filepath.c_str());
}

}  // namespace dnnopt
