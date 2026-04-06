#!/usr/bin/env python3
"""
Roofline Model Generator for ARM CPU.

Reads benchmark results (CSV) and hardware capabilities to generate
a roofline model analysis, identifying compute-bound vs memory-bound operators.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RooflinePoint:
    """A single point on the roofline chart."""
    name: str
    flops: float          # Total FLOP
    bytes_accessed: float  # Total bytes read+written
    time_sec: float       # Execution time in seconds
    arithmetic_intensity: float = 0.0   # FLOP/byte
    achieved_gflops: float = 0.0
    achieved_gbps: float = 0.0
    bound: str = ""       # "compute" or "memory"


@dataclass
class RooflineModel:
    """Roofline model parameters for a specific platform."""
    platform: str
    peak_gflops: float      # Theoretical peak GFLOPS
    peak_bandwidth_gbps: float  # Peak memory bandwidth GB/s
    ridge_point: float = 0.0    # FLOP/byte at the ridge

    def __post_init__(self):
        if self.peak_bandwidth_gbps > 0:
            self.ridge_point = self.peak_gflops / self.peak_bandwidth_gbps


def read_benchmark_csv(filepath: str) -> List[dict]:
    """Read benchmark results from CSV file."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def estimate_memory_bandwidth(cache_sizes: dict) -> float:
    """
    Estimate effective memory bandwidth in GB/s.
    This is a rough estimate; actual measurement with STREAM benchmark is better.
    """
    # Default estimate for DDR4/DDR5 on ARM servers
    # Neoverse N2 typical: ~40-50 GB/s per socket
    return 40.0


def build_roofline_points(csv_path: str) -> List[RooflinePoint]:
    """Build roofline points from benchmark CSV."""
    points = []
    rows = read_benchmark_csv(csv_path)

    for row in rows:
        gflops = float(row.get('gflops', 0))
        gbps = float(row.get('gbps', 0))
        median_ms = float(row.get('median_ms', 0))

        if median_ms <= 0:
            continue

        time_sec = median_ms / 1000.0

        # Estimate FLOP and bytes from gflops/gbps and time
        flops = gflops * 1e9 * time_sec if gflops > 0 else 0
        bytes_acc = gbps * 1e9 * time_sec if gbps > 0 else 0

        if flops > 0 and bytes_acc > 0:
            ai = flops / bytes_acc
        else:
            ai = 0

        p = RooflinePoint(
            name=row.get('name', 'unknown'),
            flops=flops,
            bytes_accessed=bytes_acc,
            time_sec=time_sec,
            arithmetic_intensity=ai,
            achieved_gflops=gflops,
            achieved_gbps=gbps,
        )
        points.append(p)

    return points


def classify_points(model: RooflineModel, points: List[RooflinePoint]):
    """Classify each point as compute-bound or memory-bound."""
    for p in points:
        if p.arithmetic_intensity <= 0:
            p.bound = "unknown"
            continue

        # Roofline ceiling at this AI
        memory_ceiling = p.arithmetic_intensity * model.peak_bandwidth_gbps
        compute_ceiling = model.peak_gflops
        roofline = min(memory_ceiling, compute_ceiling)

        p.bound = "compute" if memory_ceiling >= compute_ceiling else "memory"


def print_roofline_report(model: RooflineModel, points: List[RooflinePoint]):
    """Print a text-based roofline analysis report."""
    print("=" * 72)
    print("  ROOFLINE MODEL ANALYSIS")
    print("=" * 72)
    print(f"  Platform:          {model.platform}")
    print(f"  Peak compute:      {model.peak_gflops:.1f} GFLOPS")
    print(f"  Peak bandwidth:    {model.peak_bandwidth_gbps:.1f} GB/s")
    print(f"  Ridge point:       {model.ridge_point:.2f} FLOP/byte")
    print("=" * 72)
    print()

    if not points:
        print("  No data points to analyze.")
        return

    # Header
    print(f"  {'Kernel':<40} {'AI':>8} {'GFLOPS':>10} {'%Peak':>8} {'Bound':>10}")
    print("  " + "-" * 78)

    for p in sorted(points, key=lambda x: x.achieved_gflops, reverse=True):
        if p.achieved_gflops <= 0:
            continue

        pct_peak = (p.achieved_gflops / model.peak_gflops * 100
                    if model.peak_gflops > 0 else 0)

        print(f"  {p.name:<40} {p.arithmetic_intensity:>8.2f} "
              f"{p.achieved_gflops:>10.2f} {pct_peak:>7.1f}% {p.bound:>10}")

    print()
    print("  Legend: AI = Arithmetic Intensity (FLOP/byte)")
    print("         %Peak = percentage of theoretical peak GFLOPS")
    print()

    # Summary
    compute_bound = [p for p in points if p.bound == "compute"]
    memory_bound = [p for p in points if p.bound == "memory"]
    print(f"  Summary: {len(compute_bound)} compute-bound, "
          f"{len(memory_bound)} memory-bound, "
          f"{len(points) - len(compute_bound) - len(memory_bound)} unknown")

    # ASCII roofline sketch
    print()
    print("  Roofline (ASCII sketch):")
    print(f"  GFLOPS ^")
    print(f"  {model.peak_gflops:6.0f} |" + "-" * 40 + " (peak)")
    print(f"         |       /")
    print(f"         |     /")
    print(f"         |   /   (bandwidth ceiling)")
    print(f"         | /")
    print(f"       0 +-------------------------------------------> AI (FLOP/byte)")
    print(f"         0       {model.ridge_point:.1f}")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: roofline.py <benchmark_csv> [peak_gflops] [peak_bw_gbps]")
        print()
        print("Example:")
        print("  python3 roofline.py bench_gemm_results.csv 48.0 40.0")
        sys.exit(1)

    csv_path = sys.argv[1]
    peak_gflops = float(sys.argv[2]) if len(sys.argv) > 2 else 48.0
    peak_bw = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0

    model = RooflineModel(
        platform="ARM Neoverse N2",
        peak_gflops=peak_gflops,
        peak_bandwidth_gbps=peak_bw,
    )

    points = build_roofline_points(csv_path)
    classify_points(model, points)
    print_roofline_report(model, points)


if __name__ == "__main__":
    main()
