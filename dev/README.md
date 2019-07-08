# Benchmarking Guide

# Running specific benchmarks:

Run benchmarks through the 
To run a single benchmark, utilize the pytest `-k` flag to select a target
test. This runs a benchmark and outputs a benchmark result json under
`dev/bin/benchmark/...`.

Example:
  `dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball`

# Profiling benchmarks:

Benchmarks can be under under the nvprof profiler to capture perf traces
for analysis. This runs the benchmark *without* profiling for timing
information then executes a single run under the profiler for tracing. Use
the nvprof `-o <outfile>` option to capture a trace for visual inspection.

Example:
  `dev/bin/profile_benchmark tmol/tests/score -k cuda-full-lk_ball -- -o profile.nvvp`

Then view via nvidia visual profiler:

  `/usr/local/cuda/bin/nvvp profile.nvvp`
