# Benchmarking

tmol keeps benchmark helpers under `dev/bin`.

## Running Benchmarks

Use `dev/bin/benchmark` with pytest selectors:

```bash
dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball
```

The wrapper enables pytest benchmarks, prints a summary, and writes JSON results
under `dev/benchmark/`.

## Comparing Revisions

`dev/bin/compare_benchmark` compares benchmark results across revisions. Put
pytest arguments first, then revisions after `--`.

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-full-lk_ball -- origin/master
```

The meta-revision `TREE` means the current working tree:

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-full-lk_ball -- TREE HEAD
```

Ancillary benchmark plots live near the tests as `plot_*.py` scripts.

## Profiling

`dev/bin/profile_benchmark` runs the benchmark once for timing, then runs under
NVIDIA profiling tools for trace capture:

```bash
dev/bin/profile_benchmark tmol/tests/score -k cuda-full-lk_ball -- -o profile.nvvp
```

Open the trace with NVIDIA Visual Profiler or a compatible CUDA profiling UI.
