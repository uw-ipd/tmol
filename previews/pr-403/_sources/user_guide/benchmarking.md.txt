# Benchmarking

Performance work happens at two different levels:

- Application-level GPU batching measures a real workload over a `PoseStack`;
  use the {doc}`GPU batching workflow </workflows/gpu_batching>` for batch
  construction, synchronized timing, memory measurement, and chunking.
- The developer benchmark harness runs pytest benchmark cases to detect kernel
  or implementation regressions across code revisions.

The harness is not an application scheduler and its microbenchmark results do
not choose a production batch size.

## Running Developer Benchmarks

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

`dev/bin/profile_benchmark` wraps the legacy `nvprof` command and the tmol
pytest CUDA-profile hook:

```bash
dev/bin/profile_benchmark tmol/tests/score -k cuda-full-lk_ball -- -o profile.nvvp
```

Use it only in a CUDA environment that still provides `nvprof`; current NVIDIA
toolkits may require a separate Nsight-based profiling workflow instead.

## Related example

{doc}`GPU Batching with TMol </tutorial/02_gpu_batching>` demonstrates
application-level latency, throughput, allocator memory, heterogeneous padding,
and chunking. Use the developer harness above when the question is whether a
specific tmol implementation changed performance.
