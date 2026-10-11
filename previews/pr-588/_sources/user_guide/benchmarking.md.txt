# Benchmarking

Use the developer harness to measure implementation regressions. For application
throughput, batch size, and memory use, see {doc}`GPU batching
</workflows/gpu_batching>`. Both require a {doc}`development installation
<development>`.

## Run benchmarks

Use `dev/bin/benchmark` with pytest selectors:

```bash
dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball
```

The wrapper enables pytest benchmarks, prints a summary, and writes JSON results
under `dev/benchmark/`.

## Compare revisions

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

`dev/bin/profile_benchmark` runs a short pytest benchmark under Nsight Systems
by default:

```bash
dev/bin/profile_benchmark --output profile/ljlk \
  tmol/tests/score -k cuda-full-ljlk
```

Use Nsight Compute when kernel-level counters are needed; arguments after `--`
are forwarded to the profiler:

```bash
dev/bin/profile_benchmark --tool ncu --output profile/ljlk-kernels \
  tmol/tests/score -k cuda-forward-ljlk-100 -- \
  --kernel-name regex:ljlk --launch-count 20
```

The output prefix defaults to `dev/profile/<host>/<UTC timestamp>`. Keep pytest
selectors narrow: profiling every parametrized benchmark produces a very large
trace and makes hardware-counter collection unnecessarily slow. If the wrapper
is not launched from the development environment, pass its interpreter with
`--python /path/to/venv/bin/python` or set `TMOL_PROFILE_PYTHON`.
