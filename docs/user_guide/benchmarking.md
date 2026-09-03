# Benchmarking

Use this guide for implementation-regression benchmarks. For application
throughput, follow the batching recipe and tutorial instead.

> - **Prerequisites:** A development installation; see {doc}`Development
>   </user_guide/development>`.
> - **Deep tutorial:** {doc}`02 — GPU Batching with TMol
>   </tutorial/02_gpu_batching>` for application measurements.
> - **Related workflow:** {doc}`GPU batching </workflows/gpu_batching>`.
> - **API reference:** {doc}`Pose </api/pose>` and {doc}`Scoring </api/score>`.
> - **Rosetta mapping:** {doc}`GPU batching and external orchestration
>   </tutorial/rosetta_crosswalk>`.

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
dev/bin/benchmark tmol/tests/score -k cuda-lk_ball-full
```

The wrapper enables pytest benchmarks, prints a summary, and writes JSON results
under `dev/benchmark/`.

## Comparing Revisions

`dev/bin/compare_benchmark` compares benchmark results across revisions. Put
pytest arguments first, then revisions after `--`.

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-lk_ball-full -- origin/master
```

The meta-revision `TREE` means the current working tree:

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-lk_ball-full -- TREE HEAD
```

Ancillary benchmark plots live near the tests as `plot_*.py` scripts.

## Profiling

`dev/bin/profile_benchmark` runs a short pytest benchmark under Nsight Systems
by default:

```bash
dev/bin/profile_benchmark --output profile/ljlk \
  tmol/tests/score -k cuda-ljlk-full
```

Use Nsight Compute when kernel-level counters are needed; arguments after `--`
are forwarded to the profiler. The wrapper uses Nsight Compute's lightweight
`basic` metric set by default:

```bash
dev/bin/profile_benchmark --tool ncu --output profile/ljlk-kernels \
  tmol/tests/score -k cuda-ljlk-forward-100 -- \
  --kernel-name 'regex:.*LJLKPoseScoreDispatch.*forward.*' --launch-count 20
```

Nsight Compute matches the regex against the complete demangled kernel name,
so include leading and trailing `.*` when selecting by a class or method name.

Request the much slower full replay set explicitly with `-- --set=full ...`
when the basic occupancy and throughput counters are insufficient.

The output prefix defaults to `dev/profile/<host>/<UTC timestamp>`. Keep pytest
selectors narrow: profiling every parametrized benchmark produces a very large
trace and makes hardware-counter collection unnecessarily slow. If the wrapper
is not launched from the development environment, pass its interpreter with
`--python /path/to/venv/bin/python` or set `TMOL_PROFILE_PYTHON`.

The output prefix's parent directory must be writable inside the active
container. A path in the checkout or a writable bind mount is usually the most
portable choice. The wrapper disables CPU sampling and context-switch tracing;
its report is therefore usable on restricted compute nodes without profiling
privileges.

For Nsight Compute, the wrapper also keeps the active virtual environment out
of Nsight's own helper-process `PATH`. The benchmark itself still runs with the
selected `--python` interpreter. This prevents Nsight's embedded tooling from
loading incompatible editable-install `.pth` files.

The report includes Python startup, pytest collection, fixture construction,
and benchmark calibration as well as the measured calls. Use the NVTX ranges
and CUDA kernel names to identify steady-state scoring work, and compare timing
with the synchronized pytest-benchmark result rather than the whole process
duration.
