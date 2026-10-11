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

Run the wrapper from an activated development environment with the required
build dependencies, PyTorch, and `tmol[dev]` dependencies installed. Each revision
gets an isolated source checkout, virtual environment, native build, and JIT
cache. Installed dependencies are shared from the calling environment, but its
editable-install hooks are not loaded. Dependency changes between revisions
therefore require a separate matched dependency environment.

`TREE` includes staged and unstaged tracked edits, tracked deletions, and
non-ignored untracked files. Ignored build products are excluded. The caller's
branch, index, local changes, and stash are never modified. Revisions run in the
requested order; reverse the order on a second run to check for machine drift.
Without revision arguments the comparison is `TREE HEAD`; one revision argument
compares `TREE` with that revision.

The tool prints a results directory containing per-revision JSON, build/test
logs, a source manifest, and `comparison.txt`. These files remain available after
failure; temporary source checkouts and environments are removed. Choose a new
output directory with `--benchmark-outdir` before the pytest selectors:

```bash
dev/bin/compare_benchmark --benchmark-outdir /tmp/tmol-comparison \
  tmol/tests/score -k cuda-full-lk_ball -- TREE HEAD
```

Build and test failures stop the comparison with a nonzero exit status. The
manifest records commit IDs and a digest of `TREE` edits; logs include the
installed dependencies and PyTorch build configuration. Keep input, dtype,
thread counts, device, and benchmark settings fixed between revisions. CUDA
benchmarks must warm up and synchronize timed work. These rebuilds can take
substantial time; use the ordinary benchmark wrapper for repeated runs of an
already-built checkout.

Ancillary benchmark plots live near the tests as `_plot_*.py` scripts.

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
