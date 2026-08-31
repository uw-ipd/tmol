# Benchmarking Guide

# Running specific benchmarks:

To run select benchmarks, utilize the `dev/bin/benchmark` wrapper with the
pytest `-k` flag to select target tests. This enables benchmarks, prints
a textual summary and outputs a benchmark result json under
`dev/benchmark/...`.

Example:

  * `dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball`

# Comparing benchmarks:

The `dev/bin/compare_benchmark` performs local comparative benchmarking by
reseting the working tree to each target version, running a benchmark
pass, and then comparing the results.

`compare_benchmark` accepts pytest args and target revisions, separated by
`--`. The meta-revision `TREE` can be used to indicate the state of the
working tree, ie to compare uncommitted changes vs `HEAD`.

Example:

  * `dev/bin/compare_benchmark tmol/test/score -k cuda-full-lk_ball -- origin/master`
     Compare current tree vs `origin/master`.
  * `dev/bin/compare_benchmark tmol/test/score -k cuda-full-lk_ball -- feature_branch master`
     Compare `feature_branch` vs `master`.
  * `dev/bin/compare_benchmark tmol/test/score -k cuda-full-lk_ball`
     Compare working tree vs HEAD.

Ancillary benchmark plots can be generated via `plot_*.py` scripts checked
into the `tmol/tests` tree. See `.buildkite/bin/benchmark` for invocation
details.

# Remote benchmarks:

The CI service runs full benchmark passes and stores benchmark results for
later analysis. Use the `.buildkite/bin/fetch_buildkite_benchmark` script
to retrieve benchmark results for the most recent CI run for a target
branch. This script will fetch the result json and output the filetime to
stdout. See `.buildkite/bin/benchmark` for invocation details.


# Profiling benchmarks:

`dev/bin/profile_benchmark` captures a timeline with Nsight Systems by default,
or hardware counters with Nsight Compute when passed `--tool ncu`. Use
`--output <prefix>` to choose the report path. Arguments after `--` are sent to
the profiler; other arguments are pytest selectors.

Example:
  `dev/bin/profile_benchmark --output profile/lk-ball tmol/tests/score -k cuda-full-lk_ball`
