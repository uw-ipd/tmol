# Development tools

The [benchmarking guide](../docs/user_guide/benchmarking.md) describes local
measurements, isolated revision comparisons, result files, and Nsight profiling.
Run the tools from the repository root in an activated development environment:

```bash
dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball
dev/bin/compare_benchmark tmol/tests/score -k cuda-full-lk_ball -- TREE HEAD
dev/bin/profile_benchmark --output /tmp/lk-ball tmol/tests/score -k cuda-full-lk_ball
```

The comparison command builds each revision separately and preserves the
caller's checkout. Benchmark JSON and logs are kept after failures. Use the
current [GitHub Actions workflows](../.github/workflows) for CI; the historical
Buildkite retrieval scripts and site-specific cluster launcher are retired.
Profiling uses Nsight Systems or Nsight Compute; the old nvprof trace converter
is retired as well.

## Numerical derivations

These notebooks document formulas used by the implementation. They are
developer references, rather than executable installation tutorials:

- [Cubic Hermite interpolation](cubic_hermite_interpolation.ipynb)
- [LK-ball derivatives](lk_ball_derivatives.ipynb)

For maintained modeling examples, see the [tutorial gallery](../docs/examples_index.rst).
