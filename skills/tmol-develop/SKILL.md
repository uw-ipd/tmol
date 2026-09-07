---
name: tmol-develop
description: >
  Develop, test, benchmark, and profile TMol Python/C++/CUDA changes. Use for
  editable builds, focused and full CPU/CUDA validation, performance or memory
  regressions, launch profiling, code quality, and preparing a reviewable PR.
  Do not use for ordinary installed-library modeling workflows.
allowed-tools: Bash, Read, Write, AskUserQuestion
license: Apache-2.0
---

# TMol development

Make the smallest production change supported by correctness tests and matched
measurements. Preserve unrelated work in a dirty checkout; use a clean worktree
from the target base for integration or final A/B validation.

This workflow requires a TMol source checkout, Python 3.11+, CMake, Ninja, and a
C++ compiler. CUDA development additionally requires `nvcc` and a supported GPU.

## Build for the task

Use an editable source build and include test-only native extensions when the
affected tests need them:

```bash
TMOL_DISABLE_WHEEL_FETCH=1 python -m pip install --no-build-isolation -e ".[dev]" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON
```

For kernel iteration, `TMOL_USE_JIT=1` is faster to turn around. Use a distinct
`TORCH_EXTENSIONS_DIR` per compared source revision so one candidate cannot
reuse another revision's binary. Record compiler, PyTorch, CUDA, architecture,
and source commit.

## Test from narrow to broad

1. Run the smallest unit/regression test that exercises the changed invariant.
2. Run the affected subsystem, on CPU and CUDA when templates or dispatch code
   are shared.
3. Run `tmol/tests/optimization`, `tmol/tests/pack`, and `tmol/tests/score` for
   changes crossing those boundaries.
4. Run the repository's complete CI-equivalent lanes before merge when risk
   warrants it.

```bash
pytest -q tmol/tests/score/test_score_function.py
pytest -q tmol/tests/optimization tmol/tests/pack tmol/tests/score
pre-commit run --all-files --show-diff-on-failure
```

If a format command changes files, inspect the diff rather than committing a
mechanical rewrite blindly. Build docs when public behavior, APIs, or examples
change.

## Benchmark correctly

Read `docs/user_guide/benchmarking.md` before collecting evidence. Use
`dev/bin/compare_benchmark` for matched revisions and `TREE` for the current
worktree. For CUDA, warm up and synchronize around timed work. For CPU, record
affinity, `torch.get_num_threads()`, process count, and the one-thread baseline.

Keep the input, device, dtype, warmup, sample count, process order, and JIT/AOT
mode identical across variants. Reverse process order or alternate variants
when system drift could rival the claimed effect. Report distributions and
correctness checks, not only the best sample.

Measure peak allocated memory around the operation separately from import and
pose setup. CUDA Graph capture and compact data representation are independent:
attribute improvements to the mechanism actually changed.

## Profile before optimizing

Use `dev/bin/profile_benchmark` with a narrow pytest selector. Nsight Systems
answers orchestration and launch-count questions; Nsight Compute answers
kernel occupancy, instruction, and memory-traffic questions. On CPU, use an
operator profile plus native sampling where Python/Torch attribution is
insufficient.

Do not optimize launch count as a proxy for latency. Accept an optimization only
when a representative matched A/B is favorable or neutral in the important
regimes and the implementation remains understandable.

## Integration and handoff

- Keep one-off harnesses, raw traces, plots, build caches, and benchmark results
  outside the production branch unless they are established repository tools.
- Commit production code, focused regression tests, and user-facing docs.
- Preserve public fallbacks such as unweighted/decomposed scoring, term removal,
  custom weights, device choice, and changing input layouts.
- Run `git diff --check`, inspect the final file list, and confirm the branch is
  based on the intended upstream commit.
- Push or merge only when the user authorized the external action and required
  checks/reviews permit it.

Primary references: `docs/user_guide/development.md`,
`docs/user_guide/benchmarking.md`, `docs/contributor_guide.md`, and
`.github/workflows/ci.yml`.
