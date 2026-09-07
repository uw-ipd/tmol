# Skill Card

## Description

`tmol-develop` guides focused TMol Python/C++/CUDA changes from a clean worktree
through correctness tests, matched speed/memory measurements, profiling, code
quality, and a reviewable integration handoff.

## Owner and license

TMol maintainers. Apache-2.0; see the repository `LICENSE`.

## Requirements

A TMol source checkout and its development toolchain. CUDA work additionally
requires a compatible toolkit and GPU. Cluster submission, remote pushes, and
merges require explicit user authority.

## Risks and mitigations

- Stale JIT binaries can invalidate A/B conclusions; the skill isolates caches
  by revision.
- Asynchronous CUDA execution and CPU oversubscription can bias timing; the
  skill records execution settings and synchronizes measured CUDA regions.
- Narrow microbenchmarks can hide end-to-end regressions; the skill expands from
  focused tests to representative workflows and subsystem gates.
- Profiling artifacts can bloat PRs; the skill keeps one-off evidence outside
  the production branch.

## References and output

References: `docs/user_guide/development.md`,
`docs/user_guide/benchmarking.md`, `docs/contributor_guide.md`, and CI workflow
files.

Output: a focused source/test/docs diff, reproducible validation commands,
matched benchmark evidence when relevant, and a clean branch suitable for
review. External mutations remain subject to explicit authorization.

## Skill version

0.1.0 (2026-09)
