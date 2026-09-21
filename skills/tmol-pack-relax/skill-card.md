# Skill Card

## Description

`tmol-pack-relax` configures side-chain/nucleic conformer sampling, coordinate
or kinematic minimization, constraints, and FastRelax while keeping large CPU or
CUDA packing workloads within bounded native work units.

## Owner and license

TMol maintainers. Apache-2.0; see the repository `LICENSE`.

## Requirements

A working TMol installation, a prepared `PoseStack`, and a score function made
from the same parameter database. CUDA is optional.

## Risks and mitigations

- Repacking and design have different identity semantics. The skill never treats
  `restrict_to_repacking()` as a design task.
- Packing can change topology, invalidating a rendered scorer. The skill requires
  rerendering after layout changes.
- Large batches can exceed memory or native dispatch limits. The skill relies on
  automatic chunking, permits a lower explicit bound, and retains checked native
  failures.
- Relax is stochastic and scores are modeling objectives, not experimental
  validation. The skill seeds regression runs and reports that limitation.

## References and output

References: `docs/workflows/packing.md`, `docs/user_guide/optimization.md`,
`docs/workflows/nucleic_acids.md`, and `docs/api/relax.rst`.

Output: a new packed, minimized, or relaxed `PoseStack`, plus validation of task
masks, finite coordinates/scores, and relevant memory/timing metadata.

## Skill version

0.1.0 (2026-09)
