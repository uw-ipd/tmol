# Skill Card

## Description

`tmol-score` constructs compatible poses and score functions, evaluates totals
or term decompositions, computes coordinate gradients, and selects CPU
threading, CUDA batching, or fixed-shape graph replay.

## Owner and license

TMol maintainers. Apache-2.0; see the repository `LICENSE`.

## Requirements

A working TMol installation and an input whose chemistry can be represented by
the selected parameter database. CUDA is optional.

## Risks and mitigations

- TMol and Rosetta score values are not numerically interchangeable; the skill
  labels units and comparisons accordingly.
- A ligand pose paired with the default database can omit prepared parameters;
  the skill preserves and reuses the build context's database.
- Asynchronous CUDA timing and CPU oversubscription can produce misleading
  results; the skill requires synchronization and records the active thread
  budget.
- CUDA graph outputs are reused. The skill calls this out before downstream code
  retains a result across replays.

## References and output

References: `docs/user_guide/scoring.md`, `docs/user_guide/cpu_threading.md`,
`docs/workflows/gpu_batching.md`, and `docs/api/score.rst`.

Output: finite per-pose scores, optional canonical score-term lanes, and optional
coordinate gradients. The workflow reads structures and does not modify them
unless the user separately requests output files.

## Skill version

0.1.0 (2026-09)
