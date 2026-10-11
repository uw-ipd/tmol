# Working with TMol

TMol scores, repacks, and relaxes molecular structures with PyTorch on CPU or
CUDA. Prefer its public APIs and existing workflows before adding custom code.

## Find the right workflow

- Install or troubleshoot: [installation](docs/installation.md) and
  [setup skill](skills/tmol-setup/SKILL.md).
- Load structures, score, or differentiate: [quickstart](docs/quickstart.md)
  and [scoring skill](skills/tmol-score/SKILL.md).
- Repack, design, minimize, or relax: [packing and relax skill](skills/tmol-pack-relax/SKILL.md).
- Prepare ligands or noncanonical chemistry: [structure I/O](docs/workflows/structure_io.md)
  and [ligand preparation](docs/user_guide/ligands.md).
- Runnable examples: [tutorial index](docs/examples_index.rst).

## Start small

```python
import torch
from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cpu")  # Use CUDA with a matching CUDA-enabled installation.
pose = pose_stack_from_pdb("input.pdb", device)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose)
scores = scorer(pose.coords)  # One weighted total per pose, in TMol score units.
```

- Keep the pose, score function, and tensors on the same device. Check finite
  scores and, when needed, finite coordinate gradients.
- For prepared ligands, use the returned `context.parameter_database` when
  constructing the score function. Preserve atom identities and input chemistry.
- Reuse a rendered scorer for coordinate changes; render again after changing
  types, connectivity, atom layout, or batch layout, including after packing.
- Use `sum_terms=False` for weighted term contributions and additionally
  `apply_weights=False` for unweighted values.
- Repacking preserves sequence only when the task restricts it. Set fixed
  residues and movement masks explicitly; a new `MoveMap` enables no movement.
  Constraints need both pose attachment and a nonzero score weight.
- Batch similarly sized structures. Synchronize CUDA timings and record device,
  versions, seeds, thread counts, and peak memory for reproducible comparisons.

## Changing TMol itself

Read [the development skill](skills/tmol-develop/SKILL.md) before implementation
work; it covers native builds, focused CPU/CUDA tests, and matched benchmarks.
Python code lives under `tmol/`, native kernels alongside their modules, and
tests under `tmol/tests/`. Keep public APIs and scientific behavior intact.
Use a source checkout with development dependencies and any required test
extensions; runtime wheels need not contain tests or fixtures.

Run the affected tests and `git diff --check`; use the repository's pre-commit
checks, and build docs for changed examples or public behavior. Performance
changes need correctness checks and matched measurements of time and memory,
not just fewer operations. Keep temporary benchmarks and build artifacts out
of production commits, and state validation limitations explicitly.
