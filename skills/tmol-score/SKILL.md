---
name: tmol-score
description: >
  Score structures with TMol on CPU or CUDA, including PoseStack construction,
  score-term weights, unweighted decomposition, block-pair analysis, coordinate
  gradients, CPU thread control, batching, and fixed-shape CUDA graph replay.
  Do not use for side-chain packing or full FastRelax protocols.
allowed-tools: Bash, Read, Write, AskUserQuestion
license: Apache-2.0
---

# TMol scoring

Build a scorer whose device, chemical database, and pose layout agree, then
choose summed, decomposed, differentiable, or captured execution based on the
requested output. TMol scores are TMol score units, not kcal/mol and not
numerically interchangeable with Rosetta scores.

This workflow requires a working TMol installation. Prepared ligand chemistry
also requires Biotite and the extended `ParameterDatabase` returned by pose
construction.

## Minimal workflow

```python
import torch
from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose = pose_stack_from_pdb("input.pdb", device)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose)
scores = scorer(pose.coords)
```

Render the scorer again after changing the pose's block or atom layout. A
coordinate-only update with the same layout can reuse it.

For ligands prepared through `pose_stack_from_biotite(...,
return_context=True)`, construct the score function with
`context.parameter_database`. Read `docs/workflows/structure_io.md` and
`docs/user_guide/ligands.md` before substituting a direct PDB path for
noncanonical chemistry.

## Gradients and score terms

```python
coords = pose.coords.detach().clone().requires_grad_(True)
total = scorer(coords).sum()
total.backward()
gradient = coords.grad

unweighted = scorer(coords.detach(), sum_terms=False, apply_weights=False)
weighted = scorer(coords.detach(), sum_terms=False, apply_weights=True)
```

Changing or zeroing individual score-function weights remains supported. Do not
replace decomposed requests with a summed fused result: callers asking for
`sum_terms=False` or `apply_weights=False` require canonical term lanes.

Use `render_block_pair_scoring_module()` for block-by-block attribution. Treat
historically named `ddg` helpers as interaction-score conventions unless the
workflow explicitly constructs the required physical reference states.

## CPU execution

Inspect both process affinity and the active PyTorch budget. Set the budget
before rendering the scorer:

```python
import os
import torch

available = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
print(available, torch.get_num_threads())
torch.set_num_threads(8)
```

TMol uses the PyTorch value as a ceiling and may use fewer workers on small
inputs. Divide cores between concurrent processes. Read
`docs/user_guide/cpu_threading.md` for scheduler and environment examples.

## CUDA execution

Batch similarly sized poses to limit padding. Synchronize before and after
timed CUDA regions. For repeated default weighted scoring with a fixed shape,
dtype, device, and scorer, capture replay with:

```python
graphed = sfxn.render_whole_pose_scoring_module(pose, cuda_graph="forward")
with torch.no_grad():
    scores = graphed(pose.coords)
```

Use `cuda_graph="forward_backward"` when the repeated workload includes
coordinate gradients. Graph replay reduces launch overhead but is not required
for compact scoring or packing memory. Its output buffers are reused; clone an
output that must survive the next replay. Keep eager execution for changing
shapes or unsupported dynamic behavior.

## Validate

Check finite totals and, when requested, finite coordinate gradients. For an
optimization or implementation comparison, also compare decomposed weighted
terms, not only a final total that could hide compensating errors.

Primary references: `docs/user_guide/scoring.md`,
`docs/user_guide/cpu_threading.md`, `docs/workflows/gpu_batching.md`, and
`docs/api/score.rst`.
