# Scoring and analysis

This guide is a concise reference for whole-pose, autograd, and block-pair
scoring. Use the tutorial for a step-by-step analysis workflow.

> - **Prerequisites:** A prepared `PoseStack`; see {doc}`Quickstart
>   </quickstart>`.
> - **Deep tutorial:** {doc}`03 — Scoring and Analysis
>   </tutorial/03_scoring_and_analysis>`.
> - **Related workflows:** {doc}`Optimization </user_guide/optimization>` and
>   {doc}`Ligand preparation </user_guide/ligands>`.
> - **API reference:** {doc}`Scoring </api/score>` and
>   {doc}`Analysis </api/analysis>`.
> - **Rosetta mapping:** {doc}`Scoring and analysis
>   </tutorial/rosetta_crosswalk>`.

The default high-level preset, `beta2016_score_function()`, is inspired by
Rosetta's beta-November-2016 weights and term set. It does not provide
centroid/full-atom switching, `ref2015`, or numerical parity with Rosetta.
Report weighted totals as TMol score units.

```python
import biotite.structure as struc
import biotite.structure.io
import torch

from tmol.io import pose_stack_from_biotite
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = biotite.structure.io.load_structure(
    "1ubq.cif", model=1, include_bonds=True
)
if isinstance(structure, struc.AtomArrayStack):
    structure = structure[0]
pose_stack = pose_stack_from_biotite(structure, device)

sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
score = scorer(pose_stack.coords)
```

The rendered scoring module is a PyTorch module. It can be called repeatedly
while coordinates change, and its outputs can participate in autograd:

```python
coords = pose_stack.coords.detach().clone().requires_grad_(True)
score = scorer(coords).sum()
score.backward()
```

## CPU batch throughput

Whole-pose CPU scoring parallelizes independent poses through PyTorch's
intra-op thread pool. Set the pool from the cores actually allocated to the
process, and normally do not assign more scoring threads than poses:

```python
torch.set_num_threads(min(n_poses, allocated_physical_cores))
```

The pose boundary keeps gradients race-free, so a one-pose batch does not gain
intra-pose parallelism. When several processes or data-loader workers score at
once, divide the available cores between them to avoid oversubscription.

## Ligand-aware Scoring

When a structure introduces ligand residue types at load time, the score
function must be created from the ligand-extended parameter database:

```python
from tmol.score import beta2016_score_function

sfxn = beta2016_score_function(
    pose_stack.device,
    param_db=context.parameter_database,
)
```

Using the default database for a pose containing newly prepared ligands means
the ligand block type has no scoring parameters in that score function.

## Block-pair Scores

Block-pair scoring reports score contributions between blocks:

```python
block_pair_scorer = sfxn.render_block_pair_scoring_module(pose_stack)
block_pair_scores = block_pair_scorer(pose_stack.coords, sum_terms=False)
```

The result has shape `[n_terms, n_poses, n_blocks, n_blocks]`. Utilities in
`tmol.ops` build masks and summarize common interaction scores,
including cross-mask protein-ligand interaction scores.

By default, `calculate_block_pair_ddg(minimize=True)` Cartesian-minimizes
masked atoms before scoring. For a fixed-coordinate interaction score, pass
`minimize=False` and `pack=False` explicitly:

```python
from tmol.ops import calculate_block_pair_ddg

ddg = calculate_block_pair_ddg(
    pose_stack,
    ligand_mask,
    sfxn=sfxn,
    minimize=False,
    pack=False,
    database=context.parameter_database,
)
```

With both refinement flags disabled as above, this is a fixed-coordinate
interaction-score convention: it sums weighted cross-mask block-pair terms in
one complex and performs no bound/unbound or mutant/reference state
subtraction. Despite the historical `ddg` name, it is not a thermodynamic
binding free energy or delta-delta G.

`pack=True` additionally repacks the masked region and adjacent blocks before
any requested minimization. Use `return_pose_stack=True` when the refined
coordinates are part of the result, and `sum_terms=False` to inspect score
terms separately.

## Fragmented-ligand attribution

For a ligand represented by connected fragment blocks, attribute its
interaction with an explicit partner mask in one connected-pose score:

```python
from tmol.score import calculate_fragment_interactions

fragment_scores = calculate_fragment_interactions(
    pose_stack,
    protein_block_mask,
    sfxn=sfxn,
    sum_terms=False,
)
```

`fragment_scores.scores` has shape
`[n_terms, n_poses, n_fragments]`; set `sum_terms=True` for
`[n_poses, n_fragments]`. The fragment columns follow
`fragment_scores.mapping`. Every pose in the stack must use the same fragment
block layout, and the partner mask must exclude those fragment blocks.

Keep the fragments in one `PoseStack` and call this function once. It renders
one block-pair scorer and reduces all fragment columns together, preserving
autograd through the connected complex. Calling a complete scoring workflow
once per fragment repeats scorer and kernel-launch overhead. Rebuilding
separated fragment poses additionally changes the physical system by removing
the connected multi-block context.
