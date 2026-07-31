# Scoring

The default high-level score function is `beta2016_score_function()`, tmol's
implementation of the Rosetta `beta_nov2016` all-atom score function.

```python
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_stack = pose_stack_from_pdb("1ubq.pdb", device)

sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
energy = scorer(pose_stack.coords)
```

The rendered scoring module is a PyTorch module. It can be called repeatedly
while coordinates change, and its outputs can participate in autograd:

```python
coords = pose_stack.coords.detach().clone().requires_grad_(True)
energy = scorer(coords).sum()
energy.backward()
```

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

Block-pair scoring reports pair energies between residues or ligands:

```python
block_pair_scorer = sfxn.render_block_pair_scoring_module(pose_stack)
block_pair_scores = block_pair_scorer(pose_stack.coords, sum_terms=False)
```

The result has shape `[n_terms, n_poses, n_blocks, n_blocks]`. Utilities in
`tmol.score.score_utils` build masks and summarize common interaction scores,
including protein-ligand ddG-style block-pair energies.

```python
from tmol.score.score_utils import calculate_block_pair_ddg

ddg = calculate_block_pair_ddg(
    pose_stack,
    ligand_mask,
    sfxn=sfxn,
    minimize=False,
    pack=False,
    database=context.parameter_database,
)
```

Set `sum_terms=False` to inspect score terms separately.
