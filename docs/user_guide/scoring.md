# Scoring and analysis

Calculate whole-pose scores, coordinate gradients, and block-pair contributions.

`beta2016_score_function()` provides the default all-atom weights and terms.

```python
import biotite.structure as struc
import biotite.structure.io
import torch

from tmol.io import atom_array_from_file, pose_stack_from_biotite
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = atom_array_from_file("1ubq.cif")
pose_stack = pose_stack_from_biotite(structure, device)

sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
score = scorer(pose_stack.coords)
```

Reuse the scorer while coordinates change. Render a new one after changing
block types, connectivity, or batch layout. To calculate coordinate gradients:

```python
coords = pose_stack.coords.detach().clone().requires_grad_(True)
score = scorer(coords).sum()
score.backward()
```

## Fitting to a density map

`beta_nov16_dens_score_function()` is `beta2016_score_function()` plus the
`elec_dens_fast` fit-to-density term at weight 35, with the per-residue side-chain
density scale of Rosetta's cryo-EM refinement (in `beta_nov16_dens.sfxn`). The
term scores against the map attached to the pose stack; every pose in a stack
shares one map. Map files do not record their resolution, so `read_mrc` takes it.

```python
import attr
from tmol.score import beta_nov16_dens_score_function
from tmol.score.density import read_mrc

density_map = read_mrc("map.mrc", resolution=3.2, device=device)
pose_stack = attr.evolve(pose_stack, density_map=density_map)
sfxn = beta_nov16_dens_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
```

Each heavy atom scores `-(trunc(a_elt) / 6) * S(x)`, where `S` is the observed map
correlated with one atom-sized Gaussian, normalized, and read by a cubic B-spline
whose coefficients are the grid values, as in Rosetta. An atom therefore sees the
map only within the kernel radius, about 3.5 A for a 3.7 A map. The score is
differentiable in the coordinates and works for whole-pose, block-pair and rotamer
scoring.

`read_mrc` reads cryo-EM and crystallographic MRC/CCP4 maps, including skewed
cells; a crystal map covering less than the unit cell is expanded with its
symmetry operators. The defaults suit cryo-EM: electron scattering factors and
an open (non-periodic) map. Two independent score-function options change them;
a crystallographic map typically wants both:

```python
sfxn.set_option("density_scatterers", "xray")
sfxn.set_option("density_periodic", True)
```

## CPU batch throughput

Whole-pose CPU scoring follows PyTorch's process-wide intra-op thread budget.
Inspect the active value and, when needed, override it before rendering a
scorer:

```python
print(torch.get_num_threads())
torch.set_num_threads(8)
```

TMol parallelizes independent poses and score terms and can shard the dominant
pair traversal for a single pose. Small workloads may use fewer threads when
additional shards would cost more than they save. When several processes or
data-loader workers score at once, divide the available cores between them to
avoid oversubscription. See {doc}`CPU threading </user_guide/cpu_threading>`
for affinity-aware discovery, scheduler examples, launch-time overrides, and
benchmarking guidance.

## Scoring prepared ligands

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

## Block-pair scores

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

With both flags disabled, the result sums weighted cross-mask interactions in
one fixed complex.

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

Keep fragments connected in one `PoseStack`. This call scores once, reduces
all fragment columns, and preserves autograd. Scoring separate fragment poses
adds overhead and changes the system by removing inter-fragment connections.

## Examples and reference

{doc}`Scoring tutorial </tutorial/03_scoring_and_analysis>` · {doc}`Scoring API </api/score>` · {doc}`Score terms </api/score_terms>`
