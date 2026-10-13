# Quickstart

Load a protein, inspect its score, change its side chains, and relax it.
The examples share the same `pose`, `device`, and `sfxn`.
For an interactive introduction, try the {doc}`score playground <playground>`.

## Install and load a protein

```bash
python -m pip install tmol --only-binary=tmol
curl -L https://files.rcsb.org/download/1UBQ.pdb -o 1ubq.pdb
```

This installs CPU-only TMol. For CUDA, follow {doc}`installation` and use
`torch.device("cuda")` below.

```python
import torch
from tmol.io import pose_stack_from_pdb, write_pose_stack_pdb
from tmol.score import beta2016_score_function

device = torch.device("cpu")
pose = pose_stack_from_pdb("1ubq.pdb", device)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose)
```

A {ref}`PoseStack <glossary-posestack>` holds one or more molecular systems.
The rendered scorer is reusable while only their coordinates change.

## Score and inspect contributions

```python
score = scorer(pose.coords)
contributions = scorer(pose.coords, sum_terms=False)

print("Total:", score.tolist())  # one value per pose
for term, values in zip(sfxn.all_score_types(), contributions):
    print(term.name, values.tolist())
```

The contributions already include the score-function weights. They sum to the
total. See {ref}`score terms <glossary-score-function>` for what each measures.

To differentiate the score with respect to atom coordinates:

```python
coords = pose.coords.detach().clone().requires_grad_(True)
scorer(coords).sum().backward()
print(coords.grad.shape)  # [n_poses, n_atoms, 3]
```

## Minimize coordinates

```python
from tmol.optimization import run_cart_min

minimized = run_cart_min(pose, sfxn)
print("Before:", scorer(pose.coords).tolist())
print("After:", scorer(minimized.coords).tolist())
write_pose_stack_pdb(minimized, "minimized.pdb")
```

This optimizes atom positions continuously. Pass `coord_mask` to restrict
movement to selected atoms; see {doc}`masks and constraints <user_guide/optimization>`.

## Repack side chains

{ref}`Repacking <glossary-repacking>` searches side-chain conformations while
keeping the amino-acid sequence fixed.

```python
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.database import ParameterDatabase

task = PackerTask(pose, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(ParameterDatabase.get_default(), device)
)
task.add_conformer_sampler(FixedAAChiSampler())
task.add_conformer_sampler(IncludeCurrentSampler())
packed = pack_rotamers(pose, sfxn, task)

packed_scorer = sfxn.render_whole_pose_scoring_module(packed)
print(packed_scorer(packed.coords).tolist())
```

Render a new scorer after packing because the output atom layout can change.
See the {doc}`packing notebook <tutorial/04_packing_and_mutation_scan>` for
fixed residues, mutation scans, and residue-level results.

## Combine packing and minimization

{ref}`FastRelax <glossary-fastrelax>` alternates the two operations while
ramping steric repulsion back to its full weight.

```python
from tmol.kinematics import CartesianMoveMap, FoldForest
from tmol.relax import fast_relax

relaxed = fast_relax(
    pose,
    sfxn,
    PackerPalette(),
    CartesianMoveMap(),
    FoldForest.reasonable_fold_forest(pose),
)
relaxed_scorer = sfxn.render_whole_pose_scoring_module(relaxed)
print(relaxed_scorer(relaxed.coords).tolist())
write_pose_stack_pdb(relaxed, "relaxed.pdb")
```

This uses Cartesian minimization with all atom coordinates free. For torsion
or jump control, use a configured `MoveMap` and kinematic minimizer as shown in
{doc}`kinematic optimization <user_guide/optimization>`.

## Score a batch

```python
from tmol.pose import PoseStackBuilder

batch = PoseStackBuilder.from_poses([pose, minimized], device)
batch_scorer = sfxn.render_whole_pose_scoring_module(batch)
with torch.no_grad():
    scores = batch_scorer(batch.coords)
print(scores.tolist())
```

Group similarly sized systems to reduce padding. The same code runs on CUDA
when the inputs and score function are on that device.
See the {doc}`batching notebook <tutorial/02_gpu_batching>` for timing and memory.

## Load a protein–ligand complex

Use mmCIF to retain ligand bond information, and score with the extended
parameter database returned during preparation:

```python
from tmol.io import atom_array_from_file, pose_stack_from_biotite

structure = atom_array_from_file("complex.cif")
complex_pose, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    param_db=ParameterDatabase.get_default(),
    return_context=True,
)
complex_sfxn = beta2016_score_function(device, param_db=context.parameter_database)
complex_scorer = complex_sfxn.render_whole_pose_scoring_module(complex_pose)
print(complex_scorer(complex_pose.coords).tolist())
```

Continue with the {doc}`ligand notebook <tutorial/07_ligand_and_params>` or
{doc}`DNA/RNA notebook <tutorial/08_nucleic_acids>` for those systems.

## Further workflows

{doc}`Tutorials <examples_index>` are runnable notebooks with plots and exercises.
The {doc}`workflow reference <workflows/index>` covers preparation options,
interaction scores, constraints, threading, and extension points.

```{toctree}
:hidden:

workflows/index
```
