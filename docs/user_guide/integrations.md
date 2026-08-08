# Integrations

tmol is meant to sit inside PyTorch-based structural biology workflows. It can
score structures loaded from standard files and convert outputs from structure
prediction systems into `PoseStack` objects.

## RosettaFold2

Install tmol into the RosettaFold2 environment:

```bash
cd <tmol repo root>
pip install -e .
```

Convert model outputs into tmol:

```python
pose_stack = tmol.pose_stack_from_rosettafold2(seq[0], xyz[0], chainlens[0])
```

When minimizing inside an RF2 workflow, make sure gradients are enabled:

```python
torch.set_grad_enabled(True)
```

RF2 inference code often disables gradients globally.

## OpenFold

OpenFold outputs can be converted with:

```python
pose_stack = tmol.pose_stack_from_openfold(output)
```

## Biotite and AtomArray

The preferred path for rich structure IO is Biotite `AtomArray`:

```python
import biotite.structure as struc
import biotite.structure.io

structure = biotite.structure.io.load_structure(
    "complex.cif",
    model=1,
    include_bonds=True,
)
if isinstance(structure, struc.AtomArrayStack):
    structure = structure[0]

pose_stack = pose_stack_from_biotite(structure, device)
```

Biotite is especially important for ligands because tmol uses explicit bond
tables from the `AtomArray` during ligand preparation.
