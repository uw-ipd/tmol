# Small-molecule prep scripts

> **Deprecated:** the manual three-step shell pipeline moved to
> [`deprecated/`](./deprecated/). Prefer the Python API in `tmol/ligand/`:
> `prepare_ligand_from_mol2` / `prepare_ligand_from_smiles` and
> `write_params_from_mol2(mol2, out_params)`.

The original manual pipeline (now under `deprecated/`):

1. `deprecated/prepare_smiles.sh` # protonate smiles using dimorphite-dl
2. `deprecated/run_obabel.sh` # generate mol2 with 3d structure + mmff94 charges
3. `deprecated/run_params.sh` # generate Rosetta params from mol2 files

`build_dud80_ground_truth.sh` (the parity ground-truth generator) still uses
these tools via absolute paths.
