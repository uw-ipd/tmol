# Deprecated: manual small-molecule prep scripts

These three shell scripts were the original manual pipeline for turning a
SMILES into a Rosetta `.params` file:

1. `prepare_smiles.sh` — dimorphite-dl protonation
2. `run_obabel.sh` — openbabel 3D + MMFF94 mol2
3. `run_params.sh` — Rosetta `mol2genparams.py`

They are **superseded** by the Python pipeline in `tmol/ligand/`:

- `tmol.ligand.prepare_ligand_from_mol2(mol2)` / `prepare_ligand_from_smiles(smi)`
  prepare a ligand end-to-end.
- `tmol.ligand.write_params_from_mol2(mol2, out_params)` writes a Rosetta
  `.params` file directly.

They are kept here (not deleted) for historical reference and because the
ground-truth dataset was generated with this exact protocol (see
`scripts/ligand_prep/build_dud80_ground_truth.sh`, which still uses them via
absolute tool paths). Prefer the Python API for new work.
