# Ligand CIF fixtures (CIF -> dG path)

Small single-ligand CIFs used by `tmol/tests/ligand/test_cif_to_dg.py` to
exercise the unified CIF/atom-array -> SMILES -> params -> score path.

Each ligand is provided in two variants to cover both ingestion shapes:

- `*.bonds_present.cif` — carries an explicit `_chem_comp_bond` block
  (exercises the existing-bonds SMILES branch).
- `*.bonds_absent.cif` — atom-site records only, no `_chem_comp_bond` block
  (exercises the CCD-template branch; biotite re-infers intra-residue bonds
  from the CCD by residue code on load).

## Provenance

Ligand heavy-atom coordinates were extracted from PLINDER system structures
(`/net/scratch/ncorley/plinder`, PLINDER 2024-06 v2), one residue instance per
ligand:

| File stem | CCD code | Source system | Description |
|-----------|----------|---------------|-------------|
| `vww`     | `VWW`    | `10gs` (PDB 1GS, glutathione transferase) | S-benzyl glutathione, C/N/O/S, 33 heavy atoms |
| `sah`     | `SAH`    | `10mh` (PDB 1MHT, DNA methyltransferase)   | S-adenosyl-L-homocysteine, C/N/O/S, 26 heavy atoms |
