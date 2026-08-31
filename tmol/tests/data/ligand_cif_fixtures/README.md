# Ligand CIF fixtures

Small CIFs exercising the ligand path from a file rather than from an atom
array built in a test.

## CIF -> dG path

Single-ligand CIFs used by `tmol/tests/ligand/test_cif_to_dg.py` to exercise
the unified CIF/atom-array -> SMILES -> params -> score path.

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

## Covalent detection

`glycan_nag_pair.cif` is two N-acetylglucosamines a glycosidic bond apart
(1.4 A) with no `struct_conn` block, used by
`tmol/tests/ligand/test_ligand_pipeline.py`.

A glycan is a branched entity rather than a polymer one, so the file numbers
none of its residues along a sequence and every `label_seq_id` is `.`. What
says the residues link is the type the file declares for them, which is what
the spatial pass is gated on: read with those types the pair is flagged as
covalently linked, read without them nothing is claimed, since a
binding-pocket contact sits at the same distance as a covalent bond.

Written rather than taken from a deposited entry: one heavy atom per residue is
enough to place the contact, and a real glycan would carry the `struct_conn`
records whose absence is the point.
