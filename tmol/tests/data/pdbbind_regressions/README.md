# PDBbind regression files

Trimmed PDBbind v2013-core and PoseBusters inputs (protein/pocket PDB, ligand MOL2/SDF) that failed or were misread by TMol's
readers; `provenance.json` records the source file, its hash, the trim and the hashes of the files here. Each is
exercised by `tmol/tests/io/test_pdbbind_regressions.py`, whose tests fail without their fix.

| File | Regression exercised |
|---|---|
| `cyclic_acylguanidinium_3ge7.mol2.zst`, `iminohydantoin_4djv.mol2.zst` | Tripos `ar` bonds of a C.cat inside a ring localize to one double bond and an N cation (+2, +1). |
| `cyclic_acylguanidinium_3ge7.sdf.zst` | An SDF reads like its MOL2: the charge written on the C.cat carbon goes to N, a stray aromatic bond is single. |
| `pocket_1gpk.pdb.zst` | A blank PDB chain ID reads as a named chain; GLY 119 and TYR 121 (C-N 3.0 A across missing 120) are not bonded. |
| `numbered_gap_3kgp.pdb.zst` | Residues 37 and 38, numbered consecutively but 5.6 A apart, are not bonded (unless the threshold is 0). |
| `repeated_names_ligand_10gs.pdb.zst`, `repeated_names_ligand_10gs.mol2.zst` | Repeated atom names in one PDB residue (a peptidic ligand written as one) take the MOL2 reader's names. |
| `chloride_charge_column_6tw5.pdb.zst` | A charge in the PDB charge column (PoseBusters 6TW5 `Cl1-`) is the stated formal charge. |
