# PDBbind regression files

Trimmed PDBbind (v2013-core, v2020 prepared) and PoseBusters inputs (protein/pocket PDB, ligand MOL2/SDF) that failed or were misread by TMol's
readers; `provenance.json` records the source file, its hash, the trim and the hashes of the files here. Each is
exercised by `tmol/tests/io/test_pdbbind_regressions.py`, whose tests fail without their fix.

| File | Regression exercised |
|---|---|
| `cyclic_acylguanidinium_3ge7.mol2.zst`, `iminohydantoin_4djv.mol2.zst` | Tripos `ar` bonds of a C.cat inside a ring localize to one double bond and an N cation (+2, +1). |
| `cyclic_acylguanidinium_3ge7.sdf.zst` | An SDF reads like its MOL2: the charge written on the C.cat carbon goes to N, a stray aromatic bond is single; parameters are written from it as from the MOL2. |
| `pocket_1gpk.pdb.zst` | A blank PDB chain ID reads as a named chain; GLY 119 and TYR 121 (C-N 3.0 A across missing 120) are not bonded. |
| `numbered_gap_3kgp.pdb.zst` | Residues 37 and 38, numbered consecutively but 5.6 A apart, are not bonded (unless the threshold is 0). |
| `repeated_names_ligand_10gs.pdb.zst`, `repeated_names_ligand_10gs.mol2.zst` | Repeated atom names in one PDB residue (a peptidic ligand written as one) take the MOL2 reader's names. |
| `chloride_charge_column_6tw5.pdb.zst` | A charge in the PDB charge column (PoseBusters 6TW5 `Cl1-`) is the stated formal charge. |
| `heavy_atom_imidazole_6t88.sdf.zst`, `heavy_atom_indazole_6tw5.sdf.zst` | Heavy-atom SDF ligands (PoseBusters) keep their Kekule tautomer, and aromatic bond orders keep their atoms aromatic. |
| `capped_break_1err.pdb.zst` | Atoms a mid-chain residue cannot hold (PrepWizard's H1, H2 at 1ERR's break) make the residue after a gap a terminus. |
| `single_bonded_carboxyl_2xej.mol2.zst` | A carboxyl written C.3 with two single C-O bonds (PrepWizard 2XEJ, ligand C-terminus) is read as the planar carboxylate it is, by the shared delocalized-group routine. |
| `own_ligand_mol_3udh.pdb.zst`, `own_ligand_dgx_1igj.pdb.zst` | A ligand named with the CCD code of another molecule (MOL, DGX) keeps its own atoms and CONECT bonds: no CCD atom at NaN, no CCD bond. |
| `hydrogens_named_apart_hux_1e66.pdb.zst` | A CCD ligand (HUX) whose CONECT records bond hydrogens the CCD names apart keeps the CCD's heavy-atom bonds and its own hydrogen bonds. |
| `ligand_named_pro_3uri.pdb.zst`, `ligand_named_pro_3uri.mol2.zst` | A 65-atom ligand named PRO beside prolines is renamed `L_1` with only its CONECT bonds; read from MOL2 it is a non-polymer. |
| `free_leucine_3b3s.pdb.zst` | A free leucine ligand beside the chain's leucines, its hydrogens named apart from the CCD's, keeps its name and the CCD's LEU. |
