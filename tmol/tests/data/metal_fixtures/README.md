# Metal coordination fixtures

PDB entries, gzipped and edited only as noted below, each chosen for one
coordination context.
Expected coordination for every metal is in `expected.yaml`, seeded from the
RCSB `struct_conn` metalc records and checked against the coordinates. Read
them with `tmol.io.atom_array_from_cif`.

| file | entry | resolution | context |
|------|-------|-----------:|---------|
| `zn_tetrahedral_3ks3.cif.gz` | 3KS3 carbonic anhydrase II | 0.90 | tetrahedral Zn, 3 His, one open site |
| `ca_irregular_2fvy.cif.gz` | 2FVY glucose-binding protein | 0.92 | 7-coordinate Ca: backbone carbonyl, bidentate Glu |
| `sf4_ferredoxin_2fdn.cif.gz` | 2FDN 2x[4Fe-4S] ferredoxin | 0.94 | cluster with bridging sulfurs (xfail) |
| `heme_myoglobin_5yce.cif.gz` | 5YCE sperm whale myoglobin | 0.77 | Fe inside heme, one axial His (xfail) |
| `cu_blue_copper_2ov0.cif.gz` | 2OV0 amicyanin | 0.75 | type 1 Cu: His2Cys trigonal plane, axial sites open |
| `mg_one_donor_4e3y.cif.gz` | 4E3Y Serratia endonuclease | 0.95 | octahedral Mg with one protein donor, two copies |
| `fe_rubredoxin_30oh.cif.gz` | 30OH P. abyssi rubredoxin | 0.43 | Fe(Cys)4 as a free ion |
| `cu_zn_sod_3f7l.cif.gz` | 3F7L Cu,Zn superoxide dismutase | 0.99 | Cu(I) and Zn in one site |
| `mg_rna_aptamer_7eoh.cif.gz` | 7EOH Pepper RNA aptamer | 1.64 | Mg on phosphate and guanine N7; Mg with only waters |

Crystallographic waters are stripped by structure input, so the expected
donors are what remains without them. The waters each metal coordinates are
listed separately as reference positions for open coordination sites.

Other ions in these entries (Na in 2OV0, 30OH and 3F7L) are incidental and
carry no expectations.

`cu_zn_sod_3f7l.cif.gz` is edited: its copper is deposited as two alternates
modeled as separate residues (CU1 201, altloc A, 0.8; CU 202, altloc B, 0.2),
and the reader keeps both. CU 202 was removed and CU1 201 set to occupancy
1.00.

Every entry is also reduced to one conformer with no hydrogens: each residue
keeps only its highest-occupancy altloc, set to occupancy 1.00, and all H and
D atoms are removed so they are rebuilt.
