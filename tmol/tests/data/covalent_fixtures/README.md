# Covalently linked component fixtures

Structures whose chemistry crosses a residue boundary through a bond that is
neither a polymer backbone link nor a disulfide: glycans hanging off a protein,
and a ligand conjugated to a sidechain.

| file | source | residues | covalent chemistry |
|------|--------|----------|--------------------|
| `nglycan_tree_1ax2.cif` | PDB 1AX2 chains A, B, C | 239 + 7 + 2 | N-glycan on `ASN ND2`, three-way branch |
| `oglycan_sia_1g1s.cif` | PDB 1G1S chains D, F | 13 + 6 | O-glycan on `THR OG1`, sialylated |
| `lys_biotin_1bdo.cif` | PDB 1BDO chain A | 80 + 1 | biotin on `LYS NZ` |
| `lactam_cyclic_7ag5.cif` | PDB 7AG5 chain A | 11 | C-terminal `PRO C` on `DNP NG`, closing a ring |

Read these with `include_bonds=True`: the attachment bonds come from
`struct_conn`, and nothing else in the file records them. Unlike the
noncanonical fixtures, all three keep `chem_comp`, `chem_comp_atom` and
`chem_comp_bond` for every component they contain, so their chemistry is
self-describing and needs no component dictionary.

Metals and solvent were dropped. 1AX2 keeps its calcium- and manganese-free
form because the ligand pipeline refuses metals; the glycan is unaffected.

## What each one covers

`nglycan_tree_1ax2.cif` is a plant complex N-glycan, and its point is the
branch structure: `BMA 3` carries three children at `O2`, `O3` and `O6`, and
the core `NAG 1` carries a second child at `O3` besides the chain it continues
at `O4`. A sugar therefore needs several attachment sites at once, and which
sites those are varies per residue.

Chain C is a `NDG`-`GAL` disaccharide bonded to nothing else in the file. It is
kept deliberately: a free oligosaccharide has to keep working through the
ordinary ligand path while the attached one grows connections.

`oglycan_sia_1g1s.cif` is the sulfated, glycosylated N-terminus of PSGL-1. It
covers three things a protein N-glycan does not:

* the attachment is to `THR OG1` rather than `ASN ND2`, so the acceptor is a
  hydroxyl and the leaving atom is its proton;
* `SIA` joins through `C2`, not `C1` -- sialic acid's anomeric carbon is not
  the one a hexose rule would look for;
* the peptide carries three sulfotyrosines (`TYS`), which reach the pose
  through the noncanonical path at the same time.

`lys_biotin_1bdo.cif` is the biotinyl domain of acetyl-CoA carboxylase, the
smallest of the three. Biotin is an ordinary ligand rather than a polymer
component, so it exercises the sidechain-conjugation half alone, with no
carbohydrate topology involved.

`peptide_inhibitor_7zv5.cif` retains model 1 of 7ZV5, protein chain A
residues 144–146 and inhibitor chain B residues 1–4 (ACE–GLY–PHE–HSV).
Its CYS SG–HSV C attachment joins two polymer chains. The default FastRelax
workflow checks that the masked fallback sampler cannot add an independent row
to the correlated group; all seven residues and their connections are retained.
The 44 observed atoms and source bonds are preserved, with component definitions
written by AtomWorks. Source: local 2026-01-06 PDB mirror, compressed SHA256
`54c162cb313732e8c1bab213983d53a0bf0290c444736cf259de4f0faceefef1`.

## Sidechain-closed ring

`lactam_cyclic_7ag5.cif` is the peptide core of a calcium-dependent lipopeptide
antibiotic, `ASP-DNP-CPI-ASP-ASP-GLY-ASP-GLY-2RA-ILE-PRO`, model 1 with the
first alternate location. The N-terminal lipid, both calcium ions, the bound
ligand and the solvent were dropped; the atom records, hydrogens included, are
the deposited ones. Unlike the others here it keeps no `chem_comp` block,
only the bonds (`chem_comp_bond` and `struct_conn`).

`DNP` (2,3-diaminopropionic acid) bonds through three atoms: its alpha
nitrogen and carbonyl continue the chain, and its beta nitrogen `NG` is
acylated by the C-terminal proline's carbonyl, closing the ring. Either
nitrogen with the carbonyl makes a valid peptide backbone, alpha or beta, so
the chain is read through the alpha one. The proline's `up` carbonyl lands on
a sidechain amine rather than a backbone nitrogen, so it is a conjugation, not
a polymer connection.
