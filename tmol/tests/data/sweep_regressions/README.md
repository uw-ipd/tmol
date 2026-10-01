# Sweep regression structures

Trimmed samples of RCSB entries that failed in the whole-PDB and CCD sweeps of TMol with AtomWorks protonation.
Each is cropped with the recipe in `provenance.json` (source hash, crop command, hashes of the decompressed and
compressed file) and exercises one failure class; each test fails without its fix.

| File | Regression exercised |
|---|---|
| `zn_link_records_1hzy.pdb.zst` | PDB LINK records become bonds (Zn coordination, LYS169 NZ-FMT C); HETATM residues keep their numbers and order. |
| `capped_peptide_1coi.pdb.zst` | A coordinate-only PDB (no LINK or CONECT records) keeps its HETATM caps (ACE A:0, NH2 A:30) in the chain. |
| `bonded_bromide_1mhk.cif.zst`, `unsigned_carboxylate_charge_3zlp.cif.zst`, `oxygen_charge_seven_7tjm.cif.zst` | Deposited charges no bonded atom carries (bonded bromide -1, unsigned carboxylate +1, +7 on OE2) are ignored. |
| `shared_disulfide_sulfur_6cnb.cif.zst` | A cysteine declared in two disulfides keeps the one nearest 2.04 A. |
| `metal_tyr_6yv5.cif.zst` | A TYR donor whose given state cannot coordinate Na is left uncoordinated with a warning. |
| `ca_trace_1a1d.cif.zst` | A CA-only trace (and an empty structure) is refused with the reason. |
| `glycan_o1_2yor.cif.zst` | A database that prepared NAG without O1 refuses a copy with O1 at context time. |
| `tyz_cap_and_conjugate_3w93.cif.zst`, `pyr_serine_amide_1i72.cif.zst` | A polymer port links only to the complementary port; any other bond there is a conjugation. |
| `kik_lysine_crosslink_5lnu.cif.zst` | The incompatible conjugate chemistry error names both sites. |
| `mmt_5prime_ether_1cx5.cif.zst` | A nucleotide whose 5' oxygen is substituted has no 5' port. |
| `lcc_leaving_atoms_6c8d.cif.zst` | The declared leaving group breaks a leaving-atom tie at a polymer connection. |
| `five_prime_phosphate_9cf0.cif.zst` | Every cart_bonded length and angle of 5'-phosphate nucleotides (na5primephos P-OP3) has parameters. |
| `terminal_alkyne_7e9i.cif.zst` | gen_torsions skip dihedrals through an acyclic sp centre; analytic and numeric gradients agree. |
| `free_nucleotides_8gpb.cif.zst` | Free nucleotides sharing a chain and entity (AMP A930, A940) are not linked O3'-P for protonation. |
| `cyclic_phosphate_3prime_1hq1.cif.zst` | A nucleotide whose 3' oxygen is in a 2',3'-cyclic phosphate (CCC) has no 3' port, so no 3' terminus adds HO3'. |
| `his_pos_nmr_2lny.cif.zst` | HIS_POS (both ring protons) has cart_bonded parameters: minimization keeps its bonds; a type without parameters is reported. |
| `oxygen_acceptor_1mbo.cif.zst` | An acceptor without a base (haem-bound O2, OXY) scores hbond with finite gradients. |
| `metal_lysine_2r1w.cif.zst`, `metal_amine_terminus_3ppd.cif.zst` | A lysine or N-terminal amine bonded to a metal is the neutral amine AtomWorks makes it (LYS_DEP, nterm_neutral). |
| `cysteine_heme_1cch.cif.zst` | The pyrroles of a Cys-bonded heme keep no hydrogen once its iron is split out. |
| `metal_phosphate_7bad.cif.zst`, `per_copy_phosphate_6m8q.cif.zst` | Each phosphate copy keeps AtomWorks' state: no hydrogen on an O bound to a metal, a free copy protonated. |
| `polar_hydrogens_10gs.pdb.zst` | A PDBbind protein drawn with polar hydrogens only: a histidine or cysteine without ring or thiol H is asked of AtomWorks, not read as an anion. |
| `unresolved_tyrosine_4ndz.cif.zst` | TYR B:171 drawn without its ring (N, CA, C, O, CB) keeps TYR; the missing atoms carry no protonation state. |
| `plm_copies_8trb.cif.zst` | Every PLM copy is bonded to the protein (SER/CYS), so no per-copy state type is prepared. |
| `capped_peptide_1j8z.cif.zst` | Prepared after 3PPD in one database, the ACE-capped peptide does not take GLY's on-demand nterm_neutral as a terminus template. |

| `nitric_oxide_1koi.cif.zst` | Full deposited entry: nitric oxide keeps its zero-hydrogen state during SMILES conversion; unsupported radical preparation is refused instead of adding a hydrogen. |

| `renumbered_chloride_4eu8.cif.zst` | Renumbering decreasing water author IDs preserves component definitions and charge provenance for the structure’s chloride ions. |
