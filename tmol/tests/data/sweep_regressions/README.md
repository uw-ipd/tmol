# Sweep regression structures

Trimmed samples of RCSB entries that failed in the whole-PDB and CCD sweeps of TMol with AtomWorks protonation.
Each is cropped with the recipe in `provenance.json` (source hash, crop command, hashes of the decompressed and
compressed file) and exercises one failure class; each test fails without its fix.

| File | Regression exercised |
|---|---|
| `kgq_alternates_4m8y.pdb.zst` | KGQ A:201 (altloc A) and A:202 (altloc B) are two overlapping residues; the PDB reader keeps one. |
| `metalc_alternate_3p1o.cif.zst` | A metalc row naming conformer B of GLU A:86 binds MG A:237 only if that conformer is kept (the kept conformer A is 6 A away). |
| `ion_alternates_8a7k.cif.zst` | Mn and Mg modelled at half occupancy on each of three sites without altloc ids keep one residue per site (the site failed the 12-connection limit). |
| `ion_alternates_3f7l.cif.zst` | The two conformers of a Cu (0.8/0.2), written in two chains, keep the more occupied one (one residue per site). |
| `heme_alternates_1i54.pdb.zst` | A PDB whose heme (altloc A) and Zn-porphyrin (altloc B) are two residues bonded to the same cysteines keeps one alternate per linked group. |
| `microheterogeneity_1ejg.pdb.zst` | PRO/SER A:22 and LEU/ILE A:25 are altlocs A/B of one position; the internal PDB parser keeps one alternate instead of overwriting atom by atom. |
| `glycerol_alternates_1p4k.cif.zst` | Two half-occupied GOL without altloc ids, which struct_conn bonds to each other, keep one residue per site (the pair failed with multiple declared partners). |
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

| `nitric_oxide_1koi.cif.zst` | Full deposited entry: nitric oxide keeps its zero-hydrogen state during SMILES conversion; unsupported radical preparation is refused instead of adding a hydrogen. |
