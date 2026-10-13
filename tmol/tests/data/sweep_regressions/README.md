# Sweep regression structures

Trimmed samples of RCSB entries that failed in the whole-PDB and CCD sweeps of TMol with AtomWorks protonation.
Each is cropped with the recipe in `provenance.json` (source hash, crop command, hashes of the decompressed and
compressed file) and exercises one failure class; each test fails without its fix.

| File | Regression exercised |
|---|---|
| `alternate_polymer_chains_1gtv.cif.zst` | Full deposited entry: retain both overlapping polymer chains in the reader; reject their degenerate combined bond geometry before scoring. Either explicitly selected biological assembly scores and minimizes with finite gradients. |
| `shared_author_site_3bln.cif.zst` | Full deposited entry: overlapping MPD/MRD in separate label chains share author site A:147; select one before collapsing author identifiers. |
| `nitric_oxide_1koi.cif.zst` | Full deposited entry: nitric oxide keeps its zero-hydrogen state during SMILES conversion; unsupported radical preparation is refused instead of adding a hydrogen. |
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
| `his_pos_nmr_2lny.cif.zst` | HIS_POS (both ring protons) has cart_bonded parameters: minimization keeps its bonds; a type without parameters is reported. |
| `oxygen_acceptor_1mbo.cif.zst` | An acceptor without a base (haem-bound O2, OXY) scores hbond with finite gradients. |
| `metal_lysine_2r1w.cif.zst`, `metal_amine_terminus_3ppd.cif.zst` | A lysine or N-terminal amine bonded to a metal is the neutral amine AtomWorks makes it (LYS_DEP, nterm_neutral). |
| `cysteine_heme_1cch.cif.zst` | The pyrroles of a Cys-bonded heme keep no hydrogen once its iron is split out. |
| `metal_phosphate_7bad.cif.zst`, `per_copy_phosphate_6m8q.cif.zst` | Each phosphate copy keeps AtomWorks' state: no hydrogen on an O bound to a metal, a free copy protonated. |
| `polar_hydrogens_10gs.pdb.zst` | A PDBbind protein drawn with polar hydrogens only: a histidine or cysteine without ring or thiol H is asked of AtomWorks, not read as an anion. |
| `unresolved_tyrosine_4ndz.cif.zst` | TYR B:171 drawn without its ring (N, CA, C, O, CB) keeps TYR; the missing atoms carry no protonation state. |
| `plm_copies_8trb.cif.zst` | Every PLM copy is bonded to the protein (SER/CYS), so no per-copy state type is prepared. |
| `capped_peptide_1j8z.cif.zst` | Prepared after 3PPD in one database, the ACE-capped peptide does not take GLY's on-demand nterm_neutral as a terminus template. |
| `two_atom_residue_1gj2.cif.zst` | The O residue (O, HO) bonded to a DNA phosphate is a polymer type whose rotamer kinforest has no grandchild to frame on. |
| `glycine_ca_conjugate_5wrh.cif.zst` | A glycine whose CA is conjugated (GLY:conj_CA, no HA3) rebuilds only the alpha hydrogen it has. |
| `unresolved_frozen_chi_6sm6.cif.zst` | A frozen chi whose defining atoms are unresolved (4HH tail beyond CM) takes its ideal value. |
| `renumbered_chloride_4eu8.cif.zst` | Renumbering decreasing water author IDs preserves component definitions and charge provenance for the structure’s chloride ions. |

| `localized_oxyacid_1akw.mol2` | PDBbind1AKW FMN | Original MOL2; preserve localized phosphate O.co2 charges through the aromatic reader fallback. |

- `stated_terminus_5lh4.pdb`: original SER130/ALA132 coordinates from PDBbind; a 1.94 Å gap must preserve the supplied neutral H1/H2 terminus. Cropping and source hash are in `provenance.json`.

- `aromatic_charge_{3djf,1tou,4yt6}.{mol2,sdf}`: complete paired prepared ligands from the CoreWeave PDBbind v2020 corpus. The MOL2 files omit formal charges needed for aromatic localization; their prepared partial-charge sums constrain the missing total. Companion SDF graphs independently check the recovered resonance class. No atoms, bonds, coordinates, hydrogens or charges were edited; only trailing whitespace was removed.
  - `aromatic_charge_3djf.mol2`: original `v2020-other-PL_5/3djf/3djf_prot/3djf_l.mol2`; SHA-256 `05e780d9891ac1e9e9c082cad05c4c761d4a9752b2f132118463d92c44b076ae`.
  - `aromatic_charge_3djf.sdf`: original `v2020-other-PL_5/3djf/3djf_prot/3djf_l.sdf`; SHA-256 `fc3d77b30ebb380100f68880f553a19c4d300cd39b913cd37451ddb22b5ce1ba`.
  - `aromatic_charge_1tou.mol2`: original `v2020-other-PL_5/1tou/1tou_prot/1tou_l.mol2`; SHA-256 `93a520982cc048d4191aaf73e923c986329509528c88ee679f2fcb3b455a15c3`.
  - `aromatic_charge_1tou.sdf`: original `v2020-other-PL_5/1tou/1tou_prot/1tou_l.sdf`; SHA-256 `646939697bda584434c141f3ded2b636a3e18252958032a2ef9dff5c366fcfb5`.
  - `aromatic_charge_4yt6.mol2`: original `v2020-other-PL_5/4yt6/4yt6_prot/4yt6_l.mol2`; SHA-256 `f2d03b4efb63a3abd4e35040ea679f26952959932dd544e35b1ea39ae861c50c`.
  - `aromatic_charge_4yt6.sdf`: original `v2020-other-PL_5/4yt6/4yt6_prot/4yt6_l.sdf`; SHA-256 `f36d30d079cce2f925cb208fec9ad4f4b846ebc37b7d25ab780d66bcf731c55c`.

- `backbone_hetero_order_4fut.pdb`: PDBbind 4FUT prepared protein, author chain A residues 11–13 and 17–19. ATOM/HETATM records remain in original file order; CONECT records retain only these atoms. No coordinates, names, charges or retained bond declarations were changed. Original source SHA-256: `f5bab11d54d06a2e4e025fc1c773769f2a491e9f9fbdd2dea65538e913a05d1f`. The two modified lysines are explicitly linked into the protein despite being listed after all ATOM records.

- `terminal_name_collision_1a8i.pdb`: PDBbind 1A8I prepared protein, author chain A, residues [679, 680, 681]. ATOM/HETATM and retained CONECT records preserve original order, names, coordinates, charges and bond declarations; no other edits. Source SHA-256 `116abba54481889550476cfde8138a9abc3e1ae895c20c1dff978dac98bcfb34`.

- `supplied_terminus_2gyi.pdb`: PDBbind 2GYI prepared protein, author chain A, residues [64, 65]. ATOM/HETATM and retained CONECT records preserve original order, names, coordinates, charges and bond declarations; no other edits. Source SHA-256 `b7a2e291987114fc56ebddb2bed8604554cac537d79a115528287706fb814297`.

- `supplied_terminus_1hdq.pdb`: PDBbind 1HDQ prepared protein, author chain A, residues [273, 274]. ATOM/HETATM and retained CONECT records preserve original order, names, coordinates, charges and bond declarations; no other edits. Source SHA-256 `304b450671db888b175cbfb3738e459ed0f5cf90a0ff39684ee95e9167956af1`.

- `terminal_pocket_2jdm.pdb`: complete, unchanged PDBbind v2013-core 2JDM pocket; source SHA-256 `1de0ea35f6ee8b532791cbd27faaed781877337e54d8f8fbc3faca2640a29cfb`. The first GLY114 has a supplied OXT and one generic polar H, and precedes ASN21. Retaining OXT must not be blocked by an unrelated hydrogen name already incompatible with the inferred amino terminus.

- `terminal_aldehyde_{2f9b,5myx,3t0x}.pdb`: PDBbind prepared proteins, cropped to 2F9B L:142/T:109/T:155/T:205, 5MYX B:217 and 3T0X A:107/B:106. All retained ATOM/HETATM lines and intraselection CONECT declarations are unchanged. Each source explicitly bonds HXT to carbonyl C at about 1.09 Å, with no OXT; this is aldehyde chemistry, despite the CCD name HXT normally belonging to OXT. These unsupported terminal forms must raise clearly instead of silently generating a carboxylate. No coordinates, atom names or retained bond declarations were repaired.
