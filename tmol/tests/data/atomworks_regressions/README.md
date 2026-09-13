# AtomWorks regression structures

Except for the synthetic amine/imine attachments and separately sourced RCSB entries listed in the manifest, these files are copied unchanged from the local `atomworks-dev` checkout at
`a1bda7edfcf325bc140091889b9745220adb5eba`. `provenance.json` records each source
path and SHA-256 digest. AtomWorks is distributed under the BSD 3-Clause license.
The structure files retain their original experimental/generated metadata.

| File | Regression exercised |
|---|---|
| `schiff_base_double_bond.cif` | Prepare the declared double attachment and its hydrogen inventory; report the missing lysine backbone instead of discarding one covalent partner. |
| `unknown_heavy_atom_1a8o.cif` | Distinguish the conflicting author CG / label XYZ identities; retain the author coordinate and reject unknown label atoms or parser deletion. |
| `unresolved_unl.cif` | Retain all 28 unresolved ligand heavy atoms at NaN; explicitly reject unanchored ligand placement. |
| `modified_components_6q9t.cif` | Traverse the whole aromatic acyl cap in the covalently connected 4SO–A1IJ4 pair. The targeted test explicitly selects this pair; the original also contains zinc. |
| `plp_enzyme_7mkv.cif` | Match LLP terminal patches by scope and suffix, and supply finite charge coverage for every LLP variant. |
| `acetylated_peptide_1j8z.cif` | Recognize the BCX backbone separately from its disulfide attachment; retain the peptide connections and score it. |
| `conditional_generation.cif` | Rebuild missing sidechains and alpha hydrogens from valid backbone coordinates; check finite scores and gradients. |

Tests live in `tmol/tests/io/test_atomworks_corpus_regressions.py`,
`test_atomworks_reader.py`, and
`tmol/tests/ligand/test_atomworks_modified_components.py`.
Successful numerical checks do not independently validate the force field.

`plp_cap_5t4j.cif.gz` retains the complete RCSB entry referenced by the AtomWorks
IO suite. Rebuilding unresolved sidechains triggers packing of a PLP-derived cap
with a one-atom backbone. Its fingerprint must retain distinct substituent
positions using the cap's connection/construction frame. The integrated test
checks all retained source connections, opposite and reflected hydrogen labels,
finite scoring/gradients and minimization through both readers. The AtomWorks
route additionally carries five entirely unresolved protein residues, which
construction explicitly excludes; it is a partial-input success.

`macrocycle_1xvk.cif` is the complete RCSB entry used by the wider AtomWorks IO suite. The regression explicitly excludes free Mg and water, verifies QUI cap identity and all 18 covalent links, and scores/minimizes both original and reversed residue orders through both readers. The initial energy must be independent of residue order.

`terminal_nucleotide_145d.cif` is the complete RCSB entry used by that suite. Its first MCY must retain a DNA backbone and 5-prime patch, without proximity-inferred conjugations. The asymmetric unit contains two overlapping alternative duplexes. Both readers retain all 24 nucleotide blocks by default; explicitly selecting assembly `"1"` or `"2"` through AtomWorks gives the corresponding 12-residue duplex. The regression checks every phosphodiester link, finite scores/gradients and minimization, and intact bond lengths after relaxing each selected assembly. A temporary translated-copy assembly also checks coordinate transforms, copy identities, and equivalent direct AtomArray construction. Synthetic parameter-generation caps must inherit the source residue's transformation identity.

`conflicting_myristate_1aym.cif.gz` preserves the complete compressed entry. Its `struct_conn` category declares MYR C1 bonded to both GLY N and CA. After explicit free-zinc exclusion, construction must report both partners rather than overwrite the one MYR port. This is an input-conflict regression, not a successful whole-complex minimization.

`repeated_glycans_6mub.cif.gz` is losslessly compressed from the complete authored
AtomWorks fixture. Two MAN–MAN links share a patched type pair but have different
geometry in one generated conformer. Their transferable bond/angle targets must
come from the conformer generator's ideals, not individual strained sites. Both
readers retain every observed non-water residue, every glycan and their source
connections (entirely unresolved protein residues are explicitly excluded), produce identical
records after residue reversal/seed changes, and score/minimize both orders. The
regression disables optional geometric disulfide inference to require the declared
source graph exactly; the corpus exercises the default inference policy.

`generated_amine_attachment.cif` is a synthetic acetylated sugar. Its two declared
components require a sidechain conjugation despite the acyl fragment's inferred
polymer port. Both readers must prepare the linked glycosyl amine with one N–H
instead of the isolated protonated amine's three, conserve Frank's per-residue
charge total, retain the source bond, and score/minimize. This validates topology
and the existing charge convention, including the local amide type. Independent parameter-fit
validation remains separate work. The manifest records its complete generation
recipe and stereochemical SMILES.

`generated_imine_attachment.cif` replaces that fixture's carbonyl oxygen with a
methyl carbon, moves the nitrogen into the first component, and declares a double
N1=C4 attachment at a sugar ring carbon. Both readers must generate
double connection ports and an imine nitrogen without N–H, conserve each residue's
prepared charge, retain both blocks, and score/minimize. Multiple bonds must not
receive the generic staggered linkage sampling grid; its ring endpoint also
cannot rotate about a local ring bond. The manifest records the
exact chemical edits; initial coordinates are intentionally retained.

`af3_cyclic_peptide_7ubd.cif` retains the complete AtomWorks AF3 prediction.
Both readers construct the eight-residue cycle, remove its polymerization leaving
atoms, retain every other observed coordinate, and score/minimize. AtomWorks
continues to reject unknown atom names and preserve retained phosphate oxygens.

`chromophore_3nez.cif.gz` is the complete RCSB entry used by the AtomWorks IO
suite. NRQ supports a C-terminal patch but no N-terminal patch. Canonical ordering
must register that available end without requiring both patches. Both readers
retain all four connected chromophores and score/minimize the constructed pose;
the AtomWorks route also retains unresolved residues in its input array, which
the constructor excludes when their required backbone coordinates are absent.

`missing_ligand_carbon_5hs6.cif.gz` preserves the complete RCSB entry, including
the unresolved J3Z carbon C6. Both readers retain its chemical identity. The
regression excludes sodium and water, constructs the missing carbon and dependent
hydrogens using prepared internal coordinates, preserves observed ligand atoms,
and scores/minimizes the protein–ligand complex. The isolated ligand also checks
gradients through reconstruction against finite differences.

`free_and_attached_solutes_5xag.cif.gz` retains the full RCSB structure. It
contains both free glycerol/imidazole and one declared GOL O3–IMD N3 bond.
The test excludes free magnesium/calcium and waters, constructs the complex
with ordinary solutes plus connected variants, then scores/minimizes it.
Imidazole N3 has no departing hydrogen; its attachment frame comes from its
prepared neighbor geometry and receives the connected generator targets.
Reusing the full preparation for the free solutes must match fresh preparation.

`decreasing_water_author_ids_5xnl.cif.gz` retains the complete 5XNL entry.
Its water chains have undefined label sequence numbers and decreasing author
numbers; fallback IDs must keep these residues distinct. The regression parses
all atoms through both readers, preserves all 98,986 observed atoms and 1,076
waters, then scores/minimizes protein chain A. Scoring that selected chain does
not validate the complete photosystem's metal-bound cofactors; metals remain
outside this regression's scoring scope.

`repeated_partner_glycans_1ivo.cif.gz` and `repeated_partner_glycans_1hge.cif.gz`
retain their complete entries. One NAG connection pattern occurs with chemically
different partners: its shared local residue frame cannot own the different
junction bond lengths/angles. The connection-pair records retain those generator
targets with K300/K80. The regression excludes waters and explicitly unbonded
metals, verifies distinct pair-specific targets, and reconstructs/scores/minimizes
both residue orders through both readers without dropping glycans or their bonds.
`terminal_asj_glycans_1iau.cif.gz` retains the complete 1IAU input. The repeated
glycan workflow excludes water and only unbonded metals, then constructs, scores
and minimizes both residue orders through both readers. It also checks that the
ASJ terminal oxygens keep their supplied names and coordinates and that their
patch scope does not change ASP sidechain atom classification.
