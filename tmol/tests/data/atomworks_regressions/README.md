# AtomWorks regression structures

Except for the separately sourced RCSB 1xvk, 145d and 1aym entries listed in the manifest, these files are copied unchanged from the local `atomworks-dev` checkout at
`a1bda7edfcf325bc140091889b9745220adb5eba`. `provenance.json` records each source
path and SHA-256 digest. AtomWorks is distributed under the BSD 3-Clause license.
The structure files retain their original experimental/generated metadata.

| File | Regression exercised |
|---|---|
| `schiff_base_double_bond.cif` | Reject removal of an incomplete lysine while retaining its covalent partner. |
| `unknown_heavy_atom_1a8o.cif` | Distinguish the conflicting author CG / label XYZ identities; retain the author coordinate and reject unknown label atoms or parser deletion. |
| `unresolved_unl.cif` | Retain all 28 unresolved ligand heavy atoms at NaN; explicitly reject unanchored ligand placement. |
| `modified_components_6q9t.cif` | Traverse the whole aromatic acyl cap in the covalently connected 4SO–A1IJ4 pair. The targeted test explicitly selects this pair; the original also contains zinc. |
| `plp_enzyme_7mkv.cif` | Match LLP terminal patches by scope and suffix, and supply finite charge coverage for every LLP variant. |
| `acetylated_peptide_1j8z.cif` | Recognize the BCX backbone separately from its disulfide attachment; retain the peptide connections and score it. |
| `conditional_generation.cif` | Rebuild missing sidechains and alpha hydrogens from valid backbone coordinates; check finite scores and gradients. |

Tests live in `tmol/tests/io/test_atomworks_corpus_regressions.py`,
`test_atomworks_reader.py`, and
`tmol/tests/ligand/test_atomworks_modified_components.py`.
The wider scoring/minimization runner is `review/pr503/run_atomworks_corpus.py`.
Successful numerical checks do not independently validate the force field.

`macrocycle_1xvk.cif` is the complete RCSB entry used by the wider AtomWorks IO suite. The regression explicitly excludes free Mg and water, verifies QUI cap identity and all 18 covalent links, and scores/minimizes both original and reversed residue orders through both readers. The initial energy must be independent of residue order.

`terminal_nucleotide_145d.cif` is the complete RCSB entry used by that suite. Its first MCY must retain a DNA backbone and 5-prime patch, without proximity-inferred conjugations. The regression checks all 20 phosphodiester links and finite scoring/minimization of the 24 nucleotide blocks.

`conflicting_myristate_1aym.cif.gz` preserves the complete compressed entry. Its `struct_conn` category declares MYR C1 bonded to both GLY N and CA. After explicit free-zinc exclusion, construction must report both partners rather than overwrite the one MYR port. This is an input-conflict regression, not a successful whole-complex minimization.
