# AtomWorks regression structures

Except for the separately sourced RCSB 1xvk entry listed in the manifest, these files are copied unchanged from the local `atomworks-dev` checkout at
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

`macrocycle_1xvk.cif` is the complete RCSB entry used by the wider AtomWorks IO suite. The regression explicitly excludes free Mg, verifies QUI cap preparation, and checks an actionable rejection for the still unsupported polymer-port/chain topology. It does not claim whole-complex scoring succeeds.
