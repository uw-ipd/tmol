# Complex AtomWorks inputs: review findings 92–100

These findings extend [REVIEW.md](REVIEW.md). Upstream anchors refer to Frank's
`0593a93b07d80b0302383163d2d98c78e315ab98`. Fixes are in review commit
`01159dd7adc8ec9ff7bdb4c9d5b3867a1cc3958e`. Fixture provenance is committed in
`tmol/tests/data/atomworks_regressions/provenance.json`.

## 92. Reject loss of a retained residue's covalent partner — P1

[Upstream `_filter_supported_atoms_and_connectivity`, line 589](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/_pose_stack_from_biotite.py#L589).

> If an incomplete residue is filtered out, can we first check whether its
> removal severs a sidechain, disulfide, cyclic or cross-chain bond to a retained
> residue? The Schiff-base fixture drops LYS with missing mainchain coordinates
> and then scores its ligand partner as if that declared bond did not exist.

Fixed: check removed/retained bond boundaries before canonical conversion.
Ordinary forward adjacent peptide gaps remain supported; arbitrary crosslinks
cannot disappear implicitly. Explicit coordination is excluded from this
covalent check. The unchanged Schiff-base fixture now raises an actionable
error. This does not supply a missing-backbone placement model.

## 93. Reject unmapped heavy atoms at the canonical boundary — P1

[Upstream `_map_atoms_to_canonical`, line 482](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/_pose_stack_from_biotite.py#L482).

> Why may an observed heavy atom that does not map to the selected residue type
> disappear without an error? Hydrogen rebuilding can be intentional, but an
> unknown heavy-atom identity should require a chemical definition or explicit
> input selection.

Fixed: report unmapped non-H/D atoms with residue and atom identity. Unknown
hydrogen names remain compatible with hydrogen rebuilding. The modified 1a8o
fixture has author atom `CG` but label atom `XYZ`: native author-name parsing
preserves the CG coordinate and passes; a label-name AtomArray raises. Published
AtomWorks 2.2.1 drops XYZ during completion, so the shared reader also checks
source heavy-atom names before tmol preparation. This is a component/name
inventory guard, not full per-residue or bond provenance.

## 94. Traverse the complete retained cap scaffold — P1

[Upstream `_icoor_order`, line 463](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_polymer_builder.py#L463).

> Can the traversal continue through retained non-sidechain heavy atoms beyond
> the initially typed backbone? The 4SO aromatic acyl cap in 6q9t has a valid
> scaffold, but ordering stops near the carbonyl and leaves atoms without icoors.

Fixed: traverse every reachable retained heavy scaffold branch and use a set
for placed-atom membership. The regression checks the complete selected 4SO
cap/partner fragment and finite heavy-atom geometry. Whole-file metal failures
are reported separately; selecting this fragment is explicit in the test.

## 95. Select the patch that actually applies to the base type — P1

[Upstream `_applied_patch`, line 267](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L267).

> Should patch selection require both the variant suffix and the patch's
> applicability predicate? LLP contains a phosphate group, so atom-name overlap
> alone incorrectly selects a nucleic-acid terminal patch for this amino acid
> and omits the OXT charge.

Fixed: require the actual display-name suffix and `applies_to.matches(base)`.
The 7MKV fixture exercises LLP preparation; the test verifies the selected
terminal patch and finite charges for all generated LLP variants. Native
whole-file scoring and minimization now pass. Local AtomWorks still rejects
the file's contradictory sequence metadata before this preparation stage.

## 96. Exclude sidechain attachment ports from backbone inference — P1

[Upstream `_routes_to_polymer_path`, line 932](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L932).

> When entity/type declarations are absent, why does a known sidechain
> conjugation atom count as a backbone endpoint? BCX in 1j8z has N/C peptide
> connections plus an SG disulfide, causing the fallback profile to reject its
> valid backbone and later misclassify its neighbors.

Fixed: remove identified conjugation atoms from connection endpoints used for
backbone profiling and pass that same filtered set to polymer preparation.
Preserve the sidechain connection for conjugation/disulfide construction. The
test checks BCX polymer identity, peptide connectivity and scoring derivatives.
Mutually unknown attachment partners can still need a more general inference
policy; this fix does not resolve every ambiguous chemical graph.

## 97. Let missing-leaf fallback see unresolved nonleaf ancestors — P1

[Upstream missing-atom mask, line 169](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/details/_build_missing_leaf_atoms.py#L169).

> Can ancestor selection use the full missing-atom mask? In the conditional
> generation fixture, valid N/CA/C coordinates coexist with missing CB. HA
> construction treats CB as available because the current mask marks only
> missing leaves, so it never uses its valid backup frame.

Fixed: pass all missing atoms to ancestor selection. Compose a backup fixed
phi only where the nonleaf reference shares parent/grandparent and does not
represent the fourth atom of a sampled named torsion. Conditional-generation
pose construction, scores and derivatives now pass alongside existing leaf
geometry and gradient tests. This keeps the repair in shared construction,
where tensor/backbone inputs also benefit, rather than a file-specific rebuild.

## 98. Reuse one parsed chemical category and one shared file adapter — P2

[Upstream `pose_stack_from_cif`, line 441](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/_cif.py#L441).

> Why parse the CIF again for `chem_comp_types` after the reader already loaded
> that category? Can AtomWorks supply its completed AtomArray and parsed block
> through this same boundary, so examples and profiling do not each implement
> a private parser configuration?

Implemented: annotate component types from the already parsed category in both
reader routes. An explicit caller mapping retains precedence. Add optional
`tmol[atomworks]` and `reader="atomworks"`; examples and profiling call that
public adapter. No fallback between readers is implicit. This removes duplicate
parsing/configuration work; no isolated end-to-end speedup is claimed for it.
All-file unification and direct tensor layouts remain a separate proposed
contract in [INPUT_CONTRACT.md](INPUT_CONTRACT.md).

## 99. Do not remove every possible leaving branch for one polymer bond — P1, AtomWorks follow-up

[Local AtomWorks leaving-atom implementation](https://github.com/baker-laboratory/atomworks-dev/blob/a1bda7edfcf325bc140091889b9745220adb5eba/src/atomworks/io/utils/leaving_atoms.py).

> The 8OG template flags both OP2 and OP3 as potential leaving atoms. Why does
> forming one phosphodiester linkage remove both branches? The PR's 183d DNA
> fixture has an observed OP2 which disappears in both released and local
> AtomWorks output.

**Update:** fixed in AtomWorks #349 at `4af94d0c`, now pinned by tmol. Separate chemically equivalent branches and a bond-order budget preserve observed 8OG OP2; repeated resolution and full tmol scoring/gradient checks pass. Ambiguous observed alternatives still require explicit chemistry. See [the continued review](CONTINUED_REVIEW.md).

Historical parser evidence: the released-reader matrix
has 18 passes and this one rejected input. Direct local inspection confirms
OP2 is absent in the parsed 8OG, not merely renamed. The CCD inventory marks
both OP2 and OP3 as leaving atoms; the removal implementation unions every
flagged branch for an inter-residue connection.

Proposed fix: retain distinct candidate leaving branches and select only the
branch displaced by the actual declared polymer port, with an explicit policy
for ambiguous observed alternatives. Do not blindly preserve all leaving
atoms, which would leave invalid over-valent connections, or use missingness
alone as authorization to rewrite chemistry. Preserve a source-to-output atom
map and record each intentional deletion. AF3's 7ubd fixture also contains
observed terminal OXT atoms removed at internal links; that case needs an
explicit normalization policy rather than bypassing the identity guard.
The correction is now supplied by the required companion dependency. Full atom-provenance tracking and the AF3 normalization policy remain open.

## 100. Make hydrogen authority explicit when only heavy-atom bonds are supplied — P2

[Shared RDKit conversion](https://github.com/kierandidi/tmol/blob/01159dd7adc8ec9ff7bdb4c9d5b3867a1cc3958e/tmol/ligand/_rdkit_mol.py#L371).

> The 1IZC input contains four observed PYR hydrogens but only five heavy-atom
> bonds. Passing any explicit hydrogen to Biotite disables implicit hydrogens
> for the whole component. Can the preparation contract distinguish retaining
> a complete supplied hydrogen model from discarding and rebuilding hydrogens?

The native metal-excluded diagnostic fails on disconnected hydrogen fragments
and a radical methyl carbon during SMILES preparation. This is incomplete input
hydrogen connectivity, not a reason to infer bonds from coordinates. The
AtomWorks reader's explicit `hydrogen_policy="remove"` avoids the ambiguity;
its metal-excluded structure passes scoring and minimization. This is a concrete
benefit of the shared reader. Native compatibility still needs an explicit
hydrogen rebuilding option or an earlier actionable incomplete-bond error;
do not silently discard hydrogens when the caller requested preservation.
