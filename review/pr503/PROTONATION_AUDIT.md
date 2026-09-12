# Shared protonation audit

The engines were **not identical as complete preparation pipelines**. Current AtomWorks `dev` did not contain the histidine/preprocessing work on the older local branch. Its default rule inventory also differs from tmol. This follow-up uses one AtomWorks engine with tmol's explicit rules, shares source-chemistry repair/conversion code, and preserves observed histidine protons at the tmol parser boundary.

Implementation: tmol `75e02157b`, AtomWorks `774056c7` based on current `dev` `59afb1e2`. The companion is [AtomWorks PR #349](https://github.com/baker-laboratory/atomworks-dev/pull/349), targeting `dev`. Frank's latest fetched head remains `0593a93b0`, already merged.

## What “same” means

| Layer | Exact compatibility contract | Validation / limits |
|---|---|---|
| Protonation rules | Same ordered SMARTS, pKa values, pH interval, precision, state budget and first-state selection | tmol retains its scientific rule data and passes an explicit provider. No default AtomWorks pKa data is overwritten. |
| Dimorphite states | Same ordered canonical mapped products | 258 parseable SMILES literals from both test trees and vendor tests, at pH 2.0, 7.4 and 12.0: **774 comparisons, zero differences** against the pre-sharing tmol engine. Dynamic/generated fixtures are covered by their actual test suites, not this AST inventory. |
| Molecule identity | Original heavy-atom order, maps, noncomputed properties and conformer coordinates | Dedicated tests cover neutralization, charged input, azide, nitro, amino acids and phosphate; invalid-final-state fallback also retains identity. Chemical equality alone did not catch the original atom-order bug. |
| Histidine input selection | HD1, HE2, both, default HE2, HN proximity, and NH/NN remapping | All six existing tmol coordinate fixtures are executed with an additional direct AtomWorks/native comparison. Native tensor/GPU selection remains in tmol. |
| AtomWorks H generation | Selected ring H counts and total formal charge remain consistent before/after AddHs | Explicit `histidine_policy="input"`, mirrored DHIS, three pH values, repeated preparation, NaN H, aliases, residue/assembly identity, and full 1A8O/6LYZ/1A1E structures. Substituted/incomplete rings and unresolved HN ambiguity are rejected. |
| Source-chemistry repairs | Same optional radical-O and finite planar carboxylate rules | tmol now imports the AtomWorks implementation. The finite/NaN/infinite/coincident/nonplanar geometry cases are shared regressions. Repairs propagate into output bond tables when requested. |
| File evidence | Observed HIS ring H must survive until tautomer selection | tmol parses with H retained, then removes other H and unobserved template H. Four file-reader regressions verify HD1, HE2, both and absent-ring-H cases. |
| Force-field preparation | tmol still owns Rosetta/MMFF typing, partial charges, construction and scoring | AtomWorks formal charges and tmol force-field partial charges are different quantities. This audit does not claim equality of complete force-field preparation or a scientific pKa refit. |

AtomWorks' `ensure_hydrogens` default titrates at the requested pH; its explicit input-HIS policy preserves a structural choice. To match a consumer's ligand enumeration, pass its provider **and its state budget**: tmol uses 128, while the existing AtomWorks hydrogen pipeline defaults to 1. The input policy preserves chemistry instead of deleting generated H based on geometric clashes. Original input coordinates still may need optimization; this does not certify a relaxed hydrogen geometry.

The inventories differ in phosphate/phosphorothioate patterns, `[!H]` versus `[!#1]` SMARTS, sulfonamide/phosphoramide/nitrosamine protection, and enamine handling. With AtomWorks' default rules, five of the 774 mapped-state comparisons differ: an Fe-containing diagnostic fixture at pH 7.4, an N-hydroxy/enamine fixture at pH 2.0 and 7.4, and `NNC=C` at those two pH values. The identical tmol-provider results do not justify silently replacing either rule inventory or asserting a new metal model.

## Tests and artifacts

- AtomWorks changed/adjacent chemistry suites: **180 passed**. The existing hydrogen-policy suite and hydrogen-component cases ran in full, with required PDB data supplied. No assertions, tolerances or skips were weakened.
- Affected AtomWorks protonation/histidine/identity tests after final review: **74 passed**.
- Older local AtomWorks protonation suite: **11 passed** unchanged on its source. Its postprocessing helper tests passed despite the end-to-end bond/proton problems; they are not evidence that the old integration was correct.
- tmol native-HIS/direct-AW comparisons plus protonation, geometry and ligand selection: **82 passed, 6 skipped**.
- tmol reader/HIS/CIF selection after preserving histidine evidence: **23 passed**.
- Final committed-implementation checks: **46 tmol tests and 34 AtomWorks histidine tests passed** (overlapping selections, not additional unique coverage).
- Shared CIF chemistry/nonstandard-backbone selection: **79 passed, 4 skipped**.
- [Full mapped-state comparisons and provenance](results/protonation-fixture-parity.json), [test counts and XML hashes](results/shared-chemistry-validation.json).

The Slurm reruns completed: 1,468 broad passes, 15 skips, ten initial failures (eight inherited references, one subsequently fixed stale mock and one intermittent unchanged-scorer CUDA repeat assertion). All 36 examples pass. Both 66-trial corpus modes retain every prior pass/partial/fail outcome. See [broad results](results/shared-broad-validation.json) and [corpus results](results/shared-corpus-validation.json). No numerical tolerance or scientific baseline was weakened.

## Additional anchored review comments

These comments include dependency and follow-up defects discovered during consolidation; they are not all defects introduced by Frank. The original 100 findings remain in [REVIEW.md](REVIEW.md) and [CORPUS_FINDINGS.md](CORPUS_FINDINGS.md).

### 101. Identical engine code does not establish identical scientific rules

At the [tmol rule inventory](https://github.com/kierandidi/tmol/blob/75e02157b/tmol/ligand/site_substructures.smarts) and [AtomWorks inventory](https://github.com/baker-laboratory/atomworks-dev/blob/59afb1e2/src/atomworks/external/dimorphite_dl/site_substructures.smarts): keep rule provenance, ordering, pH/precision and variant budget explicit. A single unversioned default cannot reproduce both libraries. **Fixed:** the tmol facade supplies its exact provider; AtomWorks exposes provider/budget selection and bounded compiled-rule ownership.

### 102. Histidine postprocessing can erase bonds without enforcing the selected state

At the [older local application function](https://github.com/baker-laboratory/atomworks-dev/blob/a1bda7ed/src/atomworks/io/utils/histidine_tautomer.py#L239): pruning named H after generation cannot add a missing selected proton or consistently set charges, and concatenating separately sliced residues drops inter-residue bonds. Name-only evidence treats unresolved H as observed; the old name list omits tmol's DHIS. **Fixed in #349:** capture finite evidence before stripping, rename on a whole-array copy, assign ring H/charges before AddHs, and test full covalent structures and mirrored DHIS.

### 103. Reaction products preserve maps but not source atom order/properties

At [neutralization](https://github.com/baker-laboratory/atomworks-dev/blob/59afb1e2/src/atomworks/external/dimorphite_dl/dimorphite_dl.py#L337): RDKit reaction order is not input order. An AtomArray-indexed target or charge update can therefore act on the wrong atom even when canonical mapped SMILES looks right. **Fixed in #349:** propagate reaction-source properties and restore original order and conformers before returning sanitized molecular states. Add order/coordinate/property assertions, not only set equality of maps.

### 104. Molecular fallback must preserve the same identity contract

At [direct molecular enumeration](https://github.com/baker-laboratory/atomworks-dev/blob/59afb1e2/src/atomworks/external/dimorphite_dl/dimorphite_dl.py#L1249): storing/reparsing fallback SMILES loses molecule metadata and geometry. **Fixed in #349:** retain molecular fallback states and restore their input correspondence; a forced invalid-final-state regression exercises this path.

### 105. The tmol adapter stripped the evidence needed for histidine selection

At [the pre-sharing adapter](https://github.com/kierandidi/tmol/blob/9a6bc4f8d/tmol/io/_atomworks_reader.py#L33): `hydrogen_policy="remove"` erases an observed HD1 or HE2 before tmol chooses the variant. This was a defect in this follow-up's initial adapter. **Fixed:** parse with H, retain finite HIS ring H, and verify four file-level states. The adapter now requires the pinned current API and removes released/local API dispatch.

### 106. Template graph rematching is unnecessary once identity is preserved

At [component protonation](https://github.com/baker-laboratory/atomworks-dev/blob/59afb1e2/src/atomworks/io/utils/protonation.py#L354): matching a protonated graph back onto the input repeats graph work and introduces symmetry ambiguity. **Fixed in #349:** consume the identity-preserving Mol directly, including consistent bond updates for explicitly requested repairs/HIS selection. Existing backbone-H and full-structure tests verify the boundary behavior.

### 107. Shared RDKit conversion needs an explicit H and coordinate contract

At [the converter](https://github.com/baker-laboratory/atomworks-dev/blob/59afb1e2/src/atomworks/io/tools/rdkit.py#L728): keeping supplied H is not the same as forbidding additional implicit H, and `set_coord=False` must not be replaced by an automatic truthy expression. **Fixed in #349 and consumed by tmol:** explicit inference control, owned retained annotations, direct source-column iteration, bulk conformer positions, and no duplicate tmol bond restoration/map. Tmol retains its own source-typing and normalization policy.

### 108. Authored CIF chemistry should use the shared parser without changing authority

At [tmol's previous component reader](https://github.com/kierandidi/tmol/blob/9a6bc4f8d/tmol/io/_cif.py#L55): maintaining a second atom/bond category parser duplicates AtomWorks, but unconditional CCD supplementation would replace authored chemistry for reused component codes. **Fixed:** AtomWorks supplies `supplement_from_ccd=False`, accepts omitted component metadata/aromatic flags, and tmol delegates while preserving its template contract. Dictionary lookup is forbidden by a regression for a custom molecule named ALA.

## Review questions

1. Should the public shared protonation-rule profile be maintained with AtomWorks or kept alongside the force-field consumer? The implementation preserves tmol's data until that scientific ownership is decided.
2. Should input HIS selection be an explicit parser/preparation setting throughout other AtomWorks consumers? Changing generic pH titration defaults would be a scientific behavior change.
3. What should a modified/substituted histidine ring request: input-preserving chemistry, ligand titration, or an explicit user state? The current input policy reports unsupported substitution.
4. Should clash handling optimize H coordinates while preserving charge/valence, rather than remove chemically required protons? The input policy already avoids chemistry-changing deletion.
5. Which released AtomWorks version should replace the temporary immutable development pin? The tmol draft currently requires authenticated access to the companion revision; a public release dependency is required before package publication.

## Measured conversion efficiency

Against the converter from AtomWorks `dev` `59afb1e2`, with the same runtime and identical output chemistry/coordinates, synthetic 20-atom and 200-atom cases improve from 0.230 to 0.123 ms and 1.812 to 0.986 ms. A 3,000-atom case carrying 32 unused annotations improves from 58.920 to 14.116 ms; Python-traced peak memory falls from 49.261 MB to 0.176 MB. Nine warm repetitions were measured. Native allocations are not included in tracemalloc; these are converter microbenchmarks, not end-to-end scoring speedups. [Measurements](results/shared-converter-benchmark.json).

The bulk setter is available at AtomWorks' RDKit 2024.3.5 minimum ([release source](https://github.com/rdkit/rdkit/blob/Release_2024_03_5/Code/GraphMol/Wrap/Conformer.cpp)); it receives a contiguous float64 array, avoiding the older strided-array issue. Native batched tmol coordinate/gradient paths are retained.

## Legacy vendor self-test discrepancy

The vendor's built-in `TestFuncs.test()` was also executed, without altering its
expectations, on latest AtomWorks dev, pre-sharing tmol and the shared engine.
All three stop at the **same** assertion: secondary aniline `CCNc1ccccc1` remains
neutral at the self-test's extremely acidic pH instead of its expected cation.
The existing `*Amide_conjugated2` rule (`[NH][c,n,o]`) matches that N before the
aniline rule and protects it as an amide. This is an inherited scientific-rule/
self-test inconsistency, not introduced by sharing. The actual first failure and
all three sources are recorded in [vendor-builtin-tests.json](results/vendor-builtin-tests.json).

The built-in harness is fail-fast, so this result does not claim all later
embedded assertions passed. Their parseable molecular inputs are included in
the expanded mapped-state comparison. The proposed scientific follow-up is to
narrow or reprioritize the overly broad protective rule against independent
aniline/conjugated-amide references. This PR preserves the requested exact tmol
policy and does not silently refit it to force that legacy test green.
