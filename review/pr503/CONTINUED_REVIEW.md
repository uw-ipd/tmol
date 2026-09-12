# Continued review: integrated AtomWorks inputs and model tutorials

The comparison is Frank's `0593a93b07d80b0302383163d2d98c78e315ab98` (already
merged), and AtomWorks dev `59afb1e2`. The shared AtomWorks implementation is
now `4af94d0c`; the immutable tmol GPU checkpoint is `4b4ff72d9`. Later cap-frame
and candidate-diagnostic changes are validated separately. Findings 109–112
concern inherited/dependency code, not new defects attributed to Frank.

## 109. Source carbon charge must not permit five bonds — fixed

[AtomWorks leaving-atom resolution at the previous shared checkpoint](https://github.com/baker-laboratory/atomworks-dev/blob/774056c7/src/atomworks/io/utils/leaving_atoms.py#L329).

> Can charge correction be separated from the valence budget used to displace
> hydrogens? The authored ACE carbon in 1j8z has a stale +1 charge; increasing its
> valence allowance by that charge leaves five bonds and prevents preparation.

The shared resolver now limits carbon to four bonds before correcting the
charge. Integrated full-file keep/remove-H tests verify the neutral carbon and
its four bonds. The tmol AtomWorks route now scores the previously rejected
acetylated peptide. This is chemistry normalization, not a new charge model.

## 110. Neutralizing one fragment discards other molecules — fixed

[Dimorphite neutralization](https://github.com/baker-laboratory/atomworks-dev/blob/774056c7/src/atomworks/external/dimorphite_dl/dimorphite_dl.py#L290).

> Does the reaction result retain every disconnected component? For a mixture
> containing acid, amine, azide, nitro and chiral phosphate, RunReactants returns
> only the reacting component. The previous engine silently returns acetate.

Neutralize components independently and combine them before restoring original
atom order, maps, properties and conformers. The common connected-molecule path
does not copy a molecule to discover its fragments. The integrated mixture
regression verifies all components and charged motifs together. All original
774 mapped-state comparisons still match. Three comparisons for this newly
introduced mixture intentionally differ from the old, lossy engine; these
failures remain visible in the raw comparison artifact.

## 111. Cache-control flags invalidate the parse cache — fixed

[Parse cache key](https://github.com/baker-laboratory/atomworks-dev/blob/774056c7/src/atomworks/io/parser.py#L63).

> Should storage location and save/load controls be part of chemical identity?
> Saving with load disabled and loading with save disabled produces different
> keys, so the second call never uses the saved parse.

Exclude only cache location/control settings from the chemical parse key.
The existing full 4NDZ workflow, including changed-chemistry cache misses and
its fourfold cached-speed assertion, passes without modifying the test.

## 112. Inter-residue detection materializes large string arrays — optimized

[Inter-residue bond extraction](https://github.com/baker-laboratory/atomworks-dev/blob/774056c7/src/atomworks/io/utils/leaving_atoms.py#L225).

> Can this use the same contiguous residue boundaries as the rest of the
> parser, including assembly/transformation identity, instead of constructing
> and comparing string keys for every atom and bond endpoint?

Use integer residue indices from AtomWorks' patched boundary function and reuse
the extracted bond table during leaving-group resolution. Full-structure tests
duplicate assemblies with repeated author IDs and preserve their heavy graph,
coordinates and inter-residue links through hydrogen regeneration.

Nine warm repetitions on synthetic contiguous residues preserve identical bond
indices. At 1k/10k/100k atoms, medians change from 1.349/17.273/177.221 ms to
0.034/0.187/1.680 ms. At 100k atoms, Python-traced peak changes from 136.014 MB to
3.781 MB. These are isolated extraction measurements, not end-to-end speedups;
tracemalloc excludes native allocations. See `benchmark_residue_bonds.py` and
`results/residue-bond-benchmark.json`.

## 113. An internal alpha-like motif renames a cap's frame — fixed

[Frank's unconditional alpha renaming](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L360).

> Should alpha canonicalization depend on the selected polymer profile? QUI in
> 1xvk is a cap with frame references C2/C/O1. An internal alpha-like motif
> renames these source atoms although the selected cap profile still uses them,
> causing KeyError C2 before parameter generation.

Apply alpha renaming only to an alpha profile. Preserve other profiles' named
frames. The complete RCSB 1xvk fixture is retained with provenance. Its test
explicitly excludes free Mg, completes preparation, checks QUI identity/finite
ideal coordinates, and exercises the remaining construction error below.
This test does not claim whole-complex scoring succeeds.

## 114. Partner backbone discovery and chain ordering disagree — open

[Partner connection classification](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L712)
and [conjugation inference](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L741).

> How are polymer ports established when both neighbors have sidechain
> crosslinks? In 1xvk, N2C has N/C/CB and NCY N/C/SG. Their unfiltered three-port
> profiles are absent, so their legitimate carbonyl links to MVA are treated as
> conjugations. MVA then loses its down polymer port. QUI also occurs after its
> connected partner in input order and is requested as the wrong terminal class.

The current construction rejects MVA and QUI because no prepared type matches
the inferred topology. The new diagnostic reports pose, residue, component,
terminal class and variant even when the candidate set is empty. Previously
it emitted only a generic failure threshold. Do not manufacture a matching
candidate or drop these residues to obtain a numerical pass.

Proposed fix: establish a consistent graph of polymer ports before generating
conjugation patches, using declared sequence and chemically supported endpoint
pairs; resolve mutually crosslinked neighbors together. Then order polymer
blocks from that graph or represent nonsequential backbone edges explicitly.
Require stable results under residue order, repeated instances and chain
permutation. A one-pass partner lookup or choosing the first two ports cannot
safely establish this contract.

## 115. Proximity adds bonds absent from the supplied graph — open

[Cross-residue detection](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_detect.py#L852).

> Why does proximity add arbitrary polymer heavy-atom links even when an
> explicit bond graph is present? Native 145d contains 20 declared inter-residue
> phosphodiester bonds. The proximity pass instead reports many base/sugar
> contacts as MCY/5CM conjugations, including C1', C2', C4' and ring nitrogens.

This is a topology and efficiency defect: false sites generate large numbers
of combined variants before refinement fails. The supplied graph and the
inferred partner map were inspected separately; the false links are not bonds
returned by the native reader. Simply disabling the pass for supplied bond
tables removes these phantom sites but exposes a separate missing MCY terminal
mapping and changes the existing glycan-without-link-record fallback contract.
That experimental change is **not shipped** and no existing test was weakened.

Proposed fix: distinguish complete authoritative connectivity from explicitly
incomplete connectivity in the input contract. Let the shared parser resolve
supported polymer/link metadata; request chemistry-constrained inference only
for inputs declaring missing connectivity. Record each inferred edge. A close
contact alone, even between polymer residues, cannot establish a chemical bond.

## 116. Combined patches can retain removed torsion atoms — open

[Conjugation torsion generation](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_conjugation_patches.py#L190).

> Are generated torsion frames validated against the complete patched graph,
> including earlier terminal patches? In 145d, generated 5CM:na5prime:conj_C5'
> variants contain a chi frame referencing P after na5prime removed P. The
> analogous refinement failure in metal-excluded 1aym references absent H.

Reject incompatible patch combinations before building refined residue types;
select a valid retained heavy-atom frame where the same physical torsion exists.
Do not just delete a sampling torsion to suppress KeyError. Bound combination
allocation and generate only the combinations required by actual input sites.
The 145d phantom-site explosion must also be addressed at its source (115).

## Finding 99 update: equivalent leaving branches — fixed with explicit ambiguity

AtomWorks now retains separate leaving branches and verifies rooted chemical
equivalence for single-bond alternatives. It removes only the number required
by newly declared bonds, preferring absent/unobserved alternatives. Repeated
resolution retains the already selected state. The 8OG phosphodiester keeps
observed OP2; both native and AtomWorks tmol routes score it. Ambiguous observed
alternatives still require explicit connected chemistry. The source-heavy-atom
guard remains, including its AF3 OXT rejection; this is not complete provenance
tracking or permission to silently delete observed atoms.

## Model input interfaces: removed, with executable application examples

The OpenFold and RoseTTAFold2 constructors, ordering/packed-type factories,
exports and vendored model tables are removed fully. No compatibility wrappers
remain. This removes another 1,922 production/vendor lines beyond the earlier
1,532 net shared-chemistry removal. The original model predictions remain as
fixtures. Their layout metadata is test data, not an installed model API.

`docs/model_inputs.rst` includes executable examples from
`docs/examples/model_inputs.py`. Applications provide named residue/atom layouts,
prepare mappings once, and bind tensors through Torch into generic canonical
construction. OpenFold uses AtomWorks atom14 metadata; RF2 accepts the model's
actual tables and explicitly chooses preserved or rebuilt hydrogen coordinates.
Exact C-alpha coordinates, finite scoring, nonzero input gradients and zero
gradients on rebuilt RF2 hydrogen slots are checked on saved predictions.

The example caches mappings, not pose topology. A fully unified prepared-topology
atom14/atom37/backbone4 API remains a proposal. AtomWorks remains the shared file
chemistry dependency; native CIF reading is still the default while authored
identity/normalization policy is settled. NaN completion retains identity; it
neither implies finite imputation nor preserves a Torch tape through NumPy.

## Test consolidation and remaining scientific validation

At the user's request, AtomWorks additions decreased from 151 cases in ten files
to 48 cases in six files. Full structures now test parsing, protonation, assembly
identity and graph/coordinate retention together. Boundary/fallback/concurrency
checks remain where full structures cannot trigger the contract. Existing
upstream tests, tolerances and stored chemistry were not changed.

The full AtomWorks IO rerun has 766 passes, 13 skips and four 1twr failures.
All four reproduce on unchanged dev 59afb1e2. The suite ran the final consolidated
implementation before the allocation-only neutralization cleanup; its five
identity cases were rerun after that cleanup and pass. No ML-suite or complete
documentation-build result is claimed. Final GPU and expanded-corpus evidence
is recorded in the accompanying continued-validation artifacts.

## Expanded PDB failures: proposed follow-ups

The 96 PDBs referenced by the AtomWorks IO tests are separately checksummed in
`results/continued-pdb-input-provenance.json`; the corpus runner accepts their
case manifests and preserves all 192 native/AtomWorks trial records. Free metal
ions are explicitly excluded in this extension. Metal-containing components
remain present and deferred. The files overlap in chemistry and are not 96 new
independent validation systems.

| Observed failure | Proposed action / existing review contract |
|---|---|
| 1xvk MVA/QUI; 3bdp empty candidate set; 4ndz BDP endpoint | Resolve polymer ports and nonsequential chain topology before type selection (114); avoid component-name/global-instance inference. |
| 145d P; 1aym H during refined-type setup | Validate patch composition and frames on the final graph (115–116). |
| 1iau ASJ OD1; AtomWorks 1en2 NAG O1; 5t4j ABU O | Compare source chemistry, variant deletion and alias provenance; preserve the current unknown-heavy-atom rejection instead of deleting coordinates to pass (93). |
| 5xag / 5xaf candidate failures | Reconcile provided atoms with generated terminal variants and include variant charge/atom completeness in preparation validation (22,95); do not refresh score goldens. |
| 2msb unresolved MAN; 5hs6 incomplete J3Z; unanchored PCA in 1aco | Define anchored ligand completion and stereochemistry policy, separate from polymer sidechain packing (97). Unresolved identity should remain present. |
| 3ne7 / 8cuy UNL without bonds; 4js1 PO4; 1d9d U31 | Require explicit unknown chemistry; audit source bond orders/charges and supported valence before SMILES conversion. Do not infer unknown bond orders solely from distance (92,108). |
| 5e5j D; 4hbt CL; 7zcy OH; native 1arx IOD / 155c UNK | Define isotope and one-atom ion handling separately from organic molecule preparation, and reject underdetermined polymer types earlier. No isotope or ion force field is fitted here. |
| 3nez NRQ lookup | Validate all preparation-context mappings before construction and report the missing component/variant; investigate the fragment/conjugation lookup rather than skipping the component. |
| SF4, HEM, VER, WO4, CHL, IUM, ALF | Metal/unsupported-element parameterization is deferred as requested. Free-ion exclusion does not remove these components. |

Each numerical pass still requires finite scores/gradients, nonincreasing energy
and unchanged constructed topology. Partial means the constructor excluded
additional residues under its current policy. Convergence and connection-length
alerts are recorded independently; no default attachment model was silently
changed to improve the success count.
