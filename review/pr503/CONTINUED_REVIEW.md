# Continued review: integrated AtomWorks inputs and model tutorials

Latest follow-up: tmol `cdca836b2`, AtomWorks `4af94d0c`. Earlier checkpoints
below retain their original scope; current evidence is in
[chemistry-followup-validation.json](results/chemistry-followup-validation.json).

## 118. Mol2 delocalization is restored as invalid aromatic chemistry — fixed

[Frank's source annotation restoration](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_rdkit_mol.py#L166).

> Can annotation restoration distinguish aromatic ring bonds from Tripos
> delocalized carboxylates and the single bond joining two aromatic rings?
> The O-glycan SIA conversion restores non-ring aromatic atoms after bond
> normalization, while its carbonyl remains single and both oxygens are negative.

Require ring membership for restored aromatic atoms/bonds and reuse AtomWorks'
carboxylate bond-order repair on generated mol2 geometry. Run that repair only
when a Tripos carboxylate still lacks its double bond; valid carbonyls avoid
the extra copy/sanitization. Existing partial charges are retained. Integrated
acetate and biphenyl workflows both fail before the fix and pass afterward,
including chemical identity, parameter generation, scoring and gradients.
The combined semantic/typing selection has 115 CPU passes.

## 119. Conflicting declared partners overwrite one connection port — fixed

[Frank's conjugated connection construction](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/details/_select_from_canonical.py#L1210).

> Can connection occupancy be validated before writing the pose graph?
> Complete 1AYM declares both GLY N–MYR C1 and GLY CA–MYR C1 in struct_conn.
> Assigning both to MYR's one port overwrites an edge and leaves a nonreciprocal
> graph; batched device writes also make the surviving edge undefined.

Validate each port's partner on the host before device writes, allowing
identical repeated edges but rejecting different partners with residue, atom
and port names. The unmodified compressed CIF is now a regression fixture with
source provenance. Both readers reject the conflict without mutating their
input bond tables. This fixes handling of invalid input; it does not silently
choose one source bond or make original 1AYM a successful minimization case.

## Attachment parameter convention: follow Frank's branch

[PR503's description](https://github.com/uw-ipd/tmol/pull/503) specifies Hahnbeom's
gen_bonded hybrid model. Its Cartesian generator uses ideal generated geometry
with `K=300` for lengths and `K=80` for angles; generic bonded scoring supplies
proper/improper torsions. Existing MMFF use for charges and conformer cleanup
does not imply MMFF harmonic force constants should replace those values.

The selected attachment implementation must preserve that convention. A CPU
diagnostic now generates complete capped biotin, O-glycan and N-glycan geometry
and verifies native attachment energy/gradients against an independent Torch
expression in whole-pose and weighted block-pair modes. All six comparisons
pass. This diagnostic still uses the prototype's MMFF coverage checks and does
not independently validate stereochemistry. Default attachment/local parameter
integration remains outstanding; the MMFF harmonic and local-delta prototypes
remain private diagnostics. No new default force field is installed.

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

## 114. Partner backbone discovery and chain ordering disagree — targeted fix

[Partner connection classification](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L712)
and [conjugation inference](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_preparation.py#L741).

> How are polymer ports established when both neighbors have sidechain
> crosslinks? In 1xvk, N2C has N/C/CB and NCY N/C/SG. Their unfiltered three-port
> profiles are absent, so their legitimate carbonyl links to MVA are treated as
> conjugations. MVA then loses its down polymer port. QUI also occurs after its
> connected partner in input order and is requested as the wrong terminal class.

At `102cda060`, resolve unique chemically supported peptide endpoint pairs for
all partners before generating conjugation patches. Retain explicit sequential
and nonsequential polymer bonds during canonical conversion and determine termini
from occupied ports, including intrinsic caps. D-serine's DSN input identity
resolves its DSER database type. Complete 1xvk now retains all 18 links, with
finite scores/gradients and decreasing energy in ten LBFGS steps through both
readers and both original/reversed residue orders on CPU. The same test exposed
the independent scoring defect in 117. Broader corpus/CUDA validation is pending;
this is not a claim that every ambiguous polymer has an inferred profile.

Canonical conversion also preserves declared disulfides when sulfur coordinates
are missing or distant. Connection installation and reciprocal-port validation
use batched tensor operations instead of per-edge device scalar reads.

## 115. Proximity adds bonds absent from the supplied graph — fixed

[Cross-residue detection](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_detect.py#L852).

> Why does proximity add arbitrary polymer heavy-atom links even when an
> explicit bond graph is present? Native 145d contains 20 declared inter-residue
> phosphodiester bonds. The proximity pass instead reports many base/sugar
> contacts as MCY/5CM conjugations, including C1', C2', C4' and ring nitrogens.

Remove the spatial conjugation pass. The parser owns supported bond inference;
ligand detection consumes the supplied graph. The updated integration assertion
requires an explicit glycan bond, and confirms that coordinates cannot add or
remove it. This deliberately replaces the old proximity-fallback contract.
Recognize terminal nucleoside backbones before attempting peptide cap inference:
MCY's base N4 must not become a peptide endpoint. Complete 145d now retains 24
DNA blocks and exactly 20 phosphodiester links, with finite scoring and decreasing
energy during the integrated CPU minimization check. No phantom conjugation
variants are allocated. The complete fixture and provenance are retained in tmol.

## 116. Combined patches can retain removed torsion atoms — guard implemented

[Conjugation torsion generation](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_conjugation_patches.py#L190).

> Are generated torsion frames validated against the complete patched graph,
> including earlier terminal patches? In 145d, generated 5CM:na5prime:conj_C5'
> variants contain a chi frame referencing P after na5prime removed P. The
> analogous refinement failure in metal-excluded 1aym references absent H.

Reject incompatible combinations before mutating a residue: a new torsion must
reference retained atoms/connections, and a retained connection's existing
torsion cannot lose supporting atoms. Both terminal/conjugation patch orders
are checked together using the generated 5CM bundle. Valid conjugations retain
their sampling torsions. This is a support guard, not a general alternate-frame
generator; the full 1aym corpus rerun is still required.

## 117. Inter-block generic torsions depend on residue order — fixed lookup

[Native lookup](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/genbonded/potentials/genbonded_pose_score.impl.hh#L90)
and [table generation](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/genbonded/_genbonded_energy_term.py#L499).

> Can intra- and inter-block lookup share reversed matches, bond bins and
> multiplicity priority? Reversing complete 1xvk's residue order preserves every
> coordinate and type but changes gen_torsions from 13.692318 to 84.313545.
> The entire 70.621246 energy shift comes from this term.

Build native tables from the database's bidirectional lookup index, with one
stable rank per multiplicity/source-order priority. Native lookup uses that
rank. For a 3+1 atom path, select the internal central bond's order/ring bin;
the external connection's bin applies only when it is the central bond. Share
this behavior across pose/rotamer forward and backward paths. The integrated
1xvk order/score/minimization regression now passes on CPU; CUDA is pending.
No force-field coefficient or score golden was changed. A separate limitation
remains: ring membership of a connection closing a ring across blocks is not
represented by the existing per-type connection metadata.

## Attachment parameter source: follow Frank's branch

The user selected Frank's intended scoring model. His
[PR description](https://github.com/uw-ipd/tmol/pull/503) specifies Hahnbeom's
generic-bonded hybrid mode. In this implementation, the generic term scores
proper/improper torsions; his
[Cartesian generator](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L32)
uses `K=300` for lengths and `K=80` for angles, with equilibrium values from
generated ideal coordinates. Its intra-ligand convention is direct source
evidence; applying that convention to missing attachment records is the intended
extension, not a claim that Frank already supplied those records.

Keep the existing Rosetta/generic ownership and generated-geometry conventions.
The private MMFF94 harmonic-curvature prototype is diagnostic only; do not install
it as the default or interpret its successful stiffness test as reference-model
validation. Frank's existing MMFF94 charge generation and conformer cleanup are
separate uses of MMFF and remain part of his pipeline. The default-source choice
is settled; implementation and validation of missing attachment/local terms
(38/43/58) remain outstanding. Reuse his generator for complete capped chemistry
and preserve curated peptide references rather than substituting new constants
or fitting targets to the coordinates being scored.

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

`docs/model_inputs.rst` links the executable Colab tutorial
`notebooks/example_02_model_inputs.ipynb`. Applications provide named residue/atom layouts,
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
