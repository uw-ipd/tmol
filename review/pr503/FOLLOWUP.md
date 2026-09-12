# Correctness and performance follow-up

Active objective: address the review changes and establish correctness and
performance for supported CPU/GPU workloads. Starting implementation/reference:
`96ff844e82ecb7dff0958d6b419e7aa2a946694a`. The original review and results remain
historical evidence, not a claim that the follow-up is complete.

## Completion requirements

| Area | Required evidence | Status |
|---|---|---|
| Import/closure inference (1) | Fresh checkout collection; explicit/inferred closure, padding, breaks and caps on both devices | Existing replacement; extend audit |
| Mixed chirality (2) | Published reference parameters/formula; LL/DD/LD/DL permutation and reflection energies/gradients; whole-pose and packing parity | Rosetta mixed distribution and shared derivatives implemented; independent LL/LD/DL/DD energy/gradient, permutation and reflection checks pass on CPU/CUDA; per-term whole-pose/block-pair reflected gradients also pass CPU/CUDA; default reflected conformer sets and all per-term interaction matrices now agree on CPU/CUDA; single-position CUDA packing reaches the exhaustive minimum on both sides (comments 73–76, final manifest below) |
| Group identity and safety (3–7,9,30,31,33–37) | Real multi-pose/multi-database groups, repeated chi names, jagged/empty counts, early rejection; native parity | Capped anchors, rigid cyclic cores, external attachments and sampled pendant branches tested CPU/CUDA, including packing and two-pose reuse; later masks now constrain geometry and ownership; task-dependent tests recorded below |
| Sampling budgets/caches (8,14,27) | Explicit budget semantics; task propagation; immutable sampler reuse; actual library/extra/proton counts; reproducible HYP count | Explicit task overrides and actual group/library bounds implemented/tested CPU/CUDA; aggregate/default budget policy and adaptive library expansion remain open |
| Residue identity/completion (10–12,16,32) | Insertion codes, chain identity, explicit/CCD authority, missing atoms, custom names; realistic scaling | Reader annotation reuse and finite-geometry repair checks pass; AtomWorks parser/converter profiles recorded; broader identity/authority contracts remain open |
| Content/profile caches (13,20,29,39,40) | No serialization aliases; safe object lifetimes, bounded memory; repeated preparation | Content framing and shared bounded weak caches fixed for rotamer/alpha/NA profiles; identity/lifetime/LRU/concurrency tests pass; remaining caches still need audit |
| Group kinematics/performance (15,17,35) | Profile CPU/GPU; batch invariant work; retain all covalent constraints for tree/cyclic/multiple-anchor topologies | Bounded conformer batches and 32-entry shape caches verified/profiled; independent axes preserve rigid cycles/external boundaries; task-imposed fixed members now constrain axes; correlated ring-pucker sampling remains open |
| Native maintainability (18) | Shared improper enumeration with unchanged canonical scores and independent analytic/numeric derivatives | Shared helper in all four paths; canonical references, numerical gradients, independent Cartesian reference and group packing pass CPU/CUDA (238323, 238529) |
| Duplicate work (19) | Removal covered by chemistry regression suite | Existing fix; final suite pending |
| Correct test parameters (21) | Prepared database used throughout examples and packing; generated parameter coverage | Existing correction; final audit pending |
| Terminal caps (22) | ACE/amide caps build intact chains, finite coordinates/gradients, correct bond topology; CPU/GPU packing | ACE/NH2/NME construction, bonds, score/gradient and rotamer construction pass CPU/CUDA; amide geometry and actual packing pass CPU/CUDA (237089) |
| Reference scores (23) | Explain reference differences; independent scoring checks; reproducible pinned structures/parameters | Pending; no blind golden refresh |
| D geometry (24) | Signed stereocentre volumes and actual library sampling after repacking; reject mislabeled geometry | Every offered D rotamer and packed result retains signed alpha-centre volume on CPU/CUDA (237089); broader stereocentre coverage remains |
| Fold trees/fragments (25,26) | Branch-point invariant, fragment restoration/minimize/pack/DDG and multi-pose checks | Existing fixes; final suite pending |
| Scientific ownership (general 2,7) | Independent potential checks and canonical change audit; sampling/scoring reference separation | Pending |
| Connection bonded potentials (38) | Every declared attachment has explicit length/angle energy ownership; independent stretching/bending forces; parameter persistence; no double counting; packing and Cartesian minimization | Backend, bundle persistence and private capped-MMFF harmonic generator validated CPU/CUDA; default integration/local chemistry corrections remain pending, so default preparation retains the 14 missing attachment potentials |
| API/authority/reproducibility (general 1,6,8,9) | Executable input-route examples, settings/seed provenance, documented migration and fresh-checkout CI | Pending |
| Workload scale/release (general 4,5,10) | Explicit matrix of chemistry/topology/input/batch/stage/backend coverage; paired profiles/timings/memory, no masked regressions | Paired baseline/candidate 19-fixture CPU/CUDA matrices recorded; larger-scale and full-stage matrix pending |

Performance measurements must separate JIT/cold setup from warm execution,
synchronize CUDA around timed regions, use the same environment for paired
comparisons, and report per-stage latency and memory. Python/native profiling
is separate from uninstrumented timing. No speed improvement counts if it
drops chemistry, breaks a bond, changes a required sampling contract, or
weakens correctness checks. A smaller passing subset does not complete this
objective.

Baseline worktree: `/mnt/home/kdidi/projects/tmol-pr503-perf-baseline` (detached,
unchanged reference). Working branch: `review/pr503-chemistry-efficiency`.
Live results and diagnostic profiles: `/mnt/home/kdidi/tmol-pr503-followup`.

## AtomWorks investigation (user-requested extension)

See [ATOMWORKS.md](ATOMWORKS.md) for the broader ownership/reuse audit and
[atomworks_example.py](atomworks_example.py) for the executable parse-once route.
AtomWorks local branch `review/tmol-pr503-shared-chemistry`, commit `4762d7e5`,
contains isolated chemical scopes, correct scoped caches, faster leaving-group
traversal, generic polymer-entity handling and corrected dependency metadata.
Its 52 targeted/existing tests pass, and a fresh core-only installation parses
without PyTorch. Median paired parser speed ratio: 1.23x across the 16 previously
passing fixtures, with identical full atom-coordinate and bond inventories.
All 19 reader fixtures now parse.

Tmol's annotation consumer and two additional chemistry fixes (replace a free
terminal hydroxyl during capping; retain one phosphate double bond during atom
completion) enable all 19 AtomWorks fixtures to prepare, construct, score,
differentiate and build rotamers on CPU and H200 CUDA. The 19-fixture native-reader
matrix also passes these stages on both devices. This is progress toward the
full workload matrix, not a claim that minimization, packing, scale, all input
routes, or every topology has been verified.

Slurm evidence: 236298 (38 focused CPU/CUDA tests pass); 236514 (native-reader
CUDA baseline/candidate stage comparison); 236725 (216/224 expanded chemistry
tests pass and 19/19 final AtomWorks CUDA stage runs pass). The eight failures
are the existing noncanonical score references, four classes on each device.
HYP sampling counts, budget enforcement, broader native/topology work and the
remaining completion requirements above are still open.

Initial CUDA profile runs used an unindexed `cuda` device that failed the
Dunbrack sampler's device-equality assertion; corrected `cuda:0` runs supersede
those rotamer results. No failed stages are counted as successful measurements.

### Additional sampler checks

Job 237089 passes 29 sampler/stereochemistry/cap tests and actual NH2/NME
packing on both CPU and CUDA. Excluding unrelated protein/glycan chi records
from NA annotation reduces the glycan fixtures’ proton sampling tensor by 80%.
Single-run annotation timings and exact sizes are recorded in
`results/na-annotation.json`; these timings are diagnostics, not a repeated
benchmark. HYP’s 18 conformers are two library states times nine expanded
hydroxyl angles, verified independently on CPU and CUDA (237980). Budget enforcement is still open.

Hydrogen optimization was found to move terminal generated 5CM heavy atoms by
up to 2.44 Å. Its jump root was O5′, which also carries a terminal proton chi.
Reference-score updates remain deferred while this geometry defect is corrected
and tested across modified nucleotides.

Job 237980 passes 73 nucleotide/OptH/NA/noncanonical sampler tests across CPU
and CUDA, including heavy-atom preservation in every offered proton rotamer
and actual packing for 5CM, 8OG, PSU, 2OM and TTD fixtures. The generated
nucleotide jump root now follows the sugar side of its glycosidic torsion,
with an interior-backbone fallback for unclassified nucleotides. In the 5CM
diagnostic, the maximum heavy displacement falls from 2.44 Å to 0.0000049 Å.
Old noncanonical score references are not refreshed by this fix.

Job 238307 passes 33 OptH/NA/cache-setting tests on CPU and CUDA. Alternating
expanded/unexpanded budgets on shared residue types now rebuilds the relevant
RT/PBT tables; changing NHQ flips updates buildable types; changing chemical
element assignments updates sugar-ring recognition. One most-recent table is
retained per object, avoiding unbounded per-configuration caches. This fixes
sequential sampler reuse; it does not implement the task-level budget API.

Job 238312 passes 26 group regression/packing tests across CPU/CUDA with four
existing CPU skips for large full-packing cases. Anchor-library sampling now
uses an isolated task mask containing only group anchors. Exact anchor output
arrays match the previous implementation on all three fixtures and both
devices. Seven paired warm measurements show CPU stage speedups of 1.46×,
1.80× and 1.79×; GPU stage latency is essentially unchanged (about 1%).
Enumerated rows fall from 1781→60 (biotin), 3028→3 (O-glycan), and 5849→66
(N-glycan). These are stage measurements, not end-to-end packing speedups.
See `profile_anchor_library.py` and `results/anchor-library-{cpu,cuda}.json`.

Jobs 238323 and 238529 validate the shared connection-improper enumerator.
The former passes 48 canonical cartbonded/group-packing/mirror-image cases
with four existing large-packing CPU skips. The latter passes four independent
peptide/proline improper energy and gradient cases on CPU/CUDA: Cartesian
plane normals and PyTorch autograd are evaluated at distorted coordinates,
with asymmetric block-pair weights. No score/parameter values changed in this
refactor. CPU-only existing cartbonded suite: 16 pass, 11 CUDA skips.
Machine-readable cases for these and the preceding follow-up runs are in
`results/followup-geometry-sampling-native.json`.

### Explicit budgets and CPU packing precision

Explicit task limits now survive `SetPackerTask` conversion and configure private
NA/OptH sampler views. A reused caller-owned sampler retains its settings.
Groups count actual anchor-library rows, every member and the offered current
state before allocating their Cartesian product. When child torsions freeze,
the empty product still supplies one state. Proton chi retain a placement even
when alternative means are removed. Required library states that cannot fit
raise before final rotamer arrays are allocated; Dunbrack also enforces an
explicit task limit at its native count stage. See [BUDGETS.md](BUDGETS.md) for
exact limits and remaining API policy decisions.

The four full O-/N-glycan packing tests previously skipped on CPU are enabled.
This exposed a separate LJ/LK whole-pose accumulation error: adding thousands
of float32 block-pair totals rounded away small contributions. CPU whole-pose
totals now accumulate in double and cast once; pair calculations, output dtype
and CUDA accumulation are unchanged. Packing/whole-pose energy agreement passes
without loosening its tolerance. CPU LJ/LK plus group suite: 38 pass. Fresh CUDA
subset: 18 pass (240101). A separate run passes both CPU accumulation branches,
the formerly failing CPU N-glycan energy case and all 13 explicit-task budget
cases on CPU/CUDA: 16 pass (240099). Earlier failed JIT builds in job 240094 are
superseded by these fresh-process checks, not counted as successes.

Paired ubiquitin batch 1/4/16 benchmarks reduce CPU discrepancy from a precise
block-pair sum from 0.00031231 to 0.00000944175 (33-fold). This accuracy fix has
a measurable CPU cost: roughly 3–5% in the initial comparison and the first two
alternating-order rounds. The third round's baseline slowed by over 2x for an
unresolved environmental reason; all raw data are retained, and that anomalous
round does not establish a speedup. The initial CUDA comparison changes latency
by about 0–1%. This is an accuracy/performance tradeoff, not a universal speedup.
Scratch memory is 24 bytes per CPU pose; no CUDA scratch allocation is added.
See `profile_ljlk_accumulation.py` and `results/ljlk-*.json`.

With budgets 100/1000 and actual anchor rows held fixed, N-glycan enumeration
falls from 595 to 67 conformers (4760 to 536 member rotamers); chi arrays shrink
from 11900 to 804 bytes. Seven warm paired samples give stage latencies of
1.571→1.144 ms on CPU and 3.224→2.294 ms on CUDA (240110). Biotin and O-glycan
counts stay unchanged. The comparison loads the prior enumerator and budget
helpers from `fccd9ad5c`; see `profile_group_budget.py` and
`results/group-budget-{cpu,cuda}.json`. These are enumeration-stage results,
not measurements of total packing latency or retained conformational accuracy
at every budget.

### Batched group kinematics

Group conformers now share forward-kinematics scans and each member's inverse
kinematics runs in batches. Global DOF destinations are transferred/validated
in bulk. Each native call handles at most 4096 atom rows, except an indivisible
single group larger than that. This bounds transform workspace without dropping
conformers or changing the configured sampling budget.

Final tests: **17 CPU + 17 CUDA pass**, including scalar-versus-batch comparisons
in float32/float64, empty/one/multiple conformers, chunk boundaries, a reused
sampler on two real poses with different anchor geometry, and full biotin,
O-glycan and N-glycan packing/bond/energy checks (CUDA job 240294). Full rotamer
arrays match the scalar implementation exactly on CPU and within 0.000046 Å
maximum coordinate difference on CUDA; existing bond and score tolerances pass.

Seven alternating-order warm samples against `09258fbad`, with identical
rotamer counts, show group-stage speedups of 2.93–3.21× on CPU and 7.25–8.41×
on CUDA. Full rotamer construction latency falls 6–11% on CPU and 21–33% on
CUDA. These do not measure the full annealing/packing pipeline.

CUDA temporary group-stage allocation peaks are 1.31–1.44 MB, compared with
0.54–0.70 MB for the scalar loop. The initial unbounded batch version needed
up to 3.38 MB. Bounded batches retain most of its speedup with less temporary
memory, but do not claim lower memory than the scalar baseline. Measurements
exclude persistent input/output tensors and report PyTorch allocated memory,
not CUDA allocator reservations. See `profile_group_kinematics.py` and
`results/group-kinematics-chunks-{cpu,cuda}.json`; the initial unbounded-batch
measurements are retained as `results/group-kinematics-{cpu,cuda}.json`.

### Shared profile cache lifetime

Alpha and nucleotide profiles also used unbounded integer-ID caches. They now
share the same weak-identity LRU implementation as rotamer-reference profiles,
with 32 entries per cache, referent checks and expiration callbacks. Cache
values must not retain their database, and callers must publish a new database
instead of mutating a cached one. Concurrent misses may compute twice but
publish one value; this does not make mutation of shared RT/PBT sampler tables
thread-safe.

The affected alpha/nucleotide preparation and scoring suites pass **136 cases**
across CPU/CUDA (240300 plus the six separately executed nonstandard-backbone
score smokes in 240302). These finite-score checks do not resolve the separate
pinned-score differences tracked above. Five cache tests cover owner identity,
real ChemicalDatabase collection, LRU bounds, configuration separation,
simulated stale identities and simultaneous misses.

A 100-database/three-profile churn experiment retains zero profile entries,
versus 300 previously. Traced Python memory after collection falls from
1,153,528 to 3,288 bytes; peak traced allocations fall from 1,361,972 to
513,604 bytes. Alpha/DNA/RNA profile fields match exactly. Seven warm samples
of 30,000 lookups measure 0.098→0.437 µs per lookup: a small absolute cost for
lifetime/identity checking, not a lookup speedup. This measures Python cache
allocation, not process RSS or all ligand preparation memory. See
`profile_polymer_cache.py` and `results/polymer-cache.json`.

### Chemical geometry of group conformers

Checking only inter-block bonds missed internal ring damage. The original
O-glycan sampler stretches an NGA C5–O5 bond from 1.443 to 3.768 Å in offered
conformers. Generated attachment chi at anomeric carbons used a ring bond as
an independent axis. Such sites now use the bond across the residue boundary;
central atoms resolve through group connections instead of assuming both are
local to the chi's owning residue.

A separate frame issue affected lysine/asparagine attachments: a departing
hydrogen's icoor can reference another hydrogen on the same centre. That is a
valid placement frame, but may not be a bonded torsion path. References now
come from the chemical bond graph. Final-rotamer tests independently resolve
and measure every sampled torsion, check every internal and inter-block bond,
all bond angles and every four-neighbor stereocentre.

**51 group cases pass on CPU/CUDA** (240633), including full packing and energy
agreement, scalar/batch kinematics and the new chemical-geometry checks.
**42 covalent-input/topology cases pass** (240951). These extend the evidence;
older bond-survival tests and scalar/batch parity alone did not prove internal
ring integrity. Through AtomWorks, all three fixtures pass the stronger geometry
checks on CPU and CUDA, plus six full CUDA bond/packing-energy cases (241055).
`check_group_input_route.py` runs the same assertions with an explicit reader.

Seven alternating-order paired preparation/construction samples replay the
previous patch and group-coordinate methods from `fa41a727e`, using the same
current native kernels, input reader and fixed ligand seed. O-glycan's maximum
internal bond error falls from 2.325 Å to 0.000027 Å (CPU) / 0.000033 Å (CUDA).
All three fixtures retain identical rotamer counts and coordinate allocation
sizes. Median preparation and rotamer-construction changes are about 0–1%;
this correction does not trade away the previous batching improvements. The
construction timings include fresh per-pose annotation, unlike the earlier
warm reuse benchmark. See `profile_group_geometry.py` and
`results/group-geometry-{cpu,cuda}.json`.

Group discovery also retains cycle-closing bonds, internal polymer/disulfide
edges and all external attachments. Seven topology shapes, each with two poses,
verify exact internal/external edge inventories on both devices. This is the
information needed for constrained sampling, **not** a claim that cycles or
multiple external anchors are already sampled safely. Rigid cyclic cores,
external constraints, duplicate axes and the generic attachment-grid policy
still need explicit handling and integration tests.

### AtomWorks conversion and missing-geometry correction

The shared-rule audit found that both carboxylate correction implementations
could rewrite bond orders at unresolved coordinates because comparisons with
NaN do not reject a threshold. Both now require finite, nondegenerate local
geometry; valid sites elsewhere in the same molecule still get corrected.
AtomWorks also resets both oxygen charges consistently. Seven tmol regression
cases reproduce the old defect; all 18 new and 25 existing ligand-unit cases pass.
Draft inline comment 32 records the pinned upstream location.

AtomWorks conversion now reads source columns directly and copies only retained
annotations, while preserving input/output independence. Its explicit false
coordinate option now works, and automatic coordinates require finite values.
All 71 affected AtomWorks tests pass; replayed prior methods fail 12 of the new
cases. All 19 AtomWorks-input preparation/scoring/gradient/rotamer fixtures pass
again on CPU. Full matrix data is in `results/atomworks-conversion-matrix.json`.

The paired converter measurements show 1.23–1.47× speed ratios for three CCD
components and all three hydrogen policies. A 3,000-atom synthetic case takes
30.41→20.94 ms. Adding 32 unused U128 annotation columns demonstrates the copy
cost: 109.91→21.31 ms and 49.67→0.40 MB traced peak Python allocations. This is
not native RDKit memory, a typical ligand claim, or an end-to-end tmol speedup.
Exact molecule inventories match in all 11 paired comparisons. See
[ATOMWORKS.md](ATOMWORKS.md) for the shared-API recommendation and remaining
input/converter contracts; reader replacement is still an open gate.
AtomWorks changes are committed locally at `cdda3c07` on
`review/tmol-pr503-shared-chemistry`; they have not been pushed to its upstream.

### Cyclic and externally constrained group sampling

Group identity now follows the declared polymer property, so fully terminal
amino acids still anchor their conjugates after both up/down ports are removed.
Generic conjugated ligands request the existing non-ring heavy-chi records;
their internal torsion definitions alone previously supplied no samples.
Independent samplers are disabled across owned group members, including a
second polymer member, rather than just the primary anchor.

The complete atom graph determines which axes can turn independently: a cycle
edge cannot, and a tree subtree containing fixed external connection atoms,
their neighbors or externally attached polymer mainchains cannot move. Cyclic
cores retain their input geometry, while movable pendant branches still sample.
The primary library is projected onto movable axes, keeping the first distinct
projected rows. Duplicate axes are owned once, and budget depth follows the
actual tree child even when the torsion names its axis in reverse order.
An empty axis set yields one input conformer. Fully constrained anchors skip
library generation; partially constrained libraries can require temporary raw
rows beyond their final group budget, as now explicit in [BUDGETS.md](BUDGETS.md).

Synthetic propylsuccinyl-linked lysines cover a free three-member group, a
peptide bond closing its cycle, and two external ALA–LYS backbone attachments.
The constrained cases retain two pendant chi, nine grid combinations plus
current, and can reduce to exactly one current conformer at a three-member
budget. Every offered conformer preserves all bonds, bond angles, tetrahedral
signs and requested chi. Full packing compares the annealer's energy with the
imposed pose score and preserves every group/internal/boundary bond and angle.
Unrelated ordinary ALA CA–CB idealization is outside that invariant; the first
packing diagnostic exposed a 0.000885 Å change there, not a broken group bond.

Validation: **101 broad CPU/CUDA cases pass** (242362), **six additional full
packing cases pass** (242551), and **eight two-pose sampler-reuse cases pass**
(243134, including the original biotin case). The latter compares batch results
with independent builds after an anchor-geometry perturbation. AtomWorks input
passes all three original fixture geometry checks on CPU/CUDA and six full GPU
packing checks (242741). Separate test runs overlap; these counts are not a sum
of unique tests. Results are in `results/group-constraints-tests.json` and
`results/group-constraints-atomworks-{cpu,cuda}.json`.

A controlled ablation removes only the new axis constraints from the current
implementation. It produces bond errors up to 13.82 Å for the cyclic example
and 26.29 Å for the external example; the valid outputs stay below 0.000003 Å.
Those invalid ablation conformers are not an accuracy-matched performance
baseline and were not used as packing inputs. This is not a pristine upstream
reproduction or independent force-field fitting.

Topology and rotamer trees each use a 32-entry LRU on their owning PBT; external
constraint ports are part of topology identity. The raw spanning tree is reused
when constructing its rotamer tree. Seven alternating-order warm pairs compare
rebuilding these shapes with reusing them, with identical valid conformers and
coordinates. Cache reuse makes the measured library/enumeration/group-coordinate
stage **1.25–2.04× faster on CPU**, **1.17–1.57× on CUDA** across six cases. These
are not full-pack speedups or process/GPU-memory reductions. Persistent caches
trade bounded retained data for avoided work; eviction and configuration tests
verify the bounds. See `profile_group_constraints.py` and
`results/group-constraints-{cpu,cuda}.json` (CUDA 243133).

At this milestone, task-mask handling was still open: disabling the group sampler emitted its
235 conformers plus fallback (236 per member); disabling one member gives
235/235/236 counts. The next section records the fix and its separate tests.

### Task masks, fixed members and rigid-member ownership

The group sampler now checks both allowed residue types and its per-block mask
before generating a library or Cartesian product. A fully disabled group emits
no rows, leaving one input rotamer per member. A partially disabled group fixes
every atom in inactive members, retains compatible motions elsewhere, and emits
correlated rows only for active members. The bounded topology cache includes
fixed-member identity. It never stores coordinates or a previous task's masks.

Limits count `active_members * conformers + fixed_members`. Freezing the second
lysine in the diagnostic yields 10/10/1 member rotamers instead of 235/235/236.
Freezing both lysines can retain all nine pendant grid states plus the input
within 12 total rotamers. Without an anchor library, the complete anchor remains
fixed. A member with no local heavy-chi samples still advertises group ownership,
preventing an extra independent fallback rotamer from disrupting correlation.

Tests cover free/cyclic/externally attached crosslinks, sampler and packing masks,
each individual fixed member, two or all fixed members, budget accounting,
all offered bond lengths/angles/stereochemistry, actual packing/energy parity,
rigid-linker sampling definitions, a library-free anchor and three-pose batches
with different masks using the same sampler. Final device results are recorded
in `results/group-task-masks-tests.json`. Slurm **243810: 121 passes** across
CPU and H200 CUDA. The earlier broad run **243626: 163 passes, 12 failures**
used the old test helper, which unconditionally dereferenced group collapse
when only one member was movable or all members were fixed. Production packing
already guards that case; the corrected helper follows production and all 12
cases pass in 243810. They were not skipped or removed. Other broad group tests
passed in 243626. Run counts overlap and should not be summed as unique tests.

Final command: `python -m pytest -q
tmol/tests/pack/rotamer/test_group_task_masks.py
tmol/tests/pack/rotamer/test_group_sampling_regressions.py` in the recorded CUDA
container/environment. The independent CPU environment passed 58 selected
tests, followed by three rigid-member and three library-free cases.

Still open: changing member chemical types, reenabling conflicting independent
samplers after group setup, multiple member libraries, free groups without a
polymer anchor, coupled ring conformers, and remaining design/multi-database cases.
Disabling a sampler is covered here; arbitrary combinations of additional samplers
and chemical design are not claimed to be supported by this fix.

### Shared protonation identity contract

AtomWorks branch `44641189` preserves atom maps through reaction provenance,
retains ordered molecular products instead of an unordered SMILES roundtrip,
and protects the charge-separated organic-azide motif during neutralization.
The 34 identity/protonation tests pass. The cross-project 21-case diagnostic
records before/after full ordered chemical-state/map inventories: 12 AtomWorks
identity failures before, zero after; the azidoethane charge mismatch at pH 2
also disappears. See [ATOMWORKS.md](ATOMWORKS.md). This closes another prerequisite
for API consolidation; tmol still retains its vendored engine until the shared
dependency/version and public ownership contract are established.

### Noncanonical reference audit and fixed-input replay

The current standalone CPU run has two failures (DNA and beta peptide), with
HYP and TTD matching the old references. Repeated preparation at seed 20250828
within each environment produces exactly the same generated chemical/cartbonded
records and input coordinates. Across the standalone and container environments,
198 HYP, 2,129 beta-peptide, 631 modified-DNA and 1,304 TTD parameter fields change.
Those include generated equilibrium geometry and icoors. Unresolved HMR sidechain
placement can differ by over 7 Å. CPU and CUDA in the same container have identical
generated parameters, with raw reconstruction differences below 0.000008 Å.

Replaying the standalone run's existing `.tmol` parameter export plus exact
coordinates in the container removes that preparation variable. Across 192
per-term comparisons (four classes, raw and OptH coordinates, 24 score types),
the largest CPU difference is 0.00000191. The largest CUDA difference is 0.002636
in the HYP omega term; all remain within the existing score-test tolerance.
The replay asserts full atom identity before applying saved coordinates.

The baseline/candidate CUDA preparation records differ only in the six patched
jump-atom fields for each modified nucleotide. Their raw coordinates are identical.
The current DNA OptH result keeps heavy atoms fixed, reducing `cart_lengths` from
the broken-geometry reference near 253 to approximately 11. This change is covered
by the earlier independent heavy-atom invariants. The old beta-peptide reference
near 663 LJ repulsion remains unreproduced; the available checkout produces about
121–122 in these environments. Its historical generated inputs and environment
were not recorded, so the audit does not claim an exact reconstruction of that
number. Old goldens remain unchanged.

Scripts: `diagnose_noncanonical_scores.py` supports generated runs and `--replay`
of exported parameters/coordinates. Slurm 244280 compares baseline/candidate and
container CPU/CUDA generation; 244831 replays identical inputs. The standalone
score test, raw comparisons and replay records are kept with this audit.

### Missing connection stiffness: next implementation gate

The parameter-ownership audit found that conjugation patches add chemical
connections and charges but no matching cross-connection bond/angle energy rows.
The unmatched native cartbonded lookup skips these paths. Generic bonded scoring
supplies torsions, not a replacement bond-length or bond-angle term.

At follow-up `026475f0e`, all 14 attachment bonds in the three existing fixtures
(one biotin, seven N-glycan and six O-glycan) have effectively zero stretching and
bending stiffness on CPU and CUDA. Three normal peptide-bond controls return
369.445 kcal/mol/Å² and nonzero angle response. `diagnose_connection_stiffness.py`
uses a rigid connected component after cutting one graph bridge, preserving every
other bond. It differentiates the energy analytically and differences projected
forces at ±0.1 Å. The optional biotin experiment invokes the real full-score
Cartesian minimizer for 100 iterations on the ligand coordinates, keeping the
protein fixed. The unperturbed 1.329 Å link lengthens to 1.660 Å; displaced starts
end at 2.150 and 2.556 Å, each with a lower score.

CUDA reproduces the full minimization behavior at 1.658, 2.120 and 2.558 Å
(244924). Seeded stretching/bending probes on CPU and CUDA are recorded in
`results/connection-stiffness-{cpu,cuda}.json` (CUDA 244891); full minimization
records are in `results/connection-minimize-{cpu,cuda}.json`. The fixed-input
score replay includes all 13 exported parameter/coordinate/score files with
SHA-256 checksums in `fixtures/noncanonical-score-replay/`. Raw scores and
parameter/coordinate comparisons are in `results/noncanonical-score-audit.json`.
All four Slurm audit jobs completed successfully; they are diagnostics exposing
the missing potential, not a passing suite establishing that it is repaired.

This is independent of the passing packing-geometry checks: those retain bond
geometry through kinematics. The next change must supply actual bonded potentials
with explicit chemistry/connection ownership, preserve existing canonical and
fragment parameters, and test all score/derivative/packing/minimization paths.
An input distance or inherited hydrogen frame alone is not independently validated
equilibrium geometry; the source of the generated link parameters must be explicit.
Do not replace this gate with a zero-gradient check or a frozen-link workaround.

### Connection parameter source exploration

`diagnose_link_mmff_coverage.py` sets every coordinate to NaN before deriving
bonded chemistry and queries RDKit's MMFF94 bond/angle parameters. All 15
non-peptide attachment bonds in the source CIFs return parameters: the 14
anchored-group links probed above plus the free NDG–GAL disaccharide in 1ax2.
The pair-local and complete connected-group contexts give identical attachment
bond/angle parameters, endpoint types and hydrogen counts in these fixtures.
The biotin amide, N-glycosidic and O-glycosidic equilibrium lengths are 1.369,
1.436 and 1.418 Å respectively. Results are in `results/link-mmff-coverage.json`.

This identifies a possible topology-based source, not an implemented potential
or independent parameter validation. RDKit's MMFF coefficients have their own
units and anharmonic energy forms; copying them directly to tmol's harmonic
terms would be incorrect. See the [RDKit parameter API](https://rdkit.org/docs/source/rdkit.ForceField.rdForceField.html)
and [version-pinned bond energy implementation](https://github.com/rdkit/rdkit/blob/Release_2026_03_6/Code/ForceField/MMFF/BondStretch.cpp).
Full partner chemistry, hydrogen-name mapping, retained intrablock geometry,
exact/default parameter precedence, serialization and one-owner scoring remain
implementation requirements. In particular, pair-local equivalence in these
fixtures does not establish equivalence for arbitrary conjugations or cycles.

### Bounded shared rule compilation

Both engines now cache one fixed compiled rule set, independent of pH. State
containers are created for the current request; the direct molecule path borrows
private read-only queries. Public rule-loader results own the query molecules
and nested state lists. No input molecules or pH history are retained. Tmol's
previous unbounded pH cache and AtomWorks' per-request SMARTS compilation are
removed. Six new tests pass per project, alongside all 34 AtomWorks identity/
protonation tests (40 total). The 19-fixture AtomWorks integration rerun passes preparation, construction,
finite scoring/gradients and rotamer construction on CPU. Its full stage records
are in `results/atomworks-rule-cache-matrix.json`; this does not revalidate
Cartesian minimization or the unresolved attachment energies.

Seven alternating-order warm measurement pairs compare 21 molecule/pH cases,
five repetitions per sample. All 21 ordered chemical/map inventories and all
rules under 15 pH-range/precision settings match each engine's own baseline.
AtomWorks improves 1.493→0.515 ms per molecule (2.90×). Tmol changes
0.509→0.527 ms (3.5% slower); after 500 distinct pH requests its retained traced
Python allocations fall from 16,104,673 to 64,783 bytes. AtomWorks retains
33,604→43,598 bytes for its new bounded cache. Tracemalloc excludes native RDKit
allocations and is not a total-process memory measurement. Raw timings/source
hashes are in `results/{atomworks,tmol}-dimorphite-cache.json`.

An expanded cross-project contract now covers 33 cases and finds four charge-
state differences: enamine and vinylogous amide at pH 2 and 7.4. All 33 preserve
heavy-atom maps. Tmol's additional `Enamine` SMARTS/pKa rule causes the difference;
its scientific scope/default ownership needs an explicit decision before
consolidating the rule files. The profiler initially detected the different
rule files in the baseline snapshot; the reported per-engine benchmarks use
each engine's own unchanged rules. No rule values were altered in this change.

AtomWorks cache implementation is committed locally as `0e4ffe8f` on
`review/tmol-pr503-shared-chemistry`; no AtomWorks remote was written.

### Parameter injection invalidation prerequisite

Before introducing connection parameter records, an executable cache audit found
that the public `inject_residue_params()` route changed cartbonded data without
changing its hash. Existing block/PBT annotations then used whichever database
was scored first. Updating alanine's CA–CB equilibrium length/stiffness gave
zero change on an already annotated pose, including weighted block-pair scores.
Rebuilding the hash fixes this. Four CPU regressions failed before and pass
after; Slurm 245573 passes 14 CPU/CUDA tests, covering both annotation orders,
whole-pose/block-pair energies and independent coordinate gradients, plus the
existing manual-parameter replacement and connection-improper reference checks.
The source database remains unchanged. This fixes parameter invalidation; it
does not yet supply the missing attachment potential records.

### Explicit connection-owned bonded records

A new `ConnectionCartRes` describes the complete bond and two-block angle set
for a pair of exact patched block types and named connections. Leading `+`
atom names refer to the second block. Compilation validates topology coverage,
duplicate/extraneous paths, finite nonnegative force constants, valid targets,
conflicting records and exchange symmetry when both endpoint types/connections
are identical. Matching records replace the legacy cartbonded length/angle
lookup once per connection. Existing proper/improper ownership remains separate.
Missing records retain the existing fallback behavior; they do not become new
physical parameters merely by installing this backend.

A sparse hash identifies the connection pair in either orientation. Both keys
share one compact path/parameter span; the reversed key switches the local
atom sides without duplicating parameter values. Storage follows the number of
explicit records, not the square of the block-type count. An ordinary five-term
heterotypic record needs 336 additional bytes of packed integer/float data,
excluding allocator overhead and Python metadata. Canonical databases with no
records have empty additional tables. Empty tensor strides and an empty legacy
parameter database are handled explicitly.

One shared native evaluator replaces four duplicated connection-path loops.
It serves pose/rotamer forward/backward kernels. The independent synthetic
harmonic tests cover whole-pose/block-pair energies and arbitrary weighted
coordinate gradients; opposite record orientations; exact single ownership with
legacy rows present; connection-only databases; mixed pose batches; exchange
symmetry; rotamer pair enumeration/gradients; actual packing; and Cartesian
minimization of a bond displaced by 1 Å. The minimizer restores the declared
length within 0.005 Å while keeping the first block unchanged.

Slurm **245896: 70 passes** for the initial backend and complete existing
cartbonded suite on CPU/CUDA. **246259: six passes** for added identity/batch
checks on both devices. **246520: nine passes** for Cartesian minimization,
extended serialization and existing fragment score/DDG/minimize/pack checks.
The fragment integration cases select CPU internally; they are not represented
as CUDA fragment validation. Some tests overlap across runs, so these counts
must not be added as a unique-test total. See `results/explicit-connection-tests.json`.
Cartbonded YAML roundtrips preserve records, including companion generated
files and files containing the derived hash. Public parameter injection retains
existing connection records when other residue rows are updated.

The native comparison compiles `caaa27b6a`'s kernel under a distinct namespace
and passes both versions identical current coordinate/parameter tensors. Seven
alternating-order warm pairs use 1, 16 and 64 ubiquitin poses. CPU scores and
gradients match bit for bit; forward/backward timing is within about 1.1%.
H200 CUDA improves approximately 1% for one pose, 20% for 16 and 25–26% for 64.
Maximum CUDA score/gradient differences are 0.0000458/0.00000382, within the
comparison tolerance. This measures only cartbonded native scoring and its
wrapper, not full application setup or total-score/packing speed. Scripts and
raw timings are `profile_cartbonded_connections.py` and
`results/connection-backend-profile-{cpu,cuda}.json` (GPU 246259).

**Still required before closing comment 38:** generate and persist parameters
for the actual conjugated chemistry. The topology-only MMFF coverage probe is
an input to this work, not a drop-in energy model: its force constants, units
and anharmonic form differ from tmol's harmonic representation. Validate local
bonded geometry changed by conjugation, atom typing/charge state and proper/
improper ownership as well as the new cross-bond rows. In particular, replacing
an amine hydrogen by an acyl group changes more than the attachment distance.
The `.tmol` export/reload path must retain canonical partner patches, charges,
connection records and parameter provenance. Repeated names in distinct chemical
contexts must not silently share incompatible records. Angles spanning two
connections and three blocks need separate ownership beyond this two-block
backend. Automatic generation must not claim such cases are covered by the
passing two-block tests. The original default biotin/glycan stiffness failure
remains open until the real prepared fixtures pass force, packing and Cartesian
minimization checks with generated records.

The fragment integration tests now accept the device fixture instead of forcing
CPU. **246982: seven passes** adds all five fragment score/DDG/minimize/pack
cases on CUDA and verifies on CPU/CUDA that public residue-parameter updates
retain the already-installed connection records and their actual energies.
The earlier 246520 statement describes that historical run; CUDA fragment
coverage is now explicit. All four backend/extension Slurm jobs completed 0:0.

## Conjugate persistence and bounded params export

This goal turn made progress from `e2107de71`; it does not close the automatic
connection-parameter gate or the overall review. Preparation/export/reload of
all three actual conjugate fixtures failed on CPU before the changes: the saved
file omitted the canonical partner patches and charges, and the loader also
discarded patches whose base definitions were supplied by the standard database.

The preparation records now carry shared partner metadata as well as optional
`ConnectionCartRes` records. Canonical conjugation generation returns patches
and charges to this same bundle; direct preparation and export use the same
injection path instead of adding canonical partners separately only in memory.
Public injection applies new patches to existing partners before adding new
base residues. Loading a bundle after its ligand base type has already been
registered still installs shared additions. Repeating an identical bundle is
idempotent; conflicting patch definitions with the same name raise explicitly.
The loader retains each shared record once in the returned preparation list,
and rejects metadata-only bundles that have no residue to carry it. Inject the
whole list to restore the bundle. Base residue definitions already registered
under a name still follow the existing skip policy; this does not solve the
broader conflicting chemical identities under one name.

The writer preserves bond order in the definition and full numeric internal
coordinates. Previously it sorted bonds for appearance and rounded distances
and degree-converted angles. `ConnectionCartRes` fields, including provenance,
now survive `.tmol` serialization. Writers emit **version 2.0** so old readers
reject files whose connection/partner metadata they would otherwise silently
drop; this reader accepts both version 1 and version 2. Old files cannot recover
metadata already omitted by their writers. Exact residue/charge/coordinate/
connection comparisons and per-term score/gradient checks pass for biotin,
N-glycan and O-glycan on CPU/CUDA. Synthetic explicit connection records retain
their actual scoring/gradient effects after reload, both with and without an
already registered ligand base type.

Export also allocated and globally registered a fresh Python marker class for
every atom/charge/parameter row. A single module-level marker class removes this
retention and the extra dictionary copy. In seven alternating-order warm writer
pairs on shared prepared records, median export improves **1.06–1.08×**. Twenty
exports retain **10.19/53.51/51.85 MB** of traced Python allocations for biotin/
O-glycan/N-glycan before, versus **1.1 KB** each after garbage collection. The
YAML registry grows by **4,220/22,360/22,780** classes before and zero after;
traced peaks fall from **10.90/57.60/55.99 MB** to **0.70/4.06/4.16 MB**. These
are YAML export measurements, not native/process memory or total preparation
performance. The benchmark is `profile_params_writer.py`, with raw pairs and
allocation counts in `results/params-writer-profile.json`.

Validation before the format-version guard: **116 CPU passes, nine skips** for
roundtrip, entry-point and nonstandard-backbone tests; **52 CPU passes, ten
skips** for roundtrip, additional bundle validation, pipeline and reference-I/O
checks. Slurm **247637: 64 CPU/CUDA passes, exit 0:0**, including the complete
explicit connection backend and injection-cache regressions. These runs overlap
and must not be added as a unique-test count. Format-version validation adds
**40 CPU entry-point passes**. The first v2 GPU job, **247959**, has **11 passes
and four failures in a new test's YAML quote-style assertion**; all failures
occur after successful parsing/record equality and before native scoring. The
assertion now compares parsed YAML rather than literal quote style. The affected
checks are rerun separately and recorded in `results/conjugate-roundtrip-tests.json`.
The final rerun, Slurm **248153: ten passes, exit 0:0**, covers all four explicit
connection-bundle CPU/CUDA cases and the six loader compatibility/validation
cases. The v2 quote-style test failures are resolved.

Review drafts now contain **42 inline comments and 12 general questions**.
Comments 41/42 cover conjugate persistence and per-record YAML class retention.
No comments were posted to the upstream PR.

**Next correctness gate remains automatic parameter generation for actual
conjugation chemistry**, including local geometry/typing/proper-improper
ownership changed by conjugation. The full default 14-link stiffness failure
is still open. The new backend and persistence path are prerequisites, not
substitutes for generated parameters. Next, probe lysine acylation's local
geometry and torsion ownership with independent energy/force checks before
choosing and documenting a generated bonded model. Three-block angles and
reused residue names in incompatible partner contexts also remain open. Other
previously recorded gates—sampling budgets and late sampler conflicts,
AtomWorks default-reader/rule-profile contracts, cache lifetime, broad workflow
and release validation—remain active. Atom-type element mappings for newly
introduced custom type names are also not yet serialized by `.tmol`; canonical
and existing generic types cover the current fixture roundtrip checks.

## Acylation audit and isolated bonded parameters for exact variants

This goal turn made progress from `b25bb991b`. The default biotin preparation
still uses an amine-like local parameter set after removing lysine hydrogens.
The existing `conjugated_chemistry()` helper reports one hydrogen, site type
`Nad`, site charge −0.7301 and hydrogen charge 0.37, but preparation uses only
the hydrogen count. The patch installs `Nbb` through a static type mapping;
the effective NZ charge is +0.35643 after moving the departing hydrogen charges
onto it. These numbers identify the code paths; this comparison alone is not a
validated replacement partial-charge model.

The surviving CE–NZ–HZ1 angle retains `x0=1.91114 rad, K=51.348`. Three
CE–NZ proper torsions involving HZ1 retain their original LYS coefficients
(`k3=2.704` each). With the heavy atoms held fixed, moving the hydrogen from
109.5° to 120° costs **0.8621957 kcal/mol** in cartbonded angles. Rotating it
180° about CE–NZ changes the old lysine proper energy by **14.7585565 kcal/mol**.
The latter movement changes cross-connection angles as well, so it is not an
equal-energy symmetry comparison. CPU/CUDA agree on these changes to about
3e−14. No cart improper root covers NZ and the installed `Nbb` generic improper
lookup returns None, but proper torsions do respond; do not report a complete
absence of planarity energy.

The topology-only MMFF probe has local angle targets 119.6°, 120.066° and
120.277°. Its out-of-plane coefficient is −0.02, illustrating why copying it
into a positive harmonic improper is not justified. The probe uses an uncapped
residue pair, whose formal charge includes artificial free backbone termini;
it is **not a reference for the conjugate's net charge**. These diagnostics are
in `diagnose_acylated_lysine.py` and `results/acylated-lysine-{cpu,cuda}.json`.
RDKit's [MMFF parameter API](https://rdkit.org/docs/source/rdkit.ForceField.rdForceField.html)
documents the bond, angle and out-of-plane queries; the numerical observations
here come from the executable diagnostic, not from a new parameter fit.

**Implemented:** exact patched names in `CartBondedDatabase.residue_params`
can now own complete `CartRes` replacements. Previously the scorer fetched only
`base_name`, so rows for `LYS:conj_NZ` could not change its local geometry without
also changing ordinary lysine. The scorer uses a private atom-ID namespace only
for types with explicit replacements. Other variants, other score terms and
the shared atom-ID tensor remain intact; when no replacements are active, the
scorer reuses the existing ID tensor. No native kernel or signature changes are
needed. A replacement is a complete record, not a merged delta; use an evolved
copy of the base `CartRes` for a local correction. There is no implicit matching
to other terminal/patch combinations. Existing wildcard and explicit
connection-record precedence remain unchanged.

`LigandPreparation.additional_cartbonded_params` carries these shared records;
the writer and loader retain them even when the canonical partner's base is
not defined in the bundle. They also apply when the source ligand is already
installed, with repeat injection idempotent. A shared collector rejects
contradictory bonded definitions within the same bundle. Metadata-only empty
bundles raise rather than discard the records. The format remains the branch's
unreleased v2 schema, now retaining these existing `residue_params` keys too.

Validation: the old Python scorer, loaded from `b25bb991b` in a separate module
on the same current poses/databases, produces **eight CPU failures**: every
exact-variant override has zero effect. The new scorer passes independent
harmonic energy/gradient checks on jagged AAA/AAAA batches containing unpatched,
N-terminal and C-terminal ALA, both annotation orders, and whole-pose/weighted
block-pair paths. Rotamer tests use jagged KKK/KKKK batches and independent
per-rotamer energies and gradients. A biotin test persists a synthetic local
angle replacement, reloads into both fresh and already prepared databases, and
checks its isolated energy/gradient effect. Slurm **248669: 111 CPU/CUDA passes,
exit 0:0**, includes these tests, the complete explicit connection backend,
injection-cache regressions, existing cartbonded suite and conjugate roundtrips.
Final CPU bundle/entry-point validation has **63 passes, 15 skips** and includes
the added contradictory/empty-record checks. Counts overlap; see
`results/variant-cartbonded-tests.json`. Review drafts now have **43 inline
comments and 12 general questions**; no upstream comments were posted.

**Next:** build automatic connection and local parameter generation from the
complete conjugated chemistry, including an explicit protonation/charge policy,
proper/improper ownership and provenance. The harmonic conversion of native
MMFF coefficients requires documented units and approximations; current
intra-ligand generation instead uses fixed K=300/80 and optimized geometry.
Neither approach has yet been installed for the default conjugate links. The
14-link default stiffness failure, acylation corrections, three-block angles,
incompatible chemistry sharing names, broader sampling budgets, AtomWorks
reader/rule-profile contracts, cache lifetime and full workflow/release gates
remain open. The new variant and persistence support are prerequisites, not a
claim that those default chemical failures are fixed.

## Complete capped conjugate models, with source annotations retained

This goal turn made progress from `b723e88c7`. The automatic parameter generator
now has a private, tested chemical-model builder, `tmol.ligand._conjugate_model`.
It consumes explicit bonds and a prepared chemical database, cuts ordinary
up/down polymer connections, and uses the existing polymer cap profiles to
model the remaining connected covalent groups. It retains every attachment in
a group, including unanchored groups and groups with more than one polymer
anchor. Source atom and residue-instance indices are retained separately from
atom names; synthetic cap atoms have source index −1. Unknown residue
definitions, duplicate atom names within an instance, or caps that remove an
attachment atom raise explicitly. The builder is not yet called by the default
preparation pipeline and does not install energy parameters.

The underlying capping code used to discard source annotations and construct
coordinates even when only chemical topology was needed. It now retains the
annotations of surviving source atoms, including formal charges, chemistry
tags and insertion codes; synthetic cap fields start empty/zero except residue
identity. Cap names are widened as needed. `include_coordinates=False` skips
frame construction and gives all-NaN coordinates. Bond remapping is batched,
default annotation buffers are reused, and new group construction filters
cross-residue bonds in NumPy before Python classification. Group links are
indexed by residue so collecting many independent groups does not rescan all
links for every group. No new global cache or duplicate molecule converter is
introduced.

Model tests cover all **15 source attachment links** in the three fixtures
(biotin 1, N-glycan 8 including its free disaccharide, O-glycan 6), as well as
duplicated inputs. Retained atom annotations and declared bond orders agree
with the original input; protonation retains a complete heavy-atom map, every
group is one connected molecule, and MMFF supplies each connection bond and
its adjacent angles. Distorted coordinates and all-NaN inputs produce identical
models/SMILES. The capped biotin model is neutral, unlike the artificial free
backbone ammonium in the earlier uncapped diagnostic. This is a parameter-model
charge, not a claim that every terminal form of the actual pose is neutral.

A constructed two-anchor case attaches a second lysine to biotin C10 while
retaining the original C11 amide. Both lysine instances survive in the same
model: the acylated nitrogen has one hydrogen and formal charge zero; the
alkylated nitrogen has two hydrogens and formal charge +1 at the tested pH.
Their MMFF atom types differ despite sharing a residue name. This is a useful
regression input for the remaining context-sensitive patch/parameter generator;
the current default preparation does not yet distinguish these contexts.

`profile_capping.py` compares identical residues/profiles with the previous
source implementation loaded separately. Before timing, it verifies identical
coordinates, bonds and common annotations; the old output is missing `charge`
and the added source chemistry tag. Seven alternating-order sets of 100 calls
on HYP/MLE/B3K/FGA give **2.46–2.64×** topology-only speedups. Coordinate-producing
calls cost **7–13% more** (approximately 20–33 µs per call here) to retain their
annotations. Raw pairs are in `results/capping-profile.json`. This does not
measure full preparation, molecule conversion, or end-to-end scoring speed.

Validation: **77 CPU passes, four skips** for capping identity and the existing
nonstandard-backbone suite. Slurm **248818: 104 passes, two skips, exit 0:0**
includes cap/group chemistry checks plus CPU/CUDA backbone and conjugate
roundtrip tests. Final focused CPU checks after buffer reuse and cross-bond
filtering have **16 passes, two skips**. Counts overlap and the graph/capping
tests themselves are CPU chemistry tests even when run inside a GPU job.
Final container/CUDA roundtrips are recorded in `results/capped-model-tests.json`.
Slurm **248821: 32 passes, two skips, exit 0:0** verifies the final capping and
model changes together with CPU/CUDA conjugate roundtrips.
Review drafts now have **44 inline comments and 12 general questions**.

**Next action:** consume these capped models to derive atom identities,
hydrogen counts, local typing/charge updates and harmonic connection records;
map generated hydrogens by their bonded parent and validate counts against
the actual patched types. Keep terminal variants and repeated residue contexts
separate, and carry the resulting records through the already validated bundle
and scoring paths. The original **14 anchored links still lack default
generated length/angle energies**. Acylation corrections, three-block angles,
fragment projection, name-context conflicts, sampling budgets, AtomWorks
reader/rule-profile contracts, cache lifetime and full workflow/release gates
remain open. No whole-force-field or scientific-fit validation is claimed.

## Capped-MMFF attachment generator and repeated-group cost

The private `tmol.ligand._connection_params` prototype now consumes complete
capped models and produces explicit `ConnectionCartRes` records. It uses the
existing molecule converter and direct protonation API, preserving heavy-atom
maps through protonation. Hydrogens map by their bonded parent and count;
attachment neighborhoods and bond orders must agree with the patched residue type. Exact
terminal variants receive separate records. Repeated names/site sets with
incompatible chemical graphs, formal charges, MMFF assignments or hydrogen
counts raise. This checks identities within the input; it does not solve
selection of context-specific residue types across poses or reused databases.

The parameters are the local harmonic curvature of MMFF94: equilibrium bond
lengths in Å, angles in radians, and `K = 143.9325 * kb` or
`K = 143.9325 * ka`. The conversion follows RDKit's pinned
[bond implementation](https://raw.githubusercontent.com/rdkit/rdkit/Release_2026_03_6/Code/ForceField/MMFF/BondStretch.cpp),
[angle implementation](https://raw.githubusercontent.com/rdkit/rdkit/Release_2026_03_6/Code/ForceField/MMFF/AngleBend.cpp)
and [unit constant](https://raw.githubusercontent.com/rdkit/rdkit/Release_2026_03_6/Code/ForceField/MMFF/Params.h).
Independent tests query RDKit's energy and Cartesian gradients with all other
terms disabled, then finite-difference the bond and angle curvature in water
and the special linear angle in HCN. These are unit/implementation checks,
not an independent validation of MMFF's fit for the conjugates. The harmonic
model omits anharmonic and stretch-bend terms; it is not the complete MMFF
energy function.

Each record contains JSON provenance with the method version, RDKit version,
pH, compiled protonation-rule hash, first-ordered-variant selection policy,
precision/variant limit and complete capped molecular SMILES. Actual generated
records roundtrip through a source ligand's `.tmol` bundle into both fresh and
already prepared databases, with repeat injection idempotent. The prototype
does not independently infer connectivity from coordinates or introduce a
second molecule/protonation implementation.

Native whole-pose and weighted block-pair energy/gradient checks cover all
**15 source links and 61 adjacent angles** in biotin, O-glycan and N-glycan,
including the unanchored disaccharide. They isolate the generated terms so
legacy wildcard parameters cannot conceal missing rows. All-NaN coordinates,
reversed atom order within residues, changed instance numbering and duplicated
inputs yield identical records. Changed amide chemistry under a reused residue
name and mismatched attachment hydrogen inventories raise explicitly.

The diagnostic now accepts `--generated-connections`, installs these records
through the public database injection API, and asserts the measured stretching
stiffness against each generated K. On CPU and CUDA, all **14 anchored links**
pass: K is **838.983** for the biotin amide, **671.301** for the ASN attachment,
and **726.427 kcal/mol/Å²** for glycosidic O–C links. Three peptide controls
remain **369.445**. Perpendicular bending has a nonzero force response; its
curvature need not be positive away from equilibrium (the fixed biotin
hydrogen geometry gives −1.134 in this particular scan).

With the protein fixed, 100-iteration full-score biotin minimizations from
initial bond lengths 1.329, 1.829 and 2.329 Å finish at **1.3603, 1.3676 and
1.3794 Å on CPU**, and **1.3601, 1.3670 and 1.3792 Å on CUDA**. All lower the
score, restore the link within 0.1 Å of its generated 1.369 Å target, and leave
every masked-out coordinate unchanged. The earlier default runs instead
finished at 1.660/2.150/2.556 Å on CPU and 1.658/2.120/2.558 Å on CUDA. These are
specific minimization checks, not global-minimum or force-field validation.
See `results/generated-connection-stiffness-{cpu,cuda}.json`.

Profiling showed repeated protonation dominates generation for duplicated
glycans. Models now stream one group at a time, release cap-construction
temporaries before parameterization, and skip already processed chemistry
within a call. At most 128 content digests are retained; the local set is
cleared at its bound. There is no global model/molecule cache. The digest
includes the chemical annotations consumed by conversion, explicit bonds,
atom/residue names and relative source/cap identity. Chain numbering and the
all-NaN model coordinates do not determine chemical parameters.

`profile_connection_generation.py` compares this implementation against the
same generator with reuse disabled and models materialized. Every timed result
must equal the reference record tuple. Seven alternating-order warm pairs give:

| Fixture copies | Biotin reference → candidate | O-glycan reference → candidate | N-glycan reference → candidate |
|---|---|---|---|
| 1 | 4.47 → 4.52 ms | 17.48 → 17.65 ms | 22.86 → 23.11 ms |
| 8 | 31.66 → 9.15 ms (3.46×) | 134.34 → 22.68 ms (5.92×) | 176.52 → 36.05 ms (4.90×) |
| 32 | 125.29 → 25.02 ms (5.01×) | 533.74 → 39.47 ms (13.52×) | 699.23 → 89.59 ms (7.80×) |

Traced Python peaks at 32 copies fall from **0.643→0.595 MB**, **0.812→0.496 MB**
and **2.146→1.769 MB**, respectively. At one copy the candidate costs an extra
9–24 KB of traced Python peak memory and about 1% time; streaming retains
its group-index state while parameterizing. These measurements exclude native
RDKit/RSS allocations. They describe the new generator stage, **not a speedup
over upstream preparation**, which does not generate these parameters, or an
end-to-end preparation benchmark. Raw pairs and source hashes are in
`results/connection-generation-profile.json`.

Validation: Slurm **248831 has 92 passes, six skips, exit 0:0** for the initial
generator plus existing backend/model/bundle checks on CPU/CUDA. The first
follow-up job **248833** passed 49 tests but then failed in the diagnostic:
switching databases required annotating each block type before packed-type
setup. That harness error is fixed. **248835 has 52 passes, six skips, exit
0:0**, followed by the complete CPU/CUDA stiffness and minimization checks.
Final model/generator CPU checks have **30 passes, 12 skips**; the final
provenance and explicit angle-count checks have **20 passes, ten skips**.
The final bond-order guard and generator checks have **23 passes, ten skips**.
Counts overlap; the pure chemistry tests use CPU even inside a GPU job.
See `results/generated-connection-tests.json`. Review drafts remain at
**44 inline comments and 12 general questions**, with comment 38 updated to
distinguish the tested prototype from the still-unfixed default route.

**Next:** derive local geometry, typing/charge and torsion-ownership changes
from this same capped chemistry, then integrate one consistent parameterization
pass into preparation and export. The prototype deliberately remains private
and is not called by default preparation. Ordinary LYS parameters still govern
the local acylated site; hydrogen construction/optimization must agree with
the resulting amide chemistry. Incompatible contexts need explicit selection
or rejection across reused databases, not only within this generator call.
Scoped handling of skipped/unprepared groups, mixed existing connections,
three-block angles and fragment projection remain open. The existing broader
sampling-budget, cache-lifetime, AtomWorks reader/rule-profile and release
validation requirements also remain active. This is a working, tested
parameter-generation mechanism, not a claim that the full PR is ready.

## Electrostatic parameter identity and the local charge model

Deriving local chemistry exposed two prerequisites in the electrostatic scorer.
The existing BT/PBT annotations retain the first database's charges and
count-pair representatives, and the resolver rejects records with more than
one patch suffix. Those guards predate PR 503; parameter injection and the new
generic/Rosetta classification make their scope consequential here. Four
native CPU reuse checks plus one combined-patch lookup fail before this fix.

The scorer now retains one most-recent annotation on each BT/PBT, keyed by a
weak reference to the immutable electrostatic database and, for the packed
mask, the Rosetta typing set. Rendering captures that term's parameter
tensors, so an already rendered module survives another database's setup.
When only charges change, it reuses the quadratic representative-distance
tables. Forty successive charge databases leave no database owner alive and
only the latest charge tensor on the packed owner after collection. There is
no growing dictionary of database configurations. Global tensors are packed
once per term, and the host cutoff comes directly from the database rather
than converting a CUDA scalar during every render.

Charge and count-pair rows can name an exact complete patched residue. Lookup
tries that name, then individual patches in the existing name order, then the
base. A combined record does not apply to a residue with another added patch.
When multiple rows nominate a representative for the same atom, specificity
is resolved across those rows; within equal specificity, the existing
last-outer-atom rule remains. All **230 default refined residue types** have
bitwise-identical charges and representative mappings to `20742270f`.
Missing applicable charges on a known residue now raise rather than silently
becoming NaN; an entirely unknown residue still has the existing zero fallback.

Independent dielectric-formula checks cover changed charges, representatives,
generic masks and globals on jagged AA/AAA poses, both annotation orders,
whole-pose and weighted block-pair scores, and coordinate gradients. Jagged
KK/KKK rotamer checks exercise both charge orders, all terminal rotamer pairs
and weighted gradients. A real biotin/LYS bundle carries exact charges for
`nterm + cterm + conj_NZ`, reloads into both fresh and previously prepared
databases, and matches independent native energies/gradients. These injected
test charges establish parameter routing, not a scientific charge fit.

Final CPU electrostatic checks have **33 passes, 33 skips**; the added rotamer
checks have **two passes, two skips**. Slurm **248854 has 115 passes and four
skips** across CPU/CUDA electrostatics, conjugate bundles and generated
connection tests. The earlier job 248849 had 107 passes and four failures
from two test-construction mistakes (assuming identical atom indices between
variants, and using a nonexistent Biotite singular-mask API); both were fixed
before the final run. Counts overlap; the final checks are not an independent
full-suite or release result. See `results/elec-identity-tests.json`.

Slurm 248854 finishes **COMPLETED, exit 0:0, elapsed 2:02**, including the
CUDA profile after pytest. Seven alternating-order warm measurement pairs
from `profile_elec_setup.py` give:

| Setup stage | CPU reference → candidate | CUDA reference → candidate |
|---|---|---|
| Collect scoring arguments | 5.70 → 0.89 µs | 67.16 → 0.95 µs |
| Construct electrostatic term | 2.271 → 2.228 ms | 3.077 → 2.998 ms |
| Two charge-database switches | 14.02 → 6.09 ms | 31.14 → 12.47 ms |

The first two references load the unchanged classes from `20742270f` in the
same process. All default residue lookups and emitted scoring arguments are
checked for exact equality before timing. The switch reference uses candidate
code with a forced, correct rebuild, because timing the stale original cache
would compare against incorrect behavior. These are setup-stage measurements
on a 20-residue sequence, **not native scoring or end-to-end speedups**.
Reported Python allocation peaks exclude tensor/native allocations and are
not process-memory measurements. Raw pairs, environment and source hashes are
in `results/elec-setup-profile-{cpu,cuda}.json`.

The local chemistry audit now compares complete capped conjugates with the
same heavy-atom models whose attachment bonds have been cut and valences
completed. These disconnected reference fragments are **not physical
reactants**: for example, the biotin carbonyl carbon acquires hydrogen rather
than an acid oxygen. MMFF partial charges sum to each model's formal charge;
the artificial caps have zero charge delta in every fixture.

For biotin, the complete group's formal charge changes **+1 → 0**. Heavy-atom
plus attached-hydrogen deltas are **−0.2029 on LYS CE**, **−0.8571 on LYS NZ**
and **+0.0600 on BTN C11**. NZ changes generic type `Nam → Nad`; BTN O11
changes `Oal → Oad`, despite having unchanged MMFF type and partial charge.
O-linked oxygen changes `Ohx → Oet` and loses its hydrogen, with a **−0.28**
local charge delta balanced by **+0.28** on the attachment carbon. ASN ND2
remains `Nad` but changes hydrogen count, with **−0.3001** balanced by
**+0.3001** on its NAG carbon. Thus hydrogen count, generic type, MMFF type
and charge are distinct outputs; none alone determines the others.

One candidate policy is to preserve curated baseline charges and add the
connected-versus-disconnected MMFF deltas, explicitly accounting for removed
hydrogens on their parent. That would preserve remote backbone parameters
while representing the formal-charge change. It is a model choice requiring
validation, not an established physical reference. Replacing every capped
group charge with MMFF would also change otherwise canonical regions. No
automatic charge policy is installed by this change. Typing, surviving local
bond/angle terms, generic/cartbonded torsion ownership and hydrogen
construction must be resolved together before enabling default generation.
See `diagnose_conjugation_charge_changes.py` and
`results/conjugation-charge-changes.json`. Review drafts now contain
**45 inline comments and 13 general questions**. All earlier completion gates
remain active.

## Unresolved attachment measurements and insertion-coded residues

`_bond_lengths_by_site()` was copying every measured norm directly into the
connection patch. Setting only the attached biotin LYS NZ coordinate to NaN
therefore gave every LYS `conj_NZ` terminal combination and `BTN:conj_C11` a
NaN connection distance. The resulting example pose still happened to have
finite coordinates; the reproduced error is the invalid stored/exported
construction parameter, not a claim that every score immediately becomes NaN.
A later unresolved repeat also replaced an earlier valid observation, and
residues distinguished only by insertion code were treated as one residue.
Seven focused CPU checks fail against `daef9b803` before this fix.

The helper now identifies cross-residue pairs using Biotite's complete
contiguous-residue boundaries, filters nonfinite endpoints and nonpositive
lengths, and processes bond/coordinate arrays in chunks of at most 4096
bonds. An unresolved repeat cannot override a valid earlier measurement.
Input coordinates remain untouched. Preparation, export, fresh reload and
re-injection preserve finite icoors in all three biotin cases: missing NZ,
missing C11, and both missing. The polymer NZ can be rebuilt; a missing ligand
heavy atom still raises the existing explicit pose-construction error. The
same three cases pass with AtomWorks' parsed AtomArray in the Biotite 1.6 CPU
environment; tmol does not read or complete the file a second time.

This retains the existing no-measurement behavior: the patch inherits its
departing hydrogen's construction frame. It is **not** a chemistry-derived
heavy-atom equilibrium geometry or a new attachment energy model. Multiple
distinct finite observations still select the last instance. Default
chemistry-derived geometry/context selection and reconstruction of unresolved
ligand heavy atoms remain open; these changes do not silently place such atoms.

The paired warm measurement benchmark compares unchanged `daef9b803` with
the new helper on the same source arrays, with 1, 8 and 32 repeated copies.
Seven alternating-order pairs show **2.59–4.44×** faster collection. At 32
copies, biotin takes **28.21→8.62 ms**, O-glycan **9.54→2.15 ms**, and
N-glycan **92.04→26.76 ms**. Key inventories match; batched float32 norms
differ from individual norms by at most **1.20e−7 Å**. This measures the
measurement helper, not full preparation or scoring.

There is an allocation tradeoff: traced Python/NumPy peaks at 32 copies rise
from roughly **7/6/10 KB** to **242/192/288 KB**, respectively. Chunking bounds
the bond/coordinate temporaries; the residue-boundary scan still scales with
the input. The original streaming Python loop has the lower traced peak.
The first fully vectorized version used 332 KB for 32 biotin copies; bounded
chunks reduce that to 242 KB with a small time cost. These figures exclude
native/process allocations and must not be presented as total-memory savings.
See `profile_attachment_measurements.py` and
`results/attachment-measurements-profile.json`.

Final CPU input/bundle checks have **18 passes and eight skips**. Final
CPU/CUDA input/bundle checks have **26 passes** in Slurm **248870**,
**COMPLETED, exit 0:0, elapsed 1:13** (pytest 59.78 seconds). Additional
CPU/CUDA model, connection-generator and complete conjugate-packing checks
have **85 passes and six skips** in Slurm **248864**, **COMPLETED, exit 0:0**
(pytest 415.49 seconds). That broader run started before the bounded-chunk
revision; 248870 and the final CPU run validate the final helper. Detailed
run records, including AtomWorks, are in `results/attachment-measurements-tests.json`.
Review drafts now contain **46 inline comments and 13 general questions**.
The local charge/typing/torsion model, default generator integration, broader
budget and cache-lifetime gates, and shared AtomWorks reader/rule-profile
contracts remain active.

## Upstream update detected during this follow-up

On 2026-09-11, PR 503 advanced from `c03c1e745` to
`0f4c3bc426bca78e8681f0b730fa23c3e26ef261` (author timestamp 17:39:28 UTC).
It is fetched as `origin/pr-503-current`. The one new commit changes six files:
canonical fragment selection, fold-forest error ordering, scan-order test
arguments, two fold-forest expectations, HYP's expected count and NCAA score
goldens. The submitted tree still lacks the cyclic-search module.

The existing review/comment anchors remain pinned to `c03c1e745`; this update
must be reconciled before calling the review current. The fragment fallback
overlaps our explicit fragment selection and exact cut-bond removal. Its new
`joins_one_component()` filter compares base names, not original component
instances, so it needs a repeated-fragment crosslink check. Its fallback also
adds only the first conjugated type when a class has no unconjugated candidates;
later alternatives see a nonempty class and are skipped. These are review
hypotheses pending executable checks, not yet reproduced findings.

**Next priority:** independently exercise the six-file delta and reconcile it
with the improvement branch, preserving the cap-selection and exact fragment
identity fixes. Review the upstream golden changes against fixed-parameter
evidence; do not accept a changed expected score as scientific validation.
Then resume the local chemical parameter/default integration and all broader
completion gates above. No automatic goal completion is implied by this
intermediate checkpoint.

## Reconciliation of upstream `0f4c3bc42`

The new six-file delta has now been inspected and reconciled. Its fold-forest
error ordering correctly reports malformed indices/cycles before secondary
unrooted starts. The scan-order tests now pass the five required connection
tensors. Its two fold-forest expectation changes and HYP count of 18 agree
with our earlier corrections; the branch retains its explicit conformer seed.
The upstream golden YAML is included as upstream history, without regenerating
values on this branch or treating changed expectations as validation.

The new fragment selection code has two reproduced problems. Loading its
complete selection module into the branch's prepared-fixture environment
gives **three CPU failures**: explicit bonds between repeated fragment types
return no connections, and either order of two alternative fragment states
retains only the first candidate. `joins_one_component()` compares component
base names without source-instance identity. Its candidate fallback mutates
the same lists it uses to decide whether a class already has candidates.
`check_upstream_selection.py` reproduces these failures and returns pytest's
failure status. It is an isolated-module comparison, not a pristine upstream
import. The separate unmodified `tmol-pr503-updated-baseline` worktree remains
at `0f4c3bc42`; file inventory confirms the cyclic-search module is still absent.

Reconciliation keeps our existing explicit `is_ligand_fragment` predicate and
removes exact cut bonds during fragment expansion, where original component
instances are known. It accepts the new behavior for already-declared
connection sites without dropping the caller's bonds. Public canonical pose
construction now checks three explicit links across a jagged two-pose batch,
including crossed instances of repeated fragment types, and verifies the
complete bidirectional connection tensors. Both alternative fragment states
remain candidates in either order. The bond list is copied to the host once.

The initial CPU selection/fold/scan run has **23 passes and 14 skips**; public
fragment construction has **three passes and three skips**. The initial broader
Slurm **248877** run has **133 passes and two failures**, exit **1:0**, pytest
258.41 seconds. The two failures are the same ACE whole-versus-fragment gradient
check on CPU and CUDA. They exposed a regression in our earlier exact-variant
CartRes implementation, described below; the new upstream selection checks,
cap tests, HYP count and scan/fold tests pass.

### Fragment namespaces and cross-cut bonded terms

The gradient discrepancy is not merely float32 accumulation. Per-term probes
with identical ligand coordinates give a largest double-precision cart-angle
gradient difference of **0.0031391** and cart-length difference **0.0003216**.
Other double-precision ligand term gradients agree to roughly `2e-15`.
The full weighted-gradient check first notices a difference around 0.00165,
although the equilibrium structure's total energy comparisons pass.

Our earlier exact-name CartRes lookup assigned separate namespaces to
`LG1.1` and `LG1.2`, even though both preparations carry the source `LG1`'s
complete, identical records. A cut path then combined IDs from two namespaces
and failed to match its original source row. The fix reuses the base namespace
for a fragment whose complete record equals the source record. Equality also
covers independently deserialized copies; deliberately different exact records
retain the separate-namespace behavior. This removes redundant atom-ID overrides
for ordinary fragments while restoring the original bond and angle parameters.

Two new CPU regressions displace one whole fragment by `(0.23, −0.11, 0.17)` Å,
with identical coordinates in the unsplit ligand. Before the fix both fail;
the missing angle energy alone differs by **3.99097 kcal/mol**. After the fix,
whole/split energies and aligned coordinate gradients agree within `1e-7`,
for both shared and independently copied source records. Existing exact
terminal-variant and biotin bundle overrides remain tested. The combined CPU
checks have **15 passes and 13 skips**. No tolerance was widened.

On the complete original ACE fixture, the fixed largest per-term ligand
gradient difference is **9.54e−7 in float32** and **1.78e−15 in float64** on CPU.
See `diagnose_fragment_gradients.py` and
`results/fragment-gradient-{cpu,cuda,fixed-cpu,fixed-cuda}.json`.
This fixes the common source-record alias case; intentional fragment-specific
parameter changes still need explicit ownership/projection for cross-cut terms.
That broader fragment/custom-chemistry gate remains open.

### Updated score references and fixed-input evidence

Slurm **248878** exercises the updated upstream YAML: **eight failures and
two passes**, exit **1:0**, pytest 79.07 seconds. All four regenerated chemistry
classes differ from those references on container CPU and CUDA. Standalone CPU
has **two failures, three passes and five skips** (40.00 seconds): DNA and beta
peptide differ, while HYP and TTD match. These remain recorded failures.

The updated beta-peptide LJ reference falls from about 663 to 122, close to
the previously observed 121–122 range, but its full generated parameter inputs
and environment are still not supplied by that commit. The DNA reference
retains the earlier heavy-atom OptH displacement: its cart-length score is
about 253.6 versus approximately 10.9–11.3 after our independently tested
heavy-atom-preservation fix. Refreshing that number alone would hide the cause.

Replaying the committed `.tmol` records and exact saved coordinates reproduces
all **192 standalone CPU term scores bit-for-bit**. All **192 CUDA comparisons**
remain within the existing test tolerance; the largest difference is
**0.00263548** in HYP omega. The replay asserts complete atom identity before
installing coordinates. Slurm **248886** completes the initial gradient
diagnostic and CUDA replay, exit **0:0**, elapsed **0:58**. These are numerical
checks for fixed inputs, not proof that regenerated chemical parameters agree
across environments or that the force field has been independently fitted.
See `results/upstream-reconcile-replay-comparison.json` and the unchanged
`fixtures/noncanonical-score-replay/` input bundle.

The updated inventory has **214 files**, recorded separately in
`upstream-files-0f4c3bc42.tsv`; the original 213-file inventory and all original
line anchors remain intact. Review drafts now contain **48 inline comments and
13 general questions**, with the two new comments anchored to `0f4c3bc42`.
Default local chemistry integration, complete sampling-budget semantics,
cache-lifetime/scaling, shared AtomWorks rule/reader contracts and release
validation remain active requirements.

Final Slurm **248891** completes **161 CPU/CUDA tests with no failures or
skips**, followed by CUDA fixed-input replay and the corrected gradient
diagnostic: **COMPLETED, exit 0:0, elapsed 5:35** (pytest 265.65 seconds).
The fixed CUDA per-term ligand-gradient differences are at most **9.54e−7**
in float32 and **3.56e−15** in float64. The original whole-versus-fragment
gradient assertions pass unchanged. Detailed before/after runs and final
source hashes are in `results/upstream-reconciliation-tests.json`.

PR head was rechecked and remains `0f4c3bc42`. This reconciliation includes
that commit as an ancestor rather than maintaining a detached copy of its
changes. The next implementation priority is again the consistent capped
local chemistry model and default integration, with the unresolved
fragment-specific cross-cut ownership issue carried alongside it. The broader
goal remains active; passing this targeted reconciliation suite is not a
release-completion claim.

### Generic lookup references and attachment impropers

The capped chemistry audit requires generic lookup types on canonical neighbors:
LYS CE can remain physically CH2 while matching as CS2; its retained H can remain
Hpol while matching as HN. Simply making CE physically generic would also make
LYS chi4 generic-owned, adding a torsion alongside its Dunbrack term. `Atom` now
has an optional `genbonded_type` reference, validated once per block setup as a
known concrete type of the same element. Physical types still determine the
generic term's proper/improper ownership checks. This does not remove any named
CartRes torsions; local chemical reconstruction must also replace the affected
records, charges and internal coordinates.

The existing native improper helper only admitted fragments of one source
ligand. With corrected amide lookup types, it still gave exactly zero attachment
energy when the retained LYS–biotin H moved out of plane: two controlled CPU
failures, whole-pose and block-pair. The shared helper now admits other chemical
connections and excludes Rosetta-owned centers before neighbor enumeration.
It uses the existing physical-type mask and adds no scoring tensor. Three-block
impropers (a center with multiple remote neighbors) remain outside this
connection-pair enumeration and require an explicit future representation.

Tests independently compute the existing 80*theta^2 amide potential from plane
normals, compare full coordinate gradients, check finite differences, and verify
that correlated group rotamers use the same term. Canonical center and missing
lookup controls stay at zero. The LYS chi4 ownership test guards against a
nonzero generic table match being scored in addition to the canonical term.
Export tests cover references in residue and patch atoms; `.tmol` emits v3 only
when needed, otherwise v2, and accepts v1–v3. Rosetta `.params` export rejects
references before writing because it cannot preserve them. Repeated invalid
setup must raise without leaving partially published generic annotations.

Validation so far: 19 focused CPU passes / 10 CUDA skips; Slurm 248902 has 155
CPU/CUDA passes and no skips, including the existing generic, fragment and
bundle/entry-path suites. The final Python export preflight has six additional
CPU passes (overlapping cases). Earlier Slurm 248900 failed the initially
unisolated tests and is superseded; the isolated fragment-gate reproduction is
in `generic-references-fragment-gate.log` (2 failed, 13 passed, 8 skipped).
Slurm 248902 completed 0:0 in 5:54. Fixed-input CUDA replay preserves all 192
terms within the existing tolerances; the maximum difference from the preceding
CUDA replay is 2.29e-5 (generic term maximum 1.19e-7). It does not validate a new
parameter fit or remove the existing environment-dependent golden mismatches.

[Paired profiles](profile_generic_impropers.py) compile the frozen baseline kernel
in a distinct namespace and alternate both kernels on exactly the same pose and
parameter tensors, after warmup and with CUDA synchronization. Baseline caaa27b6a
has identical generic source to the preceding 1d1bb57a2 head. At 1/16/64 ubiquitin
poses, CPU generic-term forward cost rises 1.9–2.5% and forward+backward rises
1.5–3.1%; CUDA forward is 2.3–3.2% lower and forward+backward 0.4–1.5% lower.
Scores and gradients are identical. These small stage-specific differences are
not an end-to-end speedup claim. Avoiding generic work entirely for interactions
with no owned terms remains a useful optimization target.

The field adds 8 shallow Python bytes per Atom object (64→72), or 8,720 bytes
for the 1,090 distinct Atom objects reachable from the default patched residues.
This excludes allocator/native memory, strings and generated chemistry; there
is no added GPU scoring tensor. Exact source/log hashes and raw timings are in
`results/generic-reference-validation.json` and `results/generic-impropers-profile-{cpu,cuda}.json`.

The default preparer does not yet generate/install these references or local
chemistry corrections. This is a validated scoring prerequisite, not completion
of automatic conjugate parameters or independent scientific validation of the
existing generic amide table. Review comment 49 records the native restriction.

### Coupled local conjugate parameters and explicit sampling correlation

The private `_local_conjugate_params.py` generator now couples the preceding
connection harmonic model with local physical/lookup types, charge corrections,
changed local bond/angle parameters, removal of transferred canonical torsions,
and hydrogen/connection construction. It parameterizes the connected capped
model and its valence-completed disconnected reference once per unique group.
It shares the connected-model conversion and mapping with the connection
generator. Repeated groups, atom-order reversal, renumbering and all-NaN inputs
produce identical records, with two MMFF preparations per unique capped model.

The implemented charge policy preserves the curated baseline and adds the MMFF
connected-minus-disconnected change, subtracting charges of removed reference H
atoms already folded onto their parents by the conjugation patch. This is an
explicit provisional model, not an independent charge fit. The whole biotin
group changes by −1; O-/N-glycan groups conserve their total. Biotin LYS CE, NZ
and retained H change by −0.2029, −0.7771 and −0.08; BTN contributes +0.06.
LYS NZ becomes Nad, CE retains physical CH2 with a CS2 lookup reference, and its
retained H keeps physical Hpol with an HN lookup. Remote canonical backbone
types, charges and icoors remain unchanged in the detailed biotin regression.
The retained amide H uses the actual partner as its plane reference, with a
heavy-atom virtual-connection frame that avoids an icoor dependency cycle.

The private installer checks hashes of each input residue's chemistry, effective
charges and local bonded parameters before modifying a copied database. Reusing
the identical result is a no-op; changed local baselines raise. Connection
provenance records the charge policy and baseline hashes, alongside the existing
RDKit version, capped SMILES and protonation-rule provenance. Generation from a
database carrying those correction records is rejected, preventing accidental
reapplication of the delta. The `.tmol` test roundtrips all corrected records and
reconstructs the private result before installation. The ordinary ligand-bundle
loader still skips existing base definitions and has **not** been upgraded into
a general corrected-variant replacement API. Default preparation remains unchanged.

Execution exposed three additional restrictions. Missing-leaf construction
recognized only `up` and `down`; it now accepts any declared connection name.
OptH used the last chi as an NHQ flip even when conjugation appended a linkage
chi; it now uses the amide/ring axis and disables a flip that moves a connection
atom. Eligibility and sidechain-root lookup share that annotation, allowing
fallback when no OptH sample is available. Finally, chemical connectivity alone
incorrectly correlated independent proton samples. Joint producers now declare
considered-block groups; merging rejects added or overlapping independent states
before coordinate allocation. Both native pair masks and energy collapse use
that validated correspondence. Direct declarations reject invalid/duplicate
members and overlapping single-state groups as well.

Four controlled CPU regressions fail with the pre-fix predicates restored:
named connection lookup, independent 3/3 and 3/2 counts, and attached ASN's flip.
See [check_upstream_sampling.py](check_upstream_sampling.py). This replays selected
Python functions from `77f5d9419` on current fixtures, not an unmodified PR tree.
The initial coupled CPU run had five failures; the next attempt exposed a native
backbone-scoring bounds exit because disabled NHQ eligibility still suppressed
fallback. Slurm 248910 failed in that intermediate state. These logs are retained.

Slurm 248911 completed 0:0 in 7:56 with **141 passes / 8 fixture-specific skips**,
covering local and generated connection parameters, group/OptH regressions,
actual conjugated packing, batched kinematics and generic reference ownership on
CPU/CUDA. After adding installer/provenance checks, Slurm 248965 completed 0:0 in
1:44: **60 passes / 4 fixture-specific skips**, including the full missing-leaf
construction/gradient suite. The local final CPU-only run has 17 passes / 10
skips. These suites overlap; their counts must not be added as unique tests.
After tightening direct correlation-index validation, Slurm 248997 completed
0:0 in 1:07 with **53 CPU/CUDA passes and no skips**, including batched group
kinematics. Source and log hashes plus case inventories are recorded in
`results/local-conjugate-validation.json`.

[The paired mask profiler](profile_sampling_correlation.py) verifies identical
joint-sampler masks on all three real fixtures. Seven alternating-order warm
sets measure old lookup costs of 51–80 µs on CPU and 141–178 µs with CUDA. The
cached lookup is about 0.05–0.10 µs, with no tensor allocation or device transfer.
For ten simultaneous consumers, mask storage falls from 3,240→324 bytes (biotin),
800→80 (O-glycan) and 9,920→992 (N-glycan). These are lookup/storage measurements;
one-time construction/validation and whole rendering/scoring are outside the
timed region. The masks and rotamer inventories match exactly. The first profiler
attempt correctly rejected a 1,024-state budget smaller than one residue's 1,053
required library/extra-chi states; the successful comparison uses 4,096/2,048.

Remaining gates include independent validation of the combined physical model,
context-specific terminal corrections, H-adding reactions, three-block terms,
fragment projection, normal bundle/default-preparation integration and the other
completion requirements above. A passing finite-score/gradient check is not a
scientific validation of the fitted parameters. The goal remains active.

### Generic parameter identity and shared setup tables

The remaining generic scorer `hasattr` guards reused the first database's
annotations. Four CPU regressions now reproduce this on a reused, jagged KK/KKK
pose, in both setup orders and both whole-pose/block-pair modes. The test gives
canonical atoms synthetic generic ownership and doubles every potential strength;
the resulting energies and full weighted gradients must double. This tests cache
identity independently of a new chemistry fit. The analogous rotamer test uses
multiple actual Dunbrack conformers on both residues.

Block and packed-block annotations now record a weak database identity and their
chemical element mapping; packed annotations also record the device. Only the
latest annotation is retained on each owner. Packed setup consumes returned
block snapshots, and rendering obtains the requesting term's packed snapshot,
so later setup cannot change an existing module's parameters. Changing ownership
rebuilds block terms; changing the element mapping reruns generic-reference
validation rather than bypassing it. As with the other identity caches, source
databases are treated as immutable: changes publish a new database object.

Inter-block torsion/improper hash tables depend on the generic database and
device, not the packed set. They are now shared through the existing 32-entry
weak-owner LRU. Values contain no database references. Tests with a two-entry
cache verify shared tensors across packed sets, bounded live entries, removal
when all three test databases die, and continued scoring by modules holding the
old tensors. The duplicate, unused type-name inventory in term initialization
was removed. No native potential, parameter value or ownership rule changed.

Slurm 249000 completed 0:0 in 2:17: **119 CPU/CUDA passes / 4 fixture-specific
skips**, including the complete generic suite, coupled conjugate parameters and
ligand entry paths. The focused CPU identity suite passes seven cases with six
CUDA skips. Earlier CPU validation has 23 passes / 14 skips across identity and
generic-reference tests; these counts overlap.

[Paired setup profiles](profile_generic_setup.py), with identical shared block
types/database and exact comparisons of every annotation tensor, separate
generic setup from common parent annotations. For 230 default block types,
generic-only setup falls **2.977→1.282 ms CPU** and **3.353→1.486 ms CUDA**
(2.32× and 2.26×). Including fresh common parent annotations gives the smaller
improvements **22.092→20.599 ms CPU** and **74.672→73.391 ms CUDA** (6.8% and
1.7% lower latency). Seven alternating-order sets contain ten new packed sets
per method; CUDA is synchronized. Existing block annotations and the shared
database tables are warm; pose creation and scorer rendering are excluded.

Across ten simultaneous packed sets, retained generic tensor storage falls
**2,420,720→2,097,152 bytes** on either device. This excludes allocator/native
process overhead, other score terms and float64 copies made when rendering;
it is not an end-to-end GPU-memory claim. Slurm 249001 completed the CUDA profile
0:0 in 28 seconds. Raw timings, exact source hashes and case inventories are in
`results/generic-setup-{cpu,cuda}.json` and `results/generic-identity-validation.json`.
Cartbonded's per-database annotation dictionaries and shared parent atom-type
cache identity remain further audit targets. The full goal remains active.

### Bounded Cartbonded snapshots and common setup synchronization

Cartbonded retained an unbounded dictionary per block and packed set, keyed by
the bonded content hash. It also stored a separate ownership mask behind an
attribute-only guard, so another `rosetta_typed` configuration reused the first
mask even when its bonded hash was unchanged. Two CPU ownership-order tests and
one five-database cache-bound test fail on the preceding branch.

The cache now retains the two most recently used bonded parameter sets. This
preserves common two-configuration reuse while bounding retained historical
arrays. Packed setup consumes returned block annotations; rendering obtains the
requested packed annotation even after eviction. Previously rendered modules
retain their own tensors. Ownership and its immutable setting live in the packed
snapshot, with identical masks shared across fits. Cache-device comparisons use
the resolver's actual tensor device, so an unindexed `cuda` argument does not
force warm annotations to rebuild. Dead commented-out cache guards and an unused
render conversion helper were removed.

The [cache profiler](profile_cart_cache.py) checks all old annotation fields for
exact equality at each of six different ALA fits on 230 default block types.
Cache-reachable packed tensor bytes fall from **9,379,104 to 3,170,528**, and
per-block NumPy array bytes from **5,974,848 to 1,991,616**, on CPU and CUDA.
This is not total process or GPU allocator memory: it excludes Python parameter
dictionaries, object overhead, rendered modules and profiler-held snapshots.
Per-fit times in the artifact are single-sample diagnostics, not a benchmark
claim. Re-rendering an evicted third-or-older configuration incurs setup again;
the tests verify that this remains correct. The cache is bounded independently
of how long callers retain their own scoring modules.

The shared `AtomTypeDependentTerm` setup also read one CUDA scalar per block to
form a slice bound and one per atom to decide whether it was hydrogen. It now
uses static residue lengths and the current resolver's host indices/flags.
This removes a redundant type-name lookup and **4,968 scalar reads** for the
default 230 types / 4,738 atoms. The new fresh-packed-set regression starts from
blocks annotated with a different atom-type order and checks all current indices,
heavy-atom counts, selected positions and padding independently by element.
It does not claim that the older cache guards safely support every reuse of an
already annotated block/packed set; that shared identity issue remains open.

[Seven paired warm timing sets](profile_atom_type_setup.py) of ten fresh packed
annotations give **19.863→7.646 ms CPU (2.60×)** and **72.613→10.255 ms CUDA
(7.08×)**. All annotation tensors and identifier maps match exactly. Native
scalar reads are counted in a separate instrumented run; CUDA timing is
synchronized. Term construction, pose creation, scoring and allocator peaks are
outside the measurement. These gains must not be multiplied by the earlier
generic-only setup ratios to claim an end-to-end speedup.

Slurm 249005 completed 0:0 in 2:30 with **201 CPU/CUDA passes / 4 fixture-specific
skips** across Cartbonded, generic scoring, common atom annotations and coupled
local conjugate parameters. Slurm 249009 adds the fresh-packed-set resolver
regression and profiles; 249012 then checks the final shared mask and unindexed
device behavior: **26 passes / one CPU-only skip**, completed 0:0 in 43 seconds,
including the final CUDA cache measurement. Suites overlap. The focused initial
CPU run has seven passes / seven CUDA skips. Source/log hashes and case inventories
are in `results/cart-cache-validation.json`; paired profiles are in
`results/atom-type-setup-{cpu,cuda}.json` and `results/cart-cache-{cpu,cuda}.json`.

Review comments 54–56 record the ownership, lifetime and setup-cost findings.
Shared parent atom-type cache identity, complete default chemistry integration,
the scientific checks and the other full completion requirements remain open.

### Second upstream update: cap completion and junction roles (`0593a93b0`)

Frank's next head adds six files' worth of cap completion/tests and noncanonical
junction substitutions. The full upstream inventory is now 215 changed files,
recorded in `upstream-files-0593a93b0.tsv`. The cyclic-search implementation is
still absent from that upstream tree. Comments 57–59 and general question 14
review this delta separately from the prior pinned heads.

The improvement branch already inferred cap termini from their connections and
built a one-heavy-atom cap using the native connection ancestor. Replaying the
new upstream tests before merging gives four passes and one failure on CPU:
NH2 is planar, but its two equivalent hydrogen names are opposite the new
convention. Aligning the first hydrogen trans to the partner reference preserves
the remaining generated relative dihedrals. The native construction path now
satisfies that convention without the additional Python coordinate pass or four
packed fields. Omitted field storage is **24 bytes per padded atom**; this is a
structural storage comparison, not a measured end-to-end latency or RSS gain.

The new junction helper had no chemical-role checks. Its first-oxygen selection
can choose a leaving hydroxyl instead of the carbonyl; its lower-side mapping
turns real 5CM, 8OG and PSU phosphate frames into `N=P, CA=O5'` peptide roles.
Excluding a side solely because its endpoint is called C/N also misses changed
neighbor names. Six behavioral checks fail against that helper, plus one check
for the added retained-atom argument. Ordinary nucleotide links need not match
the extraneous peptide rows; their standard phosphodiester energy is not claimed
to change because of this error.

The final helper consumes the reconstructed residue's mainchain, connections,
atoms and bond orders directly. Separate input profile, connection-atom and
retained-name arguments are unnecessary. Carbonyl C and amine N frames are
validated chemically; identity mappings add no rows. This also fixes a further
heavy-atom-only input case: renamed FGA nitrogen NX acquires generated hydrogen
HN1, which the input graph cannot supply for its connection angle. The new test
requires the resulting `HN1–NX–+C` row. Upper/lower frame tests retain the
original database parameter records, and an independent Cartesian calculation
checks the real FGA-to-peptide bond and all four angles, including weighted
coordinate gradients in double precision, on CPU and CUDA.

A single local frame still cannot resolve arbitrary renamed atoms on both sides
of a connection: the copied remote names remain canonical. General connection
parameter coverage and scientific justification for borrowing peptide constants
remain requirements, alongside the broader completion table above. No reference
scores were updated to conceal changes in parameter coverage.

Validation: Slurm **249016** completed **0:0** in **7:05**, with **285 passes /
4 skips** across I/O, missing-atom gradients, selection, cyclic inference, cap
packing, nonstandard backbones, nucleotides and local conjugate parameters.
This run precedes the final prepared-graph simplification. Slurm **249017** adds
two CPU/CUDA independent junction-force passes. The final graph implementation
passes **16 CPU cases / one CUDA skip**, then Slurm **249019** passes **168
CPU/CUDA cases / four fixture skips**, followed by **14 cap checks**. It also
completes all **19 AtomWorks reader/preparation/construction/scoring/gradient/
rotamer fixtures on CUDA**; a separate final CPU run passes the same 19 fixtures.
Job 249019 completed **0:0** in **5:53**. These are overlapping suites, not a sum
of unique cases. An intermediate test-adapter API mismatch (two failures) is
retained as diagnostic evidence and superseded by the final passing reruns.

Source/log hashes, case names, package versions, role probes and Slurm accounting
are in [results/upstream-059-validation.json](results/upstream-059-validation.json).
The AtomWorks matrix runs use one timing sample per stage and are compatibility
checks, not new performance comparisons. Their finite energies/gradients and
rotamer coordinates do not independently validate the force-field model or
full packing for all 19 chemistries. Black, Flake8 and whitespace checks pass.

## Nonbonded parameter identity, annotation cost and cache lifetime

The common atom-type, LJ/LK, hydrogen-bond and LK-ball annotations now track
source identity and relevant immutable settings. Each object retains one current
annotation; a renderer captures the returned records so preparing another force
field cannot change its parameters. The shared identity helper uses weak
references and does not retain a history of fitted databases. Topology-only
annotations remain reusable. This fixes reused atom-type orderings, donor
inventories, LK solvation tables and Rosetta/generic pair exclusions. The
corrected baseline reproducer gives **20 failures / four passes / 24 CPU-only
skips**, all failures in energy/gradient comparisons; an earlier invalid test
removed acceptor classes needed by the database itself and is retained only as
diagnostic evidence.

Tests cover both configuration orders, fresh-annotation controls, previously
rendered modules, re-rendering, whole-pose and weighted block-pair derivatives,
and jagged two-pose rotamer scoring. Identical chemical definitions in a different
order must give the same result; disabling donors must give zero hydrogen-bond
energy and gradient. Separate annotation tests change both index order and
heavy-atom classification on the same block and packed objects.

Hydrogen-bond annotation now resolves donor/acceptor names once per chemical
catalog and gathers host arrays per residue. LK-ball retains one host type table
instead of transferring it for every residue. Five alternating baseline/candidate
process pairs, each with warmup and three measured samples, check all public
annotation arrays for exact equality over **230 block types / 4,738 atoms**.
The table includes constructor and annotation time; imports, input construction,
native compilation and result verification are outside the timed region.

| Standalone term | CPU old → new | CUDA old → new |
| --- | ---: | ---: |
| LJ/LK | 79.74 → 79.46 ms | 127.75 → 120.05 ms |
| Hydrogen bonds | 262.27 → 117.76 ms (2.23×) | 388.17 → 151.62 ms (2.56×) |
| LK-ball | 331.99 → 178.49 ms (1.86×) | 508.93 → 223.46 ms (2.28×) |

These are standalone setup measurements including inherited annotations, not
whole-score-function or kernel speedups; shared hydrogen-bond work means the
ratios cannot be added. Warm annotation hits remain about 0.6–0.9 microseconds.
Persistent host arrays grow by 2,520 bytes for hydrogen bonds on either device
and LK-ball on CPU, and 5,670 bytes for LK-ball on CUDA. Device tensor storage is
unchanged. Those counts exclude Python metadata and allocator reservations.

Both hydrogen-bond resolver caches now use a bounded multi-owner weak-identity
LRU. Chemical and scoring databases independently determine validity; collecting
either source removes the entry. Equivalent CPU/CUDA device spellings share a
resolved-device entry. Across 128 separately fitted databases, CPU cached tensor
storage decreases from **16,518,144 to 4,129,536 bytes**, then to **zero** after
sources expire. All 128 parameter fingerprints match the original implementation.
This measures cache-reachable tensor storage, excluding caller-held outputs,
Python metadata and allocator reservations.

The expanded group check also exposed an outdated empty-result assertion from
the earlier explicit-correlation change: frozen groups correctly return
`correlated_gbts=()` alongside their empty groups and plan. The assertion now
checks that complete contract; no geometry or energy tolerance changed.

The CUDA cache stress check reproduces the same byte counts and exact parameter
fingerprints. Final Slurm **249402** completed **0:0** in **5:18**, with **300
passes** across nonbonded reuse, atom types, hydrogen-bond cache lifetime, the
complete LJ/LK and LK-ball directories, reference caches and group regressions.
Naming both a hydrogen-bond test file and its parent directory made pytest
collect only that file: this run does not cover the remaining HBond directory.
The final CPU annotation/cache run gives **55 passes / 48 CUDA skips**. Earlier
Slurm **249300** gives **251 passes / one skip / two existing expected failures**
across the complete nonbonded directories and related caches, before catalog
lookup optimization. The two expected failures are obsolete native point-test
adapters and are being repaired separately. Runs overlap and must not be summed.

The first expanded final run, **249354**, recorded 232 passes and the 12 stale
empty-result assertions described above; its five-round CUDA profile completed
successfully before those test failures. Source hashes, exact case inventories,
paired profiles, cache stress results and terminal Slurm accounting are in
[results/nonbonded-validation.json](results/nonbonded-validation.json).
Black, Flake8 and whitespace checks pass. The upstream head remains
`0593a93b07d80b0302383163d2d98c78e315ab98`; review comments 60–63 cover these findings.

## Restore native hydrogen-bond point-score validation

The two expected-failure point tests also exist in the PR merge base; this is
an inherited coverage gap, not a new PR regression. Running them with
`--runxfail` confirms that they fail before evaluating scores or derivatives:
the global parameter input is a two-dimensional Torch tensor where the native
adapter expects an Eigen vector, and the raw callable lacks the generalized
vectorization signature required by `VectorizedOp`.

The test adapter now accepts flat arrays for all parameter structs and declares
the input/output core dimensions. Its global-parameter caster initializes all
six native fields, including `max_ha_dis`; that last field is used by neighbor
search, not the point kernel. A flat NumPy fixture and explicit coordinate
arguments replace signature introspection. Separate sp2, sp3 and ring cases
retain the original expected energies (**−2.40, −2.00 and −2.17**, absolute
tolerance **0.01**) and the existing finite-difference tolerances for all five
atom-coordinate derivatives. Both expected-failure decorators are removed.
No production kernel or parameter record changes.

The host native point/component suite passes **10 tests**. Slurm **249440** runs
the complete hydrogen-bond directory, without overlapping directory/file
arguments, and passes **52 tests with no skips or expected failures**. This
includes host point bindings and CPU/CUDA parameter annotation, whole-pose,
weighted block-pair, pair-coverage and cache tests. The job completed **0:0** in
**43 seconds**. It closes the final-directory coverage gap recorded above; these
counts overlap earlier runs. Source/log hashes, case names and terminal
accounting are in
[results/hbond-point-validation.json](results/hbond-point-validation.json).
Black, Flake8 and whitespace checks pass.

## Combined sampling budgets and linear rotamer merging

Explicit budgets now also cover the combined conformer count at each physical
residue, summing every sampler and allowed residue type, including input and
fallback rows. Two real-task regressions reproduce the gap: two input samplers,
and design between ALA/GLY. Each source fits a one-state limit independently,
but their union does not. The check rejects that union before row merging or
coordinate allocation and identifies the pose, block and count. A larger limit
builds finite coordinates; a jagged two-pose test checks that residues and poses
have independent limits. Required states are not silently discarded. Default
sampler behavior is unchanged when no explicit task budget is supplied.

The merge now places rows using prefix offsets instead of sorting a global
rotamer key, sorting again for uniqueness, and reconstructing inverse mappings
with `nonzero`. Its output order remains considered type, then sampler, then
original row. Direct destination indices also provide the inverse mapping used
to copy each sampler's degrees of freedom. Mixed int32/int64 source indices,
zero-count types, empty samplers and an empty considered-type axis are supported.
An independent record-enumeration oracle checks every output and source payload.

The ordering contract is validated before indexed writes: source IDs must be
monotonic and match both endpoints of every nonempty count interval. Together,
those conditions require every source row to have its declared type. Negative
counts, mismatched lengths and native-index-capacity violations are rejected.
A further regression checks four int64 counts of `2**62`, whose sum wraps to zero:
the original sort-based implementation also accepts the resulting invalid empty
plan. Validating each count against the available rows closes that edge case.
The merge no longer changes global Torch print options.

The combined-budget check runs after individual samplers have allocated their
private source rows. It does not bound that temporary workspace, total states
across the entire task, or quadratic pair-energy memory. Default/adaptive
library policy and overflow in native sampler products before the merge remain
open; [BUDGETS.md](BUDGETS.md) distinguishes those requirements explicitly.

Final paired benchmarks include count-range and source-order validation. Five
alternating warm rounds, with five samples per round, compare the exact previous
function from `9d49606e0` against the prefix implementation. Every returned count,
index, mask and inverse mapping matches exactly, including mixed index dtypes.

| Merged rows | CPU old → new | CUDA old → new | CUDA peak allocated tensors old → new |
| ---: | ---: | ---: | ---: |
| 360 | 0.192 → 0.195 ms | 0.654 → 0.614 ms | 32,256 → 21,504 bytes |
| 48,318 | 2.920 → 0.877 ms | 0.812 → 0.631 ms | 3,658,240 → 1,538,048 bytes |
| 1,548,739 | 100.564 → 14.524 ms | 1.059 → 0.654 ms | 116,422,144 → 46,385,152 bytes |

The large merge is **6.92× faster on CPU / 1.62× on CUDA**, with about **60% less
peak allocated CUDA tensor storage**. Tiny merges have no meaningful performance
change. These measurements cover only merging; input creation, chemical
preparation, source sampling, coordinates, scoring and parity verification are
outside timing. CUDA peaks are measured above held inputs using the Torch
allocator, not reserved memory or process RSS. CPU memory is not measured.

An independent count-only probe also confirms three unresolved native Dunbrack
cases before the merge: an expansion product of `2**32` becomes zero, a
library/expansion product of `3 * 2**30` becomes negative, and four valid counts
of `2**30` wrap the total to zero and produce negative offsets. The probe uses
small count tables and never allocates those enormous rotamer arrays. This is
evidence for the next native-arithmetic fix, not a claim that merge validation
already protects that earlier stage.

Final Slurm **249542** completed **0:0** in **5:00**, with **275 passes and no
skips** across merge validation, task budgets, rotamer construction, covalent
groups, real packing and nucleotide sampling on CPU/CUDA. The final focused CPU
run gives **28 passes / 27 CUDA skips**. Earlier versions pass 263 tests in job
249474 and 273 in job 249516, before the final count-range check; those runs and
intermediate regression failures are retained as evidence, not additional unique
coverage. The final job moved from the congested interactive partition to
`hpc-mid` while pending, with a revised request of four task CPUs and 32 GB;
Slurm accounting reports eight allocated CPUs and 5,529,640 KiB peak host RSS.

Source hashes, exact case inventories, paired profiles, intermediate source
snapshots, native count-probe results and terminal scheduler accounting are in
[results/sample-merge-validation.json](results/sample-merge-validation.json).
Black, Flake8 and whitespace checks pass. The upstream head is still
`0593a93b07d80b0302383163d2d98c78e315ab98`.

## Check native Dunbrack counts before allocation

The native sampler now checks library-count narrowing, expansion products and
both exclusive count scans before their results size rotamer arrays. Invalid
counts propagate an absorbing negative marker through an associative scan;
the host rejects it at the existing synchronization point. This handles both
negative wraparound and products/sums that wrap back to zero or positive values.
The checks reuse the existing count and offset buffers. Valid counts, offsets,
expansion products and sampled outputs retain their original representation.

This is an inherited arithmetic risk, not a newly introduced PR regression.
Small count-only fixtures reproduce it without attempting enormous allocations.
Tests cover the exact int32 boundary, empty and zero counts, invalid library
counts before public sampler allocation, and errors at the beginning, middle
and end of a 2,053-row parallel scan. Before the fix, seven initial regressions
fail. The final focused CPU run passes **21 tests / 21 CUDA skips**. Slurm
**249675** passes **244 tests with no skips**, covering the complete Dunbrack
directory plus task budgets, covalent groups and real packing on CPU/CUDA.
The job completed **0:0** in **7:10**, including the paired native benchmark,
with **6,980,252 KiB** peak host RSS reported for the batch step.

The benchmark builds the exact preceding header from `0db5eb5d7` in a separate
native namespace. An overflow witness verifies that the two loaded modules
actually execute different implementations; all valid output arrays match
exactly. Five alternating warm rounds give these count-stage medians:

| Buildable types | CPU old → new | CUDA old → new |
| ---: | ---: | ---: |
| 1 | 0.691 → 0.672 µs | 33.185 → 32.724 µs |
| 128 | 1.362 → 1.567 µs | 33.469 → 32.867 µs |
| 4,096 | 20.917 → 27.361 µs | 35.298 → 34.934 µs |

CUDA performance is effectively unchanged. The largest CPU count stage costs
about **6.4 µs more (31%)**; this is a safety check, not a sampling speedup.
Compilation, input construction and count reset are excluded; completion
synchronization is included. Memory and end-to-end sampling are not measured.
The initial baseline benchmark failed to compile because its test bridge still
included the original header; that include was corrected before measurement.

Source hashes, exact test inventories, terminal accounting and complete paired
profiles are in
[results/native-count-validation.json](results/native-count-validation.json).
Review comment 64 describes the upstream locations. Python Black/Flake8 and
whitespace checks pass. Python/nucleotide count arithmetic, adaptive sampling
policy and whole-task/pair-energy memory limits remain separate work.

## Validate Python and nucleotide counts before narrowing

The shared NA/OptH allocation helper now rejects negative counts, individual
counts outside int32 capacity, and totals outside that capacity. It checks the
individual range before trusting the int64 sum, since invalid int64 inputs can
themselves wrap the reduction. Valid nonnegative counts and a representable
considered-type axis cannot overflow that wide sum. Min/max/sum still transfer
to the host together once; explicit per-type budgets remain a separate check.

The nucleotide sampler previously narrowed each product to int32 before checking
it, so a large product could become zero and return an empty result. It now
bounds invalid combination counts with a sentinel before multiplying by the
small mode/step factors, validates the wide count vector, and narrows only then.
That same wide vector is reused to enumerate rows, avoiding a conversion back
from int32. The valid sampling policy and chemical parameters are unchanged.

Eight regressions fail before the fix. The corrected selected CPU suites pass
**37 tests / 36 CUDA skips**. Slurm **249685** passes **104 tests with no skips**
across the new count boundaries, nucleotide sampling, hydrogen optimization,
sampler-cache settings, explicit task budgets and packing on CPU/CUDA. The job
completed **0:0** in **1:54**, with **3,961,408 KiB** batch peak host RSS. Tests
cover empty inputs, exact capacity, combined overflow, int64 wraparound and
invalid metadata in a real RNA sampler while blocking rotamer-row allocation.
Ordinary RNA fixtures have small count products; no gigantic library is built.

Review comment 65 and [BUDGETS.md](BUDGETS.md) record the boundary and remaining
policy limitations. Source hashes, exact test inventories and terminal accounting
are in [results/python-count-validation.json](results/python-count-validation.json).
Black, Flake8 and whitespace checks pass. No performance improvement is claimed
for this guard; index capacity does not guarantee sufficient workspace or bound
quadratic pair-energy memory.

## Isolate Dunbrack sampler annotations and remove scalar device lookups

Dunbrack's RT and PBT annotations previously used `hasattr` as their entire cache
key. Two resolvers with different residue-to-library mappings therefore reused
the first mapping on shared chemical types. The regression assigns ILE the
LEU library in a private resolver; both have two chi, and their fresh sampled
outputs differ. Shared sampling now matches each resolver's fresh annotations
and outputs exactly in both orders, including repeated returns to an earlier
sampler and calls without a separate annotation pass.

One current annotation is stored per RT/PBT with a weak resolver-identity key.
Sampling and backbone-index setup refresh the calling sampler's annotations.
Resolver data are immutable inputs; changing tables requires a new resolver.
The sampler's small name/index/chi metadata are read on the host once, replacing
per-type device-tensor creation and scalar synchronization. RT table indices
now use their declared Python-integer type instead of scalar tensors. Sampler
equality also checks the other object's type and resolver identity, instead of
considering an integer equal merely because its hash matches. A probability
selection that returned 0.98 on both branches is replaced by that common value.

Four initial regressions fail before the fix. The complete native sampler CPU
suite passes **38 tests / 36 CUDA skips** before adding direct-sampling variants;
the final focused identity suite passes **6 tests / 6 CUDA skips**. Slurm
**249687** passes **294 tests with no skips**, covering all Dunbrack tests,
noncanonical sampling, explicit budgets, covalent groups and real packing on
CPU/CUDA. It completes **0:0** in **7:06**, including the paired setup benchmark,
with **6,646,388 KiB** peak host RSS for the batch step.

Five alternating warm rounds, five samples each, compare sampler construction
and first RT/PBT annotation against the exact preceding class from `35350d6e3`.
Every annotation value matches over **230 types / 4,738 atoms**:

| Device | Previous setup | Current setup | Ratio |
| --- | ---: | ---: | ---: |
| CPU | 64.738 ms | 9.425 ms | 6.87× |
| CUDA | 104.047 ms | 10.741 ms | 9.69× |

Database/resolver construction, chemical-object copies, actual sampling and
scoring are outside these timings. Annotation tensor/array fields total
**212,846 → 211,006 bytes**, removing 230 scalar tensor IDs; Python lookup
metadata and allocator overhead are excluded, so this is not a process-memory
measurement. The profiler initially rejected the intentional scalar-tensor to
integer representation change; its comparator was corrected before measurement.

Review comment 66 describes the sampler identity issue. Two independent probes
identify remaining work: the global resolver cache retains a released private
database and **67,494,320 bytes** of unique derived CPU tensor storage; a polymer
with three explicit heavy-chi means is advertised as buildable but receives
zero rotamers because the Python filter requires a library index (comment 67).
The latter probe checks that its target is allowed in the concrete task and
that ten other allowed types receive samples. Scoring-term annotation identity
also remains a separate audit item. No fixes for those three items are claimed
in this commit.

Source hashes, test inventories, terminal accounting, exact paired profiles and
the two follow-up probes are in
[results/dun-sampler-identity-validation.json](results/dun-sampler-identity-validation.json).
Black, Flake8 and whitespace checks pass. The upstream head remains `0593a93b0`.

## Sample explicit polymer chi without a library

The Python Dunbrack wrapper now selects buildable types using its existing
buildability predicate. That predicate accepts amino-acid polymers with explicit
heavy-chi samples even when no library/reference resolves; the previous filter
required a library index and silently removed those types before native sampling.
The native no-library path already provides one base state for their products.

Two further fixes make partial explicit sampling work. The native chi width now
comes from nonnegative slots in the already gathered count table, including
zero-count gaps before later chi. Counting only defined atoms truncated chi2
when chi1 was absent. For no-library types, sidechain roots now come from the
chi actually sampled through the existing shared helper. Thus sampling only
chi2 copies upstream input geometry instead of rebuilding the chi1 branch.
The slot check reuses an existing tensor and reduces directly to int32; it adds
no packed annotation fields. Library-backed root policy and parameter values
are unchanged.

Private ILE topologies with explicit chi definitions provide independent
Cartesian-product and coordinate checks. Tests cover three chi1 means, 54
expanded two-axis states, two-pose masks, an explicit budget smaller than the
required product, requested coordinate torsions, and preserving the input chi1
when chi2 alone varies. These are kinematic/enumeration fixtures, not newly
fitted chemistry. Six regressions fail before the filter fix. Adding the gap
case reveals the separate slot and root assumptions; both intermediate failures
are retained. The final complete Dunbrack CPU suite passes **47 tests / 45 CUDA
skips**.

Slurm **249696** passes **308 tests with no skips** across native sampling,
noncanonical rotamers, budgets, covalent groups and packing on CPU/CUDA. It
completed **0:0** in **6:48**, with **5,589,660 KiB** batch peak host RSS. Earlier
job 249694 was cancelled after 17 seconds when the completed CPU result exposed
changed upstream chi1 geometry; it is not counted as completed validation. The
root fix passed the CPU suite before the final job was submitted.

An additional real-coordinate probe confirms a remaining mainchain-fingerprint
cache problem. A library sampler followed by an explicit-chi2 sampler on shared
types returns the correct two-state count but retains a six-atom copy fingerprint
instead of the fresh thirteen-atom fingerprint. Frozen chi1 becomes −0.08948
instead of 1.05183 radians. Review comment 68 records this separate identity
failure and the need to represent different configurations of one sampler class
within a task. The probe's first output attempt failed JSON serialization of
NumPy indices; conversion to Python integers fixes the report, and the successful
rerun preserves the numeric evidence.

Comment 67 describes the corrected no-library path. General question 4 also
asks for an explicit input-versus-ideal policy for frozen downstream chi; the
upstream-chi regression here does not establish that broader contract. Source
hashes, exact test inventories, intermediate failures, scheduler accounting and
the fingerprint probe are in
[results/libraryless-validation.json](results/libraryless-validation.json).
Black, Flake8 and whitespace checks pass. No performance ratio is claimed.

## Preserve current sampler ownership in mainchain-copy plans

Mainchain fingerprints now follow the current sampler's actual sidechain roots
and weak chemical-database identity. Different instances of the same sampler
class can coexist in one task. Source atoms come from the union of retained
regions; selecting only the largest region loses atoms when regions are not
nested. Each target sampler still copies only its own retained atoms. Empty
samplers return an empty DOF-copy plan without looking up a nonexistent entry.

RT annotations retain only current configurations. Equivalent new sampler
instances reuse the packed tensors without retaining old sampler objects, and
PBT reuse validates the actual current fingerprint payloads. This also corrects
the old cache guard, which checked an attribute that was never stored. The
legacy source/target atom and DOF assertions remain; their source lookup now
uses the explicit source mapping instead of selecting a target sampler.

Fingerprint construction shares one all-atom descriptor calculation across
samplers on a residue and uses local element dictionaries instead of repeated
linear database scans. The full descriptor list is temporary, not an additional
persistent cache. Unique class-name lookup remains available to existing helper
callers; internal sampling uses instance identity.

Four coordinate regressions fail before the fix. Tests compare sequential
sampler switches and two sampler configurations in one two-pose task against
independently built fresh poses, in both orders. Further checks cover nonnested
regions, equivalent-instance tensor reuse over twelve rounds, changed chemical
element definitions, expired owners, and a sampler with no buildable types.
The last case exposed a separate missing-entry lookup and fails before its
empty-plan guard. Final CPU validation passes **40 tests / 35 CUDA skips**.

Slurm **249781** passes **426 tests with no skips** across packing, Dunbrack,
nucleic-acid/OptH sampling, noncanonical and conjugated-group fixtures. It ran
the initial cache implementation before the later zero-state guard and generator
optimization. Slurm **249793** then validates the final source with **75 passes
and no skips**, covering identity, original fingerprint/rotamer tests and
library-free polymers. Both finish **0:0**, respectively **12:44 / 1:40**, with
batch peak host RSS **5,650,620 / 2,509,832 KiB**. These runs overlap.

Paired five-round setup profiles compare the exact preceding implementation
from `dd99ab29e`, over 230 types. Every legal source/target transfer matches:
**80,960 maps** for Dunbrack + FixedAA and **121,440** when adding the task's
default Fallback sampler.

| Samplers | Device | Cold setup, previous → current | Repeated setup, previous → current |
| --- | --- | ---: | ---: |
| Dunbrack + FixedAA | CPU | 119.332 → 125.054 ms | 9.598 → 6.036 ms |
| Dunbrack + FixedAA | CUDA | 120.162 → 128.819 ms | 10.657 → 8.311 ms |
| Above + default Fallback | CPU | 278.561 → 191.619 ms | 15.801 → 10.422 ms |
| Above + default Fallback | CUDA | 278.974 → 199.698 ms | 18.427 → 15.610 ms |

The default three-sampler cold path improves **1.45× CPU / 1.40× CUDA**; packed
tensor fields shrink **848,240 → 813,280 bytes (4.1%)**. With two samplers,
cold setup costs **4.8% CPU / 7.2% CUDA** more and tensor fields grow
**533,600 → 548,320 bytes (2.8%)** to represent the explicit source union.
Repeated setup improves in both configurations. These timings exclude chemical,
resolver and kinforest construction, fresh object copying, sampling and scoring.
Tensor byte totals exclude Python metadata, allocator overhead and process RSS;
no whole-packer or peak-memory improvement is claimed.

Review comment 68 is updated. Exact source hashes, initial-source snapshots,
intermediate failures, test inventories, scheduler accounting and full profiles
are recorded in [results/fingerprint-validation.json](results/fingerprint-validation.json).
Black, Flake8 and whitespace checks pass. Global resolver retention and scoring
annotation identity remain separate follow-up items.

## Release unused Dunbrack resolvers

The inherited unbounded memoizer now uses the existing weak-identity LRU, with
four entries. Keys distinguish the immutable database owner, resolver class and
concrete device; equivalent CPU/CUDA spellings share one entry. The database is
not kept alive by the cache. A caller may retain a compiled resolver after the
source expires, while eviction only drops the cache's own reference. The table
constructor is moved intact behind the cache wrapper.

Five focused CPU regressions fail before this change: released owner/output
retention, more than four live owners, device aliases and subclass lookup in
both orders. The final CPU resolver/scoring/complete-Dunbrack suite passes
**60 tests / 58 CUDA skips**. The independent paired profiler compares all
**49 tensor fields and three DataFrames exactly**, with unchanged
**67,494,320 bytes** of unique derived tensor storage per default-sized resolver.
After releasing a private owner and resolver, that storage remains cached in
the old implementation and is no longer retained by the new cache. This does
not measure allocator reservations, source database memory, native/Python
metadata or process RSS, and four entries do not bound arbitrary library sizes.

Slurm **249841** passes **210 tests with no skips**, covering resolver/scoring,
Dunbrack, fingerprint identity, noncanonical sampling and real protein/group
packing on CPU/CUDA. It completes **0:0** in **00:09:57**, with batch peak host
RSS **6517516K**. The scoring-identity regressions authored after collection are
separate work and are not included in these counts.

Five alternating rounds of 100 warm cache calls measure:

| Device | Previous lookup | Current lookup |
| --- | ---: | ---: |
| CPU | 29.713 µs | 7.259 µs |
| CUDA | 33.930 µs | 8.873 µs |

These are host cache lookups, excluding construction, sampling, scoring and all
device work. The change avoids repeatedly hashing a nested database key; it
does not claim to accelerate the numerical kernels. Comment 69 records the
inherited cache defect. Exact sources, test inventories, profiles and scheduler
accounting are in [results/dun-resolver-validation.json](results/dun-resolver-validation.json).
Black, Flake8 and whitespace checks pass. Upstream remains `0593a93b0`.

## Isolate Dunbrack scoring annotations and simplify setup

RT and PBT scoring annotations now validate the immutable resolver that produced
them, through the shared weak-source annotation helper. A new packed annotation
refreshes its RT inputs; a valid packed hit reuses its tensors directly.
Rendering selects the calling term's own packed data. Previously rendered
modules continue to own their original tensor arguments after a different
parameter set updates the shared PBT. Old resolver/database owners need not be
retained for those compiled tensors to remain usable.

The regression changes or removes ILE's library mapping and compares shared
jagged poses with fresh annotations in both database orders, for whole-pose and
block-pair scoring. Energies and weighted coordinate gradients must match, and
the two parameter sets must actually give different results. A direct PBT-setup
case requires refreshing shared RT annotations. The first attempted jagged
fixture misused a helper for concatenated residue ranges; its eight setup
failures are not counted as product evidence. Correcting the fixture yields
**nine reproduced scoring/annotation failures** before the fix.

Three more tests check released-owner lifetime with a still-live rendered
scorer, reject duplicate lookup names and prohibit per-residue tensor `.item()`
reads during setup. The isolated initial candidate passes 17 CPU tests; the
final production scoring/resolver suite passes **25 tests / 25 CUDA skips**.

Slurm **249931** passes **119 tests with no skips** across scoring/resolvers,
mirror-image scoring, D-residue repacking, noncanonical rotamers, group geometry
and conjugated packing. Slurm preempted the first attempt at 00:11:19 UTC and
automatically restarted it once. Only the completed XML is counted. The final
batch completes **0:0** in **00:09:11**, with peak host RSS **7202176K**.

Setup now reads small name/index/offset metadata on the host once per term,
replacing per-RT Pandas indexing and device-scalar reads. Packing creates an
int32 NumPy table per field and transfers it once, replacing individual device
allocations/assignments for every residue type. The production file is shorter
by 32 lines. The numerical table construction, torsion definitions and scoring
kernels are unchanged.

The paired profiler compares the exact preceding class from `24df324dc`.
Across 230 types, **2,760 RT fields and all 12 packed tensor fields match exactly**.
The packed tensor storage remains **76,360 bytes**. Five alternating rounds of
five samples measure:

| Device | Cold setup, previous → current | Repeated setup, previous → current |
| --- | ---: | ---: |
| CPU | 55.487 → 3.117 ms | 26.300 → 45.369 µs |
| CUDA | 110.069 → 3.767 ms | 26.215 → 43.620 µs |

Cold setup improves **17.80× CPU / 29.22× CUDA**. Repeated setup
adds a small host cost for validating identity instead of checking only attribute
existence. Cold includes term construction and RT/PBT annotation; database and
resolver construction, fresh object copies and actual scoring are excluded.
Tensor byte totals exclude retained host lookup metadata, temporary arrays,
allocator overhead and process RSS. No kernel or whole-application speedup is
claimed. The remapped library is an identity fixture, not a fitted physical model.

Comment 70 records this separate scoring defect. Exact source hashes, intermediate
failures, test inventories, complete timings and scheduler accounting are in
[results/dun-scoring-validation.json](results/dun-scoring-validation.json).
Black, Flake8 and whitespace checks pass. Independent DOF-copy prototypes remain
outside production until their shared-table integration is checked separately.

## Build DOF-copy plans on the tensor device

The mainchain DOF-copy planner now gathers source/destination KFO atom indices
on the device instead of repeatedly round-tripping conformer-index arrays
through NumPy. Original-residue offsets use an int64 prefix sum over the flat
pose layout, with zero contribution from padded blocks. Explicit masks preserve
missing-source/destination behavior and selected conformer order. Atom-index
arrays are released before the next gather, and offsets are added in place to
freshly gathered tensors. The first vectorized prototype used more temporary
memory; these lifetime changes reduce its peak before integration.

One cached KFO table is shared with existing chi correction. For the default PBT
it is **66,240 bytes**, with identical tensor identity in both paths; this adds
no duplicate table to the normal rotamer-building path. The two production
files are **123 lines shorter**. Chi assignment, actual DOF transfer and scoring
kernels are outside this change.

The legacy source assertion compared its expected array with itself. Turning it
into a real comparison exposed invalid fixture inputs: considered-type IDs
0..4 were treated as five physical residues, while those IDs actually belonged
to alternative types for the first residue. The second fixture also constructed
IDs/types by position instead of the task mappings. Both now use actual
considered/allowed entries. Source and destination atoms, source offsets and
both conformers per residue are checked independently. The previous and new
implementations both pass these corrected oracles. The intermediate failed
assertions are fixture-repair evidence, not claimed production correctness bugs.

New tests check subselected/reversed conformer order, empty selection without
pose annotations, no retained regions, no explicit `.cpu()` index transfer and
sharing of the device KFO table. Final CPU validation passes **44 tests / 39 CUDA
skips**. The earlier isolated final prototype passes 75 CPU/CUDA cases, before
these stronger fixture/oracle changes and shared-table integration.

Slurm **250036** passes **436 tests without skips**, covering all Dunbrack,
nucleic-acid and OptH sampling, noncanonical rotamers, task budgets, covalent
groups, geometry and real packing. It completes **0:0** in **00:12:00**, with
batch peak host RSS **5706724K**. Job **250127** separately passes all **five**
mirror-image scoring and D-repacking checks, completing **0:0** in
**00:00:35**, with batch peak host RSS **3495836K**.

The broad job initially queued in `hpc-mid` with a several-day estimate, then
in `hpc-high`. Spare capacity was available on the extra H200 node through
`hpc-low`, so the same job used that partition with a 30-minute limit. Its
1-GPU/4-CPU/32-GiB request stayed unchanged. No other jobs were modified.

Five alternating rounds of five samples compare the exact previous planner
from `5060998cf` with the final implementation. Inputs are two corrected
real-chemistry fixtures, then repeated synthetically for larger conformer counts.
All source/destination indices and their order match exactly before timing.

| Conformers | Copy pairs | CPU previous → current | CUDA previous → current | Extra CUDA peak, previous → current |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 60 | 0.351 → 0.109 ms | 0.984 → 0.304 ms | 15,872 → 9,728 B |
| 10,000 | 60,000 | 2.928 → 1.360 ms | 1.560 → 0.328 ms | 5,946,880 → 4,005,376 B |
| 36 | 234 | 0.344 → 0.112 ms | 0.980 → 0.300 ms | 32,768 → 19,456 B |
| 36,000 | 234,000 | 12.192 → 6.166 ms | 2.610 → 0.324 ms | 22,110,208 → 14,440,448 B |

The first device-table construction is excluded. CUDA peak means additional
allocated tensors above prepared inputs, including outputs and temporaries;
it excludes allocator reservations, host metadata, source coordinates and
process RSS. These are index-construction measurements, excluding actual
sampling, DOF transfer and scoring. No whole-packer speedup is claimed. Prototype
profiles used the old fixture mappings and must not be substituted for these
final comparisons.

Comment 71 records the inherited host-lookup cost and ineffective source test.
Exact sources, intermediate stages, test inventories, paired profiles and
scheduler accounting are in [results/dof-copy-validation.json](results/dof-copy-validation.json).
Black, Flake8 and whitespace checks pass. The separate local AtomWorks aromatic
input fix is documented in [ATOMWORKS.md](ATOMWORKS.md).

## Keep chi assignment on device; extend mirror-image validation

Chi assignment now selects present chi entries once and gathers their residue
KFO indices and ring offsets on the tensor device. It reuses the KFO table from
DOF copying and measured-chi correction. Ring corrections retain one float32
device tensor instead of their previous host array. The two production files
are 19 lines shorter; ordinary chi still receive their later coordinate-based
correction, and ring-closing chi keep their precomputed ideal-geometry offset.

An independent dihedral-projection oracle checks the nonzero PRO ring offset,
reordered conformers, gapped/strided chi columns, all-missing chi, zero columns,
zero rows and every untouched DOF. Index assignment cannot call `.cpu()` in the
focused checks. Three initial failures were malformed chained assignments in
the new test fixture, corrected before final validation; they were not product
failures. Final CPU validation passes **33 tests / 32 CUDA skips**. Slurm
**250227 passes 446 CPU/CUDA tests**, including broad sampler, group, geometry
and actual packing coverage. The final temporary-lifetime version is separately
validated by Slurm **250330**, which passes **137 tests**.

Paired measurements load the exact preceding assignment function and host ring
builder from `458feb22d`. All DOFs match exactly on real ILE/PRO inputs, then on
1,000 synthetic repetitions. Five alternating rounds each contain ten samples;
CUDA is synchronized around timing.

| Sampled conformers | Assigned chi | CPU previous → current | CUDA previous → current | Extra CUDA peak, previous → current |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 8 | 0.098 → 0.050 ms | 0.293 → 0.136 ms | 4,608 → 3,072 B |
| 3,000 | 8,000 | 0.677 → 0.389 ms | 0.407 → 0.141 ms | 358,400 → 320,512 B |

The first device implementation increased the large-case CUDA temporary peak
from 358,400 to 480,256 bytes. Releasing row/column indices and conformer offsets
as soon as they are consumed, adding offsets in place and subtracting corrections
in the fresh sample gather produces the final measurements above. An extra
assertion verifies that the source chi samples remain unchanged.

The retained ring table is 228 bytes for this three-type fixture; the table
scales with residue types/atoms, not conformer count. This moves its retained
storage from host to device on CUDA. The KFO tensor is shared with normal
rotamer construction. First table creation is excluded from timing and peak
measurements. Peaks report additional allocated CUDA tensors above warmed
tables and prepared inputs, not reserved/process memory. No whole-packer speed
or total-memory improvement is claimed.

The mirror suite now checks every score term's reflected coordinate gradient,
finite values and energy agreement in both whole-pose and block-pair modes.
Nonuniform pair weights prevent cancellation from hiding mismatched pair
contributions. Prepared atom names and exact coordinate reflection are checked
first. CPU passes five cases (four CUDA skips); Slurm **250251 passes all nine**
CPU/CUDA mirror cases, including the pre-existing D-repacking check.

**Mirror-image packing remains unvalidated.** The new
[check_mirror_packing.py](check_mirror_packing.py) diagnostic finds **1,308 L versus
1,265 D conformers** for the exact paired `6dmz_mod` fixtures on both CPU and CUDA.
Fifteen residue/type groups differ in count. Ten further equal-count groups
contain nonmatching named heavy-atom conformers under one-to-one assignment.
Hydrogen-name discrepancies are reported separately because chemically
identical hydrogens can exchange names under reflection. These are open
sampling/packing findings; a completed diagnostic process is not a passed gate.

The native sampler floors both L and D backbone lookup coordinates, while the
mirrored sorted-probability table reflects grid points. For interior points,
this selects adjacent source cells after reflection. A private diagnostic that
shifts only the mirrored ordering cells removes 14 of the 15 count mismatches,
leaving the N-terminal ARG at 51 L versus 17 D states. The native missing-torsion
fallbacks are always −60° phi / +60° psi. The cell shift is **not integrated**:
a complete fix must cover exact bin boundaries, periodic wrapping, terminal
fallbacks, mixed chirality/design, geometry and packing energy tables.

Slurm 250256 produces the same failing structured mirror-packing diagnostic on
CUDA. Source hashes, intermediate failures, complete test inventories, timings,
CPU/CUDA diagnostic groups and terminal scheduler accounting are in
[results/chi-assignment-validation.json](results/chi-assignment-validation.json).
Black, Flake8 and whitespace checks pass. Review comments 72–74 distinguish the
completed assignment optimization from the unresolved mirror-packing gate.

The periodic-boundary audit also reproduces selection of forbidden bin 36 at
+π on CPU/CUDA. The float32 wrapped value equals the period, so the strict `>`
loop does not reduce it before indexing the 36-bin sorted table. A guarded
37×37 diagnostic assigns a valid but different rotamer to the extra row/column,
proving the wrong lookup without an actual out-of-bounds read. Equivalent phi
−π/+π yields probabilities 0.752777/0.081837; equivalent psi yields
0.220622/0.024137. Slurm 250349 completes this diagnostic in 13 seconds. Both
probability ordering and chi reconstruction contain the same index arithmetic.
Comment 74 and [reproduce_dun_periodic_boundary.py](reproduce_dun_periodic_boundary.py)
record this additional open defect; it is not fixed by the chi-assignment
optimization.


## Wrap sampling endpoints in one native helper

Both Dunbrack sampling stages now call one periodic-coordinate helper. It uses
an exclusive upper endpoint and also wraps a coordinate that rounds up to the
bin count during division. It uses the supplied period for either direction,
removing the separate hardcoded negative-angle wrap. Ordinary canonical
sampling remains covered by the existing numerical reference tests.

Four guarded regressions fail before the change: phi/psi endpoints in both
probability selection and chi reconstruction. They pass after it. The full
native Dunbrack CPU suite passes **51 tests / 49 CUDA skips**; Slurm **250352**
passes **188 CPU/CUDA tests**, including noncanonical sampling, construction,
mirror energies/gradients and D repacking, with **0:0**, **4:54** elapsed and
batch peak RSS **6,351,556 KiB**. The updated standalone diagnostic reports
endpoint agreement explicitly. No performance gain is claimed for this fix.

The initial attempted float64 cases were unsupported by the float32-only test
bindings; those four binding errors are kept separate from the four reproduced
product failures. Exact test inventories and source hashes are in
[results/periodic-lookup-validation.json](results/periodic-lookup-validation.json).
Comment 74 is now corrected on the follow-up branch. The separate L/D
conformer-set tests still fail at both default and expanded chi settings;
periodic wrapping alone does not complete mirror-image packing support.


## Mirror sampling, periodic scoring and complete glycine geometry

The branch now preserves reflected library ordering cells, missing-torsion
defaults and chi means. Probability ordering uses the source library's cell
before mapping that cell into the reflected grid; each target library resolves
its own missing-backbone defaults. Explicit library metadata survives legacy
loading, renaming and serialization. The additional sampling tensor contains
36 booleans (**36 payload bytes**, excluding allocation rounding and Python
metadata); there is no new per-conformer reflection buffer.

Mean interpolation uses the reflected angular branch for D libraries. Both
scoring paths now measure the periodic chi-minus-mean difference, so full-turn
representatives do not alter energy or coordinate derivatives. The CPU scoring
references and numerical gradient checks pass. The intermediate implementation
with corrected means but the old scorer failed three scoring checks; those
failures are retained in the manifest rather than presented as successes.

Fixed glycine sampling now rebuilds both alpha hydrogens, correcting a D-pose
conformer with only **0.081 Å** between them. The existing `with_symmetric_gly()`
option also averages their ideal C–H lengths and harmonic bonded targets/force
constants, including terminal forms. The original default YAML parameters are
unchanged. Only five GLY HA2 coordinate rows in the ubiquitin rotamer reference
were updated after verifying that every other one of its 34,630 atom rows
still matches the existing tolerance. Independent hydrogen-side, bond-length,
full-atom reflection and per-term energy checks support that update.

The exact L/D fixture now offers **1,308 conformers on each side**. One-to-one
matching covers all atoms, allowing hydrogen permutations only when atom types
and named neighbors agree. All 24 configured score components' complete
rotamer interaction matrices agree on CPU and CUDA; absent chemical classes
still contribute zero and are not independently exercised by this fixture.
Expanded-chi counts/heavy geometry and each missing-axis case pass separately.
A further native sweep covers every default grid point, adjacent floating-point
values and missing backbone axes across all 18 library pairs, for both
probabilities and chi means.

Actual CUDA packing fixes every position except one PHE. Both reflected inputs
reach the minimum found by exhaustive whole-pose scoring of every offered
choice, and the final structures reflect one another. This is stronger than a
chirality-only check, but does not require arbitrary multi-position stochastic
trajectories to make identical choices.

Final validation: CPU **24 passes / 23 skips** for canonical reference updates,
symmetric geometry and full-atom energies; Slurm **250496: 311 passes / one
intentional CPU annealer skip**, **0:0**, **9:28** elapsed, **6,468,352 KiB** batch
peak RSS. The additional all-grid sweep is **two CPU passes / two CUDA skips**
and **four CPU/CUDA passes** in Slurm **250506**, **0:0**, **20 s**, **1,721,280 KiB**.
The prior broad run **250427: 289 passes** covers the library/scoring fixes
before the later GLY changes. Job 250426 was deliberately cancelled when CPU
checks exposed the intermediate scorer defect. Job 250486 passed full-atom
energies and actual packing but failed the two old hydrogen coordinate
references; the five-row update and final run supersede those failures.

Exact source hashes, test inventories, separate failure stages, scheduler
accounting and limits are in [results/mirror-packing-validation.json](results/mirror-packing-validation.json).
No speed gain is claimed for these correctness corrections. Arbitrary custom
grid registrations, empty library families, generation based on a `d` name
prefix, the default attachment-parameter gaps, and the broader completion
requirements remain separate work.


## Generate only requested mirrored libraries

Mirrored-library generation no longer treats a `d` prefix as proof that all D
chemistry is covered. It checks requested target mappings, preserves explicit
ones, adds only missing libraries and returns the same database for a no-op.
Partial generation can be extended later without discarding existing library
objects. Source/name/target conflicts fail at generation instead of producing
ambiguous or unreachable tables.

Eight pre-fix regressions fail. The corrected initial CPU suite passes 24 cases
(three CUDA skips), and the final selective native-parity/scope suite passes ten
(one CUDA skip). Slurm **250526** passes **54 cases / one intentional CPU
annealer skip**, **0:0**, **1:27**, **5,155,204 KiB** batch peak RSS. This includes
full-atom mirrored interaction tables and actual controlled CUDA packing after
the generation change. All complete default generated library values match the
preceding function exactly; selective ARG/DARG/PHE native outputs also match.

| Requested D chemistry | Previous generation | Selective generation | New tensor storage, previous → selective |
|---|---:|---:|---:|
| Complete default set | 15.79 ms | 15.86 ms | 30,927,608 → 30,927,608 B |
| DARG only | 17.28 ms | 1.65 ms | 30,927,608 → 4,279,200 B |
| DSER only | 17.29 ms | 0.089 ms | 30,927,608 → 77,784 B |
| No D types | 17.22 ms | Early return | 30,927,608 → 0 B |

These are seven alternating warm rounds of three uninstrumented calls, with
source/database construction outside timing. Storage excludes shared L tables,
Python metadata, temporary peaks and allocator overhead. No whole-application
speedup is implied, and the default full chemistry set remains unchanged.
Exact hashes, stage-separated tests, timing samples and scheduler accounting are
in [results/mirrored-library-generation.json](results/mirrored-library-generation.json).
Comment 77 is corrected. Empty library families and non-grid-aligned custom
registrations remain separate audit items.
