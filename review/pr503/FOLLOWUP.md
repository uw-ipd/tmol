# Correctness and performance follow-up

Active objective: address the review changes and establish correctness and
performance for supported CPU/GPU workloads. Starting implementation/reference:
`96ff844e82ecb7dff0958d6b419e7aa2a946694a`. The original review and results remain
historical evidence, not a claim that the follow-up is complete.

## Completion requirements

| Area | Required evidence | Status |
|---|---|---|
| Import/closure inference (1) | Fresh checkout collection; explicit/inferred closure, padding, breaks and caps on both devices | Existing replacement; extend audit |
| Mixed chirality (2) | Published reference parameters/formula; LL/DD/LD/DL permutation and reflection energies/gradients; whole-pose and packing parity | Rosetta mixed distribution and shared derivatives implemented; independent LL/LD/DL/DD energy/gradient, permutation and reflection checks pass on CPU/CUDA; additional packing parity pending |
| Group identity and safety (3–7,9,30,31,33–37) | Real multi-pose/multi-database groups, repeated chi names, jagged/empty counts, early rejection; native parity | Capped anchors, rigid cyclic cores, external attachments and sampled pendant branches tested CPU/CUDA, including packing and two-pose reuse; later task-mask changes still violate sampling ownership |
| Sampling budgets/caches (8,14,27) | Explicit budget semantics; task propagation; immutable sampler reuse; actual library/extra/proton counts; reproducible HYP count | Explicit task overrides and actual group/library bounds implemented/tested CPU/CUDA; aggregate/default budget policy and adaptive library expansion remain open |
| Residue identity/completion (10–12,16,32) | Insertion codes, chain identity, explicit/CCD authority, missing atoms, custom names; realistic scaling | Reader annotation reuse and finite-geometry repair checks pass; AtomWorks parser/converter profiles recorded; broader identity/authority contracts remain open |
| Content/profile caches (13,20,29) | No serialization aliases; safe object lifetimes, bounded memory; repeated preparation | Content framing and shared bounded weak caches fixed for rotamer/alpha/NA profiles; identity/lifetime/LRU/concurrency tests pass; remaining caches still need audit |
| Group kinematics/performance (15,17,35) | Profile CPU/GPU; batch invariant work; retain all covalent constraints for tree/cyclic/multiple-anchor topologies | Bounded conformer batches and 32-entry shape caches verified/profiled; independent axes preserve rigid cycles/external boundaries; correlated ring-pucker sampling and task-imposed constraints remain open |
| Native maintainability (18) | Shared improper enumeration with unchanged canonical scores and independent analytic/numeric derivatives | Shared helper in all four paths; canonical references, numerical gradients, independent Cartesian reference and group packing pass CPU/CUDA (238323, 238529) |
| Duplicate work (19) | Removal covered by chemistry regression suite | Existing fix; final suite pending |
| Correct test parameters (21) | Prepared database used throughout examples and packing; generated parameter coverage | Existing correction; final audit pending |
| Terminal caps (22) | ACE/amide caps build intact chains, finite coordinates/gradients, correct bond topology; CPU/GPU packing | ACE/NH2/NME construction, bonds, score/gradient and rotamer construction pass CPU/CUDA; amide geometry and actual packing pass CPU/CUDA (237089) |
| Reference scores (23) | Explain reference differences; independent scoring checks; reproducible pinned structures/parameters | Pending; no blind golden refresh |
| D geometry (24) | Signed stereocentre volumes and actual library sampling after repacking; reject mislabeled geometry | Every offered D rotamer and packed result retains signed alpha-centre volume on CPU/CUDA (237089); broader stereocentre coverage remains |
| Fold trees/fragments (25,26) | Branch-point invariant, fragment restoration/minimize/pack/DDG and multi-pose checks | Existing fixes; final suite pending |
| Scientific ownership (general 2,7) | Independent potential checks and canonical change audit; sampling/scoring reference separation | Pending |
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

Still open: subsequent task masks and disabled members, multiple member
libraries, completely rigid member types, free groups without a polymer anchor,
coupled ring conformers, and remaining design/multi-database cases. A current
task-mask diagnostic confirms that disabling the group sampler still emits its
235 conformers plus fallback (236 per member); disabling one member gives
235/235/236 counts. The sampler must consume those task constraints before
enumeration. This is the next correctness gate, not covered by the passing
unrestricted packing cases above.
