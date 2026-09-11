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
| Group identity and safety (3–7,9) | Real multi-pose/multi-database groups, repeated chi names, jagged/empty counts, early rejection; native parity | Initial fixes exist; integration/property coverage pending |
| Sampling budgets/caches (8,14,27) | Explicit budget semantics; task propagation; immutable sampler reuse; actual library/extra/proton counts; reproducible HYP count | Explicit task overrides and actual group/library bounds implemented/tested CPU/CUDA; aggregate/default budget policy and adaptive library expansion remain open |
| Residue identity/completion (10–12,16) | Insertion codes, chain identity, explicit/CCD authority, missing atoms, custom names; realistic scaling | Initial unit fixes exist; broader profiling pending |
| Content/profile caches (13,20) | No serialization aliases; safe object lifetimes, bounded memory; repeated preparation | Content framing and bounded weak profile cache fixed; identity/lifetime/LRU tests pass; broader cache audit pending |
| Group kinematics/performance (15,17) | Profile CPU/GPU; batch invariant work; retain all covalent constraints for tree/cyclic/multiple-anchor topologies | Pending |
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
