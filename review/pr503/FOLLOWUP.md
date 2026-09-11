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
| Sampling budgets/caches (8,14,27) | Explicit budget semantics; task propagation; immutable sampler reuse; actual library/extra/proton counts; reproducible HYP count | Pending |
| Residue identity/completion (10–12,16) | Insertion codes, chain identity, explicit/CCD authority, missing atoms, custom names; realistic scaling | Initial unit fixes exist; broader profiling pending |
| Content/profile caches (13,20) | No serialization aliases; safe object lifetimes, bounded memory; repeated preparation | Content framing and bounded weak profile cache fixed; identity/lifetime/LRU tests pass; broader cache audit pending |
| Group kinematics/performance (15,17) | Profile CPU/GPU; batch invariant work; retain all covalent constraints for tree/cyclic/multiple-anchor topologies | Pending |
| Native maintainability (18) | Shared improper enumeration with unchanged canonical scores and independent analytic/numeric derivatives | Pending |
| Duplicate work (19) | Removal covered by chemistry regression suite | Existing fix; final suite pending |
| Correct test parameters (21) | Prepared database used throughout examples and packing; generated parameter coverage | Existing correction; final audit pending |
| Terminal caps (22) | ACE/amide caps build intact chains, finite coordinates/gradients, correct bond topology; CPU/GPU packing | ACE/NH2/NME construction, bonds, score/gradient and rotamer construction pass CPU/CUDA; full packing/geometry gate pending |
| Reference scores (23) | Explain reference differences; independent scoring checks; reproducible pinned structures/parameters | Pending; no blind golden refresh |
| D geometry (24) | Signed stereocentre volumes and actual library sampling after repacking; reject mislabeled geometry | Pending |
| Fold trees/fragments (25,26) | Branch-point invariant, fragment restoration/minimize/pack/DDG and multi-pose checks | Existing fixes; final suite pending |
| Scientific ownership (general 2,7) | Independent potential checks and canonical change audit; sampling/scoring reference separation | Pending |
| API/authority/reproducibility (general 1,6,8,9) | Executable input-route examples, settings/seed provenance, documented migration and fresh-checkout CI | Pending |
| Workload scale/release (general 4,5,10) | Explicit matrix of chemistry/topology/input/batch/stage/backend coverage; paired profiles/timings/memory, no masked regressions | Initial stage profiler added; baseline measurements pending |

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
