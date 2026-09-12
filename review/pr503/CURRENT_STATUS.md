# Review checkpoint: shared AtomWorks implementation

Frank's latest fetched #503 head remains `0593a93b07d80b0302383163d2d98c78e315ab98`,
already merged. The shared-production checkpoint is tmol `75e02157b` and
AtomWorks `774056c7` based on latest dev `59afb1e2`.

[AtomWorks PR #349](https://github.com/baker-laboratory/atomworks-dev/pull/349)
is open as a draft, targeting `dev`. Tmol now requires that immutable companion
revision and removes duplicate protonation, repair, converter and CIF-category
implementations. It preserves its own scientific rules and observed HIS ring H.

- AtomWorks chemistry/adjacent suites: 180 passed; final affected selection: 74 passed.
- Prior tmol engine versus shared engine with tmol rules: 774 mapped-state comparisons match.
- All six existing native HIS coordinate fixtures match the AtomWorks selector.
- Tmol reader/HIS/CIF selection: 23 passed.
- Final committed-implementation checks: **46 tmol tests and 34 AtomWorks histidine tests passed** (overlapping selections, not additional unique coverage).
- Slurm 252647 completed: 1,468 passed, 15 skipped, 10 failed. Eight are inherited score references; the stale mock is now fixed/passing; one unchanged-scorer CUDA repeat assertion remains intermittent. All 18 CPU/18 CUDA examples pass.
- Slurm 252648 completed all 132 corpus trials. Both modes keep all pre-sharing pass/partial/fail outcomes; only four trials per mode report convergence.

See [PROTONATION_AUDIT.md](PROTONATION_AUDIT.md) and the per-comparison result
artifact. There are 108 recorded findings, including dependency/follow-up issues
that were not introduced by Frank. The tmol PR description includes the completed reruns and every problem/fix
disposition and efficiency change.

Historical production `01159dd7a` validation remains recorded: 1,459 passed,
15 skipped, eight noncanonical score-reference failures, and 18 CPU/18 GPU
examples passing. Those counts are not claimed for the new shared checkpoint.
Default attachment model selection, noncanonical golden provenance, certain
incomplete inputs and 8OG leaving-branch ambiguity remain explicit limitations.
Metals are deferred. A numerical minimization pass does not imply convergence.

The temporary AtomWorks pin requires authenticated Git access and must become
a public released dependency before package publication. Full file/tensor API
unification and removal of model-specific wrappers remain design proposals.
