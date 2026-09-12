# Current review checkpoint

Frank's latest fetched head remains `0593a93b07d80b0302383163d2d98c78e315ab98`,
already merged. The current production fixes are tmol `102cda060` and AtomWorks
`4af94d0c` based on dev `59afb1e2`. Expanded CPU/CUDA validation of an immutable
snapshot is running in Slurm job **253074**; no completed result is claimed yet.

Both CIF identifier routes now share AtomWorks bond parsing. Complete 1xvk and
145d regressions retain their declared topology and pass CPU scoring/minimization.
1xvk also preserves its initial energy under reversed residue order after fixing
generic torsion direction/priority and central-bond lookup. Declared disulfides
survive missing/distant sulfur coordinates. Invalid patch combinations are
rejected before losing their torsion support. The latest focused selection has
**30 passed, 18 device skips**; preceding shared-reader/corpus checks have
**39 passed, seven skips**. See
[incremental evidence](results/topology-focused-validation.json) for revision scope.

[tmol draft PR #508](https://github.com/uw-ipd/tmol/pull/508) targets Frank's
branch. [AtomWorks draft PR #349](https://github.com/baker-laboratory/atomworks-dev/pull/349)
targets dev. Both remain drafts with explicit scientific and input limitations.

OpenFold/RF2 constructors, factories, exports and vendored model tables are
removed fully. There are no compatibility wrappers. The
[Colab notebook](../../notebooks/example_02_model_inputs.ipynb) shows application-owned
interfaces using named layouts and Torch canonical construction, including an
explicit RF2 hydrogen policy. Two complete CPU/CUDA notebook workflows pass: exact
C-alpha coordinates, finite scores/gradients, and no input gradient for rebuilt
hydrogen slots, and repeated guidance. Its download/install path also completed
in an isolated local environment; execution on Google's hosted Colab service
has not been independently verified. The notebook requires authenticated access
to the unpublished AtomWorks source. This caches mappings, not final topology; the broader unified
prepared-topology atom14/atom37/backbone4 API remains a proposal.

- Earlier `4b4ff72d9` broad GPU checkpoint: **1,478 passed, 15 skipped, eight inherited noncanonical
  score-reference failures**. All **18 CPU and 18 CUDA examples pass**. No golden
  or tolerance was changed.
- Subsequent cap-frame/diagnostic fixes: **86 CPU passes, six device skips** and
  **118 CPU/CUDA passes**. These predate the topology and generic-lookup fixes.
- AtomWorks full IO: **766 passed, 13 skipped, four 1twr failures**, all four
  reproduced on unchanged dev. The full run preceded the final allocation-only
  neutralization cleanup; its five identity cases were rerun and pass.
- Added AtomWorks tests: **48 cases in six files**, reduced from 151/ten in favor
  of integrated workflows. All 48 pass. Existing upstream tests are unchanged.
- All **774 original protonation comparisons match**. Three new mixture
  comparisons intentionally differ because the old engine discarded molecules.
- Earlier **324 scoring/minimization trials completed**: 132 local-file trials and
  192 trials over the 96 PDBs referenced by AtomWorks IO tests. Every reader/input
  ran in a fresh CUDA process, with up to 100 LBFGS iterations and a 300 s timeout.

| Corpus mode | Native pass / partial / fail | AtomWorks pass / partial / fail |
|---|---|---|
| Local files, original input | 14 / 2 / 17 | 11 / 2 / 20 |
| Local files, explicit free-metal exclusion | 16 / 6 / 11 | 12 / 8 / 13 |
| 96 PDBs, explicit free-metal exclusion | 57 / 4 / 35 | 22 / 40 / 34 |

The two authored 1j8z files now score/minimize through AtomWorks. Preserving
8OG OP2 moves metal-excluded 6w13 from read failure to partial minimization.
Every other local status is unchanged from the earlier shared checkpoint.
Partial means additional constructor residue exclusions; numerical pass does
not imply convergence. Only four trials in each local mode and nine of the
192 PDB trials report convergence. Connection-length alerts remain explicit.

There are **117 recorded findings**. See [CONTINUED_REVIEW.md](CONTINUED_REVIEW.md)
for new anchored comments, experimental diagnostics, failure triage and proposed
fixes; [PR_DESCRIPTION.md](PR_DESCRIPTION.md) maps every finding to its fix or
remaining limitation. Full results, source hashes and optimizer states are in
`results/continued-*-validation.json`.

The attachment parameter source is settled: follow Frank's hybrid model and
his generated-geometry Cartesian convention (`K=300` lengths, `K=80` angles).
The MMFF94 harmonic prototype remains diagnostic-only. Completing missing
attachment/local terms, reconciling noncanonical golden provenance, validating
broader polymer/patch contexts, and handling certain incomplete inputs remain
outstanding. Metals remain
deferred. The immutable AtomWorks dependency needs authenticated Git access and
must be replaced by a public release before package publication. Native CIF
reading remains the default while normalization/provenance policy is settled.
