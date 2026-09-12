# Current review checkpoint

Frank's latest fetched head remains `0593a93b07d80b0302383163d2d98c78e315ab98`,
already merged. Current review code is tmol `9a91bce6b` and AtomWorks
`4af94d0c` based on dev `59afb1e2`.

The immutable `102cda060` broad run finished with **1,667 passed, 15 skipped,
27 failed** and all **18 CPU + 18 CUDA examples passing**. Nineteen failures
exposed hydrogen-policy/fixture compatibility assumptions; their focused CPU
checks now pass without changing score goldens or tolerances. The other eight
are inherited fresh-preparation noncanonical score references. The expanded
`6560d12e1` run finished in Slurm **253088** with **1,693 passed, 15 skipped,
nine failed**; all **18 CPU + 18 CUDA examples pass**. Eight failures are the
noncanonical references; one stale mock lacks the required protonation-state field and
is already corrected in `88096fafe` with a passing CPU check. The final port
guard/conversion selection has **85 CPU/CUDA passes** in job **253094**, with
no failures or skips.

The private capped-model prototype now retains observed tetrahedral
stereochemistry before discarding coordinates. Three mirrored full-fixture
regressions fail before the fix and pass afterward; its focused CPU selection
has **53 passes, 22 skips (12 device, ten fixture-specific)**. The six Frank-convention native
energy/gradient comparisons pass on each of CPU and CUDA with stereochemistry retained in the chemical
identity check. Its CPU/CUDA selection has **65 passes, ten fixture-specific skips** in **253105**.
See [stereochemistry evidence](results/conjugate-stereochemistry-validation.json).
This remains a prerequisite for default attachment integration, not a new default.

A subsequent generator audit fixed DAR/DAL/U lookup through declared input
classes. Its three complete-file regressions pass. All 18 example inputs now
complete the generator audit: three produce ligand/glycan attachment records
and fifteen need none. This does not replace the final full corpus rerun.

Both CIF routes expose an explicit preserve/rebuild hydrogen policy, including
D-histidine evidence. Legacy inputs without a bond table retain geometric
disulfide detection; supplied bond graphs remain authoritative. Mol2 aromatic
annotations now require ring membership, and unresolved carboxylates reuse
AtomWorks bond-order repair without repeating it for valid carbonyls.

Targeted CUDA corpus job **253092** passes scoring/minimization for complete
145d, 1xvk and 1IZC through both readers. Original 1AYM still fails: its source
explicitly declares two partners for one MYR connection port. The new guard
reports that conflict before graph writes; both-reader complete-file regression
checks pass. Focused CPU selections have **115 semantic/typing passes** and
**26 input/topology passes, 20 device skips**. These are overlapping checks,
not counts to add to the broad suite. See
[revision-scoped evidence](results/chemistry-followup-validation.json).

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

There are **120 recorded findings**. See [CONTINUED_REVIEW.md](CONTINUED_REVIEW.md)
for new anchored comments, experimental diagnostics, failure triage and proposed
fixes; [PR_DESCRIPTION.md](PR_DESCRIPTION.md) maps every finding to its fix or
remaining limitation. Full results, source hashes and optimizer states are in
`results/continued-*-validation.json`.

The attachment parameter source is settled: follow Frank's hybrid model and
his generated-geometry Cartesian convention (`K=300` lengths, `K=80` angles).
The MMFF94 harmonic prototype remains diagnostic-only. Frank-convention generated
attachment records pass six CPU whole-pose/block-pair energy and gradient
comparisons across biotin and both glycan fixtures. This is diagnostic validation,
not default integration or independent stereochemical validation. Completing missing
attachment/local terms, reconciling noncanonical golden provenance, validating
broader polymer/patch contexts, and handling certain incomplete inputs remain
outstanding. Metals remain
deferred. The immutable AtomWorks dependency needs authenticated Git access and
must be replaced by a public release before package publication. Native CIF
reading remains the default while normalization/provenance policy is settled.
