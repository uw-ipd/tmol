# PR #503 validation

This page records the initial review runs. For subsequent mixed-chirality,
AtomWorks, terminal geometry, sampler-cache and native-refactor work, see
[FOLLOWUP.md](FOLLOWUP.md) and [ATOMWORKS.md](ATOMWORKS.md).

Review and code changes: [REVIEW.md](REVIEW.md). Machine-readable counts, per-file coverage, and failure messages: [results/test-runs.json](results/test-runs.json).

## What was tested

The exact upstream head is `c03c1e745f3bc655948ea12dac44d6c74620358f`. Both an isolated CPU environment and an H200 Slurm job failed during pytest conftest import: `tmol.io.details._cyclic_search` is absent from that commit. **No tests collected on the exact submitted tree.**

To investigate beyond that blocker, the baseline worktree received only the replacement `_cyclic_search.py` supplied on this branch. This was not recovered upstream code: it was implemented from the committed API/documentation. All later “baseline” results below mean **PR head plus that one module**, not an unmodified PR. Reviewer-only reproduction tests were added separately; they were not in the already-collected broad baseline suite.

The improvement branch started from the exact PR head, independently of both preexisting workspaces with merge conflicts. Final implementation commit: `f0817a2ee`. The later report commit changes only review artifacts. The PR author branch, upstream master, and the user's RFProteina checkout were not modified.

## Runs

Counts are per invocation and **must not be summed**: suites overlap. The broad runs began before the final test and fragment corrections. Targeted final-state runs supersede the affected results.

Across 1,044 distinct collected cases, the latest executed observations are **1,026 passed, ten failed, and eight skipped**. This is combined coverage across runs and environments, not one clean invocation of the entire suite at the final commit. The JSON record lists the precedence of runs and the unresolved cases.

| Run | Passed | Failed | Skipped | Scope / interpretation |
|---|---:|---:|---:|---|
| Exact upstream, CPU and GPU | 0 | import blocker | — | Missing `_cyclic_search.py`; no collection |
| Diagnostic baseline, CPU | 157 | 0 | 9 | D database; noncanonical amino acids, backbones, nucleic acids |
| Candidate chemistry, CPU | 157 | 0 | 9 | Same selection as diagnostic CPU |
| Broad baseline, CPU + H200 | 684 | 47 | 7 | 738 cases; all ligand tests plus new chemistry/group/scoring/kinematics suites |
| Initial broad candidate, CPU + H200 | 708 | 47 | 7 | Same suites plus 24 initial reviewer regressions; identical failing test IDs |
| Candidate compatibility, CPU | 104 | 2 | 71 | Existing I/O, residue selection, packing, score-function and fold-tree cases; two stale fold-tree assertions |
| Final focused regressions, CPU | 24 | 0 | 10 | All new unit regressions; CUDA parametrizations skipped locally |
| Final focused chemistry, CPU + H200 | 77 | 8 | 4 | All 34 new regression cases pass; prepared-database group tests, cyclic workflows, mirrors; eight upstream reference-score failures |
| Native/reference coverage, baseline CPU + H200 | 108 | 0 | 0 | Cartbonded, disulfide, Dunbrack, Rama database, named pose torsions |
| Native/reference coverage, candidate CPU + H200 | 108 | 0 | 0 | Same selection; includes whole-pose/block scores and finite-difference gradient checks |
| Corrected fold-tree assertions, CPU | 7 | 0 | 1 | Split branch-point expectations agree with the existing builder and validator |
| Final fragments/conjugations, CPU + H200 | 129 | 0 | 4 | 133 cases: fragment restoration/packing/minimization/parity, canonical selection, group packing, conjugations, fold trees |
| Remaining compatibility cases, H200 | 56 | 0 | 1 | CUDA parametrizations skipped in the CPU compatibility run; 81 other cases deselected |

The final fragment run turns all 35 upstream fragment failures into passes and reruns both corrected fold-tree tests. All four group-packing skips are preexisting CPU skips for the larger glycan fixtures; their CUDA counterparts pass. No expected score files were regenerated and no failing test was marked xfail to make the branch appear clean.

The other four distinct skips are three BTN inputs referenced through an unavailable external dataset and the CPU parametrization of a CUDA-graph test. That test's CUDA parametrization passes.

Slurm jobs: `232850` exact baseline; `232855` diagnostic baseline; `232858` broad candidate; `233033` final focused tests and GPU examples; `233153` native/reference baseline and candidate; `233224` final fragment/conjugation checks; `233343` remaining CUDA compatibility cases. All used the `interactive` partition and an NVIDIA H200. Initial native extension compilation is included in the broad run durations, so those durations are **not a packing-performance comparison**.

## Remaining failures

The two broad runs fail on exactly the same 47 test IDs. They divide into 35 fragment construction failures, two stale fold-tree assertions, eight noncanonical reference-score mismatches, and two HYP rotamer-count mismatches. The branch resolves the first 37 in the final follow-up run.

The remaining ten test failures are:

- `test_noncanonical_scores_match_baseline`: all four classes (`alpha_aa`, `dna`, `nonstandard_aa`, `nonstandard_na`) on CPU and CUDA. Baseline and candidate report the same shifted terms and values to the printed precision, except one CUDA `fa_lk` value differs by 0.0001. For the beta-peptide example, `fa_ljrep` is about 121.26 versus the committed 662.72. This is an upstream reproducibility/scientific-validation issue, not evidence that either reference or implementation should automatically be accepted.
- `test_a_borrowed_library_yields_its_own_number_of_rotamers[...-HYP]`: 18 observed versus six expected, on CPU and CUDA in both baseline and candidate. The intended contribution of extra-chi expansion needs clarification; the test expectation was not overwritten.

The standard cartbonded/disulfide checks and the genbonded noncanonical gradient checks pass. This does **not** resolve the independent mixed D/L disulfide-order bug: the ordinary suite does not exercise that permutation. [reproduce_disulfide_order.py](reproduce_disulfide_order.py) changes only residue ordering at fixed coordinates and obtains disulfide-only scores of −0.0337972641 and 0.5349465609. Scientific changes to the pair-chirality potential were left for review.

## Example workflows

[run_examples.py](run_examples.py) executes all 14 new noncanonical CIFs, all three covalent-component CIFs, and the cyclic-peptide CIF. Each run prepares ligands with seed `20260909`, uses the returned **prepared** parameter database, constructs a whole-pose score, calls backward, and checks every energy and coordinate gradient for finiteness. `no_optH=True` isolates preparation/reconstruction and scoring; packing is covered separately by pytest.

**16 of 18 examples pass on CPU, and the same 16 pass on H200.** These cover D/L peptides, beta/gamma backbones, hydroxyproline, N-methylation, phosphorylation, modified DNA/RNA, biotinylation, N/O-linked glycans, and a cyclic peptide. CPU and GPU here use different Python/Torch/NumPy/Biotite environments; these example runs are not a strict cross-device numerical-parity test. The dedicated parity and mirror tests use a shared environment.

The failures are `capped_peptide_ace_nh2.cif` and `capped_peptide_ace_nme.cif`. Both raise `RuntimeError: failed to resolve a block type from the candidates available ... Best candidate exceeds failure threshold`. The baseline reproduces both. Disabling cyclic inference still fails: ACE and the terminal amide cap have no candidates in their requested terminal slots. Their prepared base types intrinsically lack one polymer connection, while candidate classification derives terminal status from patch names. The preparation-only cap test does not expose this integration gap.

Full per-fixture timing, score, maximum gradient, and error records: [CPU examples](results/examples-cpu.json), [GPU examples](results/examples-gpu.json). Finite output is a smoke check, not an independent validation of generated chemistry or force-field accuracy.

## Performance and negative regressions

[benchmark_cif_completion.py](benchmark_cif_completion.py) compares the original and improved `_inserted()` directly. Synthetic residues contain two finite-coordinate carbon atoms and one missing carbon. Output atom annotations, coordinates (including added NaNs), and bond sets are checked for equality. Five repetitions per size follow warmup. The final run gives approximately **1.98×, 2.40×, and 3.93×** speedups for 100, 500, and 2,000 residues. At 2,000 residues, median insertion time drops from 266.93 ms to 67.98 ms. [Raw timings](results/cif-benchmark.json).

This measures only CIF atom insertion. The branch also avoids repeated CCD lookup within a read, eager construction of all representative copies, redundant hashing byte copies, and invariant per-conformer tensor allocations. No end-to-end group-packing speedup is claimed.

An initial subset of reviewer regressions was run against the diagnostic baseline, with two tests of the new helper excluded because that helper does not exist upstream: **10 failed, one passed, three skipped, two deselected**. These exercise insertion-code identity, CCD caching, numbered hydrogens, representative selection, child-chi budgeting/ownership, pose-local anchor keys, database-local tree caches, early lockstep rejection, and exclusive-end offsets. All 34 final CPU/CUDA reviewer regression cases pass on H200. Tests of the missing cyclic module validate the replacement, rather than claim to regress an upstream implementation that was not present.

## Environment and reproduction

- CPU: Python 3.12.13, Torch 2.8.0+cpu, NumPy 2.5.3, Biotite 1.7.1, RDKit 2026.3.6, OpenBabel wheel 3.1.1.22, pytest 9.1.1. Isolated environment `/mnt/home/kdidi/tmol-pr503-env`. [Core package versions](results/cpu-environment.json).
- GPU: Python 3.12.3, Torch 2.13.0+cu132, NumPy 2.1.0, Biotite 1.6.0, RDKit 2026.3.6, OpenBabel wheel 3.1.1.22, pytest 8.4.2. Apptainer image `/mnt/data/kdidi/apptainers/latent-dev-cuda13-26.06-tmol0.1.54.sif`; Torch's reported version differs from the image filename. [Core package versions](results/gpu-environment.json).
- JIT: `TMOL_USE_JIT=1`, `SPARSE_AUTO_DENSIFY=1`, `OMP_NUM_THREADS=2` (four in the first broad GPU runs), `MAX_JOBS=8` (12 for first GPU builds), `TORCH_CUDA_ARCH_LIST=9.0` on H200. Separate extension caches were used for baseline/candidate and CPU/GPU environments.
- Formatting: Black 26.3.1, target Python 3.12; flake8 on all changed/new Python files; `git diff --check`; `bash -n` on the rerun script. No native source was changed by the improvement branch.

From this branch in an environment with project, dev, and ligand dependencies:

```bash
# Run the combined reviewed selection and both available example backends.
# Existing unresolved failures produce a nonzero exit status.
bash review/pr503/run_checks.sh /path/to/results

# Rerun the insertion benchmark against a worktree at the pinned PR head.
python review/pr503/benchmark_cif_completion.py /path/to/pr503-baseline

# Reproduce the mixed-chirality ordering problem.
PYTHONPATH=. TMOL_USE_JIT=1 python review/pr503/reproduce_disulfide_order.py
```

For the shared cluster, allocate an H200 with Slurm, change to the improvement checkout, export the JIT settings above and an absolute `TORCH_EXTENSIONS_DIR`, then run:

```bash
apptainer exec --nv --bind /mnt:/mnt \
  /mnt/data/kdidi/apptainers/latent-dev-cuda13-26.06-tmol0.1.54.sif \
  bash review/pr503/run_checks.sh /absolute/path/to/results
```

The actual jobs used one GPU, 8–16 CPUs, and 96–128 GB host memory, with 2–4 hour limits for initial builds. Raw logs, JUnit XML, job scripts, diagnostic traces, and extension caches are retained at `/mnt/home/kdidi/tmol-pr503-results`. The branch commits the compact evidence needed to assess the findings; it does not commit compiled extensions or large native build logs. No comments, review, or pull request were posted to Frank's PR.
