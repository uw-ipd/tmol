# Fixed-input noncanonical scoring replay

These are diagnostic inputs, not replacement score goldens. The original
`tmol/tests/data/noncanonical_scores.yaml` was unchanged when this bundle was
recorded. The later upstream commit `0f4c3bc42` updates that YAML; this bundle
remains the original fixed input/parameter/score record for comparison.

The four `.tmol` files use tmol's existing parameter export, including generated
residues, patches and scoring records. Each NPZ stores exact pose coordinates,
full ordered atom identities, atom types and block names. `raw` precedes hydrogen
optimization; `opth` records its output. `scores.json` records the generated run's
environment, commit, seed and unweighted per-term scores. `sha256.json` covers
all thirteen data files.

Generated at tmol `026475f0e`, seed **20250828**, Python 3.12.13, PyTorch 2.8.0+cpu,
NumPy 2.5.3, RDKit 2026.3.6 and OpenBabel 3.1.0. Source CIFs are the four fixtures
listed by `tmol/tests/score/test_noncanonical_scoring.py`, already in the repo.

From the tmol repository root, in the documented JIT environment:

```sh
python review/pr503/diagnose_noncanonical_scores.py \
  --replay review/pr503/fixtures/noncanonical-score-replay \
  --output /tmp/tmol-score-replay --device cpu
```

Use `--device cuda:0` inside the recorded CUDA container/Slurm allocation. Replay
loads exported parameters, checks every atom identity, applies saved coordinates,
and scores them without regenerating conformers or rerunning OptH. This separates
scoring from environment-dependent generated chemistry. It does not establish an
independently fitted potential or explain the unreproduced historical beta-peptide
LJ reference. See `../../FOLLOWUP.md` for the full audit and remaining gates.


To check a replay rather than merely record its scores:

```sh
python review/pr503/verify_noncanonical_replay.py \
  --reference review/pr503/fixtures/noncanonical-score-replay \
  --candidate /tmp/tmol-score-replay \
  --output /tmp/tmol-score-replay-check.json
```

The verifier checks all input checksums, complete case/term inventories, ordered
atom identities and types, block types, and exact saved coordinates. It exits
nonzero when a score exceeds `max(0.01, 0.0001 * abs(reference))`. Both raw and
hydrogen-optimized cases must be present for all four classes in every repeat.
These remain diagnostic numerical references for frozen inputs; the verifier
does not replace the original fresh-generation score tests or independently
validate the chemical parameter fit.
