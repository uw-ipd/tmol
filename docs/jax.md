# JAX frontend (experimental)

`tmol-jax` is an optional, pure JAX frontend maintained alongside TMol's
PyTorch implementation. Its runtime imports `tmol_jax` and does not require
PyTorch. This implementation is under development: numerical parity,
performance, relaxation and repacking are not yet release validated.

The initial score preset implements the protein beta2016 channels. Nucleic-acid
torsion and metal-coordination terms are not implemented. Noncanonical residues, ligands and their generic bonded
terms fail explicitly. Current pair scoring uses dense arrays; it recomputes
distances from the supplied coordinates on every call. Relaxation and repacking
are not yet implemented. Do not use this experimental frontend as a replacement for a
validated production TMol calculation yet.

## Build and install

From a TMol developer environment with `tmol`, `numpy` and `build` installed:

```sh
python scripts/build_jax.py --outdir dist/jax
pip install dist/jax/tmol_jax-*.whl
```

The builder takes the version from the repository's root `pyproject.toml` and
converts the canonical parameter tables to NumPy archives. It creates both a
wheel and a standalone sdist; rebuilding that sdist needs no PyTorch. No
parameter database is copied into a second source directory.

The default JAX installation runs on CPU. For NVIDIA GPUs, install the matching
JAX runtime, for example `pip install 'jax[cuda13]>=0.11.2'`. The JAX distribution
also declares `cu12` and `cu13` extras. Current development validation uses
Python 3.12 and JAX 0.11.2; broader version support has not yet been tested.

## Score and differentiate

```python
import jax
from tmol_jax import pose_stack_from_pdb, beta2016_score_function

pose = pose_stack_from_pdb("input.pdb")
score_function = beta2016_score_function()
score = score_function.render_whole_pose_scoring_module(pose)
energies = jax.jit(score)(pose.coords)
gradient = jax.jit(jax.grad(lambda coordinates: score(coordinates).sum()))(pose.coords)
```

A rendered scorer snapshots its weights, term options and topology. Reuse it
for coordinate changes; render again after changing weights, types, connectivity
or batch layout. Use `sum_terms=False` to return channels in `ScoreType` order,
and `apply_weights=False` to inspect their unweighted energies.

The frontend does not alter global JAX precision settings. Synchronize returned
arrays with `block_until_ready()` when timing work. Record compilation and setup
separately from repeated scoring, and include setup in end-to-end comparisons.

For source development, prepare tables with
`python scripts/build_jax.py --export-data .jax-work/data`, then set
`TMOL_JAX_DATABASE` to that absolute directory and import `tmol_jax` from the
repository. Temporary tables and benchmark artifacts belong outside commits.

The initial JAX implementation is adapted from
[tudorcotet/jtmol](https://github.com/tudorcotet/jtmol), under Apache-2.0;
see `tmol_jax/NOTICE` for the source revision.

## Current validation and limitations

Development checks cover CPU and CUDA scoring/gradients, all canonical hydrogen
bond polynomial tables, isolated installation without Torch, and small protein
I/O. A three-residue fragment and 1UBQ reconstruct exactly the same coordinates
as current Torch TMol. Their weighted float32 energies agree to about 1e-4 score
units. This is limited evidence, not full biochemical parity.

Torsions use an atan2 angle rather than the native acos angle. Near planar
torsions, native float32 gradients can lose precision: the tested fragment has
a maximum raw Cartesian-torsion gradient difference of 0.026. Both float64
implementations agree within 1e-10 there, and centered finite differences agree
with the JAX result.

Pair scoring still uses dense arrays, although LK-ball selects only polar ×
heavy atoms from fixed topology. An H200/float32/1UBQ comparison with five warmup
calls and 30 synchronized samples measured median inference at 0.244 ms for
JAX versus 0.408 ms for final Torch, and value-and-gradient at 0.901 ms versus
0.877 ms. This single workload does not establish general speed parity.

XLA estimated 12.1 MB of temporary buffers for inference and 33.7 MB for
value-and-gradient. These are compiler estimates, not measured device peaks;
memory remains substantially worse than native TMol. Setup took about 25
seconds, with compilation adding further startup cost. Larger and mixed batches,
disulfide validation, relaxation, and repacking remain unfinished.

PDB loading uses the first model and Biotite's first alternate conformation.
The Biotite frontend also accepts an AtomArrayStack for same-topology batches.
Polymer connectivity follows chain labels; coordinate gaps do not automatically
create chain breaks. Prepare chain boundaries explicitly in the input.
