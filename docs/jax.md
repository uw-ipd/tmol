# JAX frontend (experimental)

`tmol-jax` is an optional, pure JAX frontend maintained alongside TMol's
PyTorch implementation. Its runtime imports `tmol_jax` and does not require
PyTorch. This implementation is under development: numerical parity,
performance, relaxation and repacking are not yet release validated.

The initial score preset implements the protein beta2016 channels. Nucleic-acid
torsion and metal-coordination terms are not implemented. Noncanonical residues,
ligands, and their generic bonded terms fail explicitly. Current pair scoring
uses dense arrays; it recomputes distances from the supplied coordinates on
every call. Cartesian L-BFGS minimization and conservative no-pack Cartesian
relaxation are available; kinematic minimization and repacking are not yet
implemented. Do not use this experimental frontend as a replacement for a
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

## Minimize Cartesian coordinates

```python
from tmol_jax import run_cart_min

move_atoms = pose.atom_is_real  # or a stricter [poses, atoms] boolean mask
minimized, result = run_cart_min(
    pose, score_function, move_atoms, max_iter=100, return_result=True
)
```

The minimizer renders the dense scorer once and keeps topology fixed. A movement
mask may select whole atoms or individual xyz components; padding is always
frozen. For a pose batch, it minimizes the sum of the per-pose scores. The
input pose is unchanged. Inspect `result["converged"]` and
`result["status"]`, and validate the final structure and energy for the intended
workflow. This is Cartesian coordinate minimization, not FastRelax or repacking.

For a conservative repulsive ramp without packing, use `run_cart_relax`. It
minimizes at increasing `fa_ljrep` weights and returns the best structure
observed under the full-weight score:

```python
from tmol_jax import run_cart_relax

relaxed, diagnostics = run_cart_relax(
    pose,
    score_function,
    move_mask=pose.atom_is_real,
    constrain_mainchain=True,
    n_cycles=1,
    return_result=True,
)
```

This Cartesian workflow does not repack sidechains and is not FastRelax. The
input score configuration is unchanged. Optional mainchain constraints tether
N, CA, and C atoms to their input coordinates within the local relax objective.

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
heavy atoms from fixed topology. A matched H200/float32 ABBA comparison used the
same coordinates, topology, weights, and one CPU thread, with five warmup calls
and 30 synchronized samples. Ranges below span two passes; times are JAX / final
Torch medians and GPU residency is the process value reported by `nvidia-smi`:

| workload | atoms | inference (ms) | value + gradient (ms) | resident MiB (inference / gradient) |
| --- | ---: | ---: | ---: | ---: |
| 1UBQ | 1 × 1,231 | 0.248–0.254 / 0.415–0.424 | 0.878–0.881 / 0.854–0.899 | 1,196 / 1,322 vs 702 / 702 |
| homogeneous 1UBQ | 4 × 1,231 | 0.542–0.546 / 0.406–0.411 | 1.896–1.901 / 0.866–0.872 | 1,256 / 1,952 vs 702 / 704 |
| mixed proteins | 4 × up to 1,615 | 0.765–0.766 / 0.428–0.440 | 5.999–6.000 / 0.888–0.913 | 1,308 / 2,054 vs 702 / 704 |
| 488-residue protein | 1 × 7,561 | 3.002–3.004 / 0.641–0.648 | 11.501–11.512 / 0.858–0.861 | 3,246 / 7,908 vs 784 / 786 |

JAX wins only the small single-pose inference case. It is slower and uses more
memory on the other measured workloads, substantially so for the large protein.
JAX setup plus rendering took 35–38 seconds per fresh process and compilation
added 6–26 seconds for inference and 11–30 seconds for value-and-gradient. The
corresponding Torch setup plus rendering took 11–14 seconds.

JAX allocator lifecycle peaks ranged from 276 MB to 3.46 GB. XLA temporary
buffer estimates ranged from 12.1 MB to 572 MB for inference and 33.7 MB to
2.80 GB for value-and-gradient; these compiler estimates are not device peaks.
Torch per-operation allocator increments are also not comparable to total
process residency. The large-protein total differed by about 0.035 score unit
and needs term-level investigation. Broader chemistry validation, kinematic workflows, and repacking
remain unfinished.

PDB loading uses the first model and Biotite's first alternate conformation.
The Biotite frontend also accepts an AtomArrayStack for same-topology batches.
Polymer connectivity follows chain labels; coordinate gaps do not automatically
create chain breaks. Prepare chain boundaries explicitly in the input.
