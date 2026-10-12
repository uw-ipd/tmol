# JAX frontend (experimental)

`tmol-jax` is an optional, pure JAX frontend maintained alongside TMol's
PyTorch implementation. Its runtime imports `tmol_jax` and does not require
PyTorch. This implementation is under development: numerical parity,
performance, relaxation and repacking are not yet release validated.

The initial score preset implements the protein beta2016 channels. Nucleic-acid
torsion and metal-coordination terms are not implemented. Noncanonical residues,
ligands, and their generic bonded terms fail explicitly. The portable
pair-scoring path uses dense arrays; the optional CUDA wheel uses bounded-memory
native pair sums. Both recompute distances from the supplied coordinates on
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

An optional CUDA kernel wheel accelerates float32 LJ/LK, electrostatics, LK-ball,
and hydrogen-bond scoring while leaving the default wheel and sdist pure JAX:

```sh
python scripts/build_jax_cuda.py --outdir dist/jax-cuda
pip install dist/jax-cuda/tmol_jax_cuda-*.whl
```

Building this local platform wheel requires `nvcc`; it compiles against the
canonical TMol scalar LJ/LK, electrostatics, LK-ball, and hydrogen-bond headers
and the installed JAX FFI headers. Set
`--cuda-architectures` to a semicolon-separated SM list when the wheel must run
on GPUs other than the build host. The wheel is version-pinned to the matching
`tmol-jax` package. It contains only the compiled adapter, not a second copy of
the parameter database or TMol native sources.

If the optional wheel is absent or cannot load, scoring uses the pure-JAX path.
The CUDA path currently supports float32 only. LJ/LK and electrostatics
reevaluate every exact count-pair-qualified atom pair, LK-ball reevaluates its
exact topology-selected polar x heavy-atom pairs, and hydrogen bonding reevaluates
the rendered donor x acceptor pairs with their exact bond-separation rule and
current distance cutoff. All use the current coordinates on every call.
Coordinate minimization therefore does not use a
frozen neighbor list.

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

The minimizer renders the scorer once and keeps topology fixed. A movement
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

The default pair scoring path uses dense arrays, although LK-ball selects only
polar × heavy atoms from fixed topology. A matched H200/float32 ABBA comparison used the
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

The optional CUDA wheel replaces the LJ/LK, electrostatics, LK-ball, and
hydrogen-bond pair sums with thin JAX FFI adapters to shared, Torch-independent
TMol scalar equations. When LJ/LK and electrostatics are both active, one
adapter shares their exact current-coordinate pair traversal while retaining
their distinct count-pair matrices and four independent score channels.
Installed-wheel runs under the same protocol measured:

| workload | inference (ms) | value + gradient (ms) | resident MiB (inference / gradient) |
| --- | ---: | ---: | ---: |
| 1UBQ | 0.197–0.198 / 0.401–0.421 | 0.438–0.440 / 0.837–0.939 | 1,182 / 1,290 vs 702 / 702 |
| homogeneous 1UBQ | 0.267–0.271 / 0.400–0.410 | 0.551–0.576 / 0.846–0.911 | 940 / 1,062 vs 702 / 704 |
| mixed proteins | 0.348–0.356 / 0.452–0.454 | 0.714–0.741 / 0.909–0.934 | 952 / 1,086 vs 702 / 704 |
| 488-residue protein | 1.181–1.190 / 0.651–0.674 | 2.181–2.188 / 0.863–0.894 | 1,446 / 1,682 vs 784 / 786 |

This preserves the single-pose win and is a material improvement over dense JAX.
Relative to otherwise identical separate LJ/LK and electrostatics adapters,
fusion reduces the large-protein time from 1.286–1.288 to 1.181–1.190 ms for
inference and from 2.342–2.345 to 2.181–2.188 ms for value-and-gradient. Small,
homogeneous, and mixed workloads beat Torch for both measured operations,
although their process residency is still higher. The large case still trails
Torch by about 1.8x for inference and 2.4–2.5x for value-and-gradient.

An isolated fused-versus-separate pair comparison used arbitrary cotangents for
all four channels. Maximum/RMS Cartesian-gradient deltas were
7.63e-6/2.49e-7 on 1UBQ and 1.34e-5/2.09e-7 on the large protein. Maximum raw
channel energy deltas were 9.16e-5 and 0.00171, respectively, from float32
parallel reduction order. Fusion reduced isolated large forward time from
0.961 to 0.880 ms and value-and-gradient from 1.430 to 1.263 ms; its gradient
compiler temporary estimate fell from 98.4 KB to 7.6 KB.

An isolated large-protein electrostatics comparison found identical float32
energies and a 1.46e-5 / 1.18e-6 maximum / RMS gradient delta between the
portable and CUDA paths. Forward time was neutral at 0.311 ms, while
value-and-gradient fell from 1.132 ms to 0.583 ms and its XLA temporary estimate
fell from 743.2 MB to 7.6 KB.

An isolated hydrogen-bond comparison produced identical float32 energies for
both 1UBQ and the large protein. Maximum/RMS gradient deltas between the
portable and CUDA paths were 1.91e-5/8.70e-7 and 2.20e-4/3.32e-6,
respectively. Large hydrogen-bond forward time was neutral at 0.119 ms, while
value-and-gradient fell from 0.948 ms to 0.123 ms and its XLA temporary estimate
fell from 198.4 MB to zero.

For 1UBQ, the full-score Cartesian-gradient maximum/RMS error against Torch is
0.0129 / 0.00104. For the large protein it is 0.0317 / 0.00193. The previous
large raw LK-ball discrepancies of about 0.23 and 0.13 score units fall to
0.0034 and 0.0011 with the shared scalar kernel. The adapter evaluates the same
topology-selected pairs and potentials, but parallel float32 atomic reduction
changes accumulation order; it is not bitwise reproducible across processes.

With all four optional pair adapters, XLA temporary-buffer estimates range from
0.1 MB to 0.8 MB for inference and 0.7 MB to 5.4 MB for value-and-gradient;
these compiler estimates are not device peaks. Exact canonical count-pair
separations are clamped to 0–6 and stored as `uint8`, instead of `int32`, in the
persistent dense matrices and CUDA adapter inputs. On the 7,561-atom workload
this reduces each matrix from 228.7 MB to 57.2 MB and lowers process residency
from 2,222 / 2,786 MiB to 1,446 / 1,682 MiB for inference / gradient. This
remains above Torch's 784 / 786 MiB. Torch per-operation allocator increments
are also not comparable to total process residency. Setup plus rendering
remains 36–38 seconds for the measured JAX processes. Broader chemistry
validation, kinematic workflows, and repacking remain unfinished.

PDB loading uses the first model and Biotite's first alternate conformation.
The Biotite frontend also accepts an AtomArrayStack for same-topology batches.
Polymer connectivity follows chain labels; coordinate gaps do not automatically
create chain breaks. Prepare chain boundaries explicitly in the input.
