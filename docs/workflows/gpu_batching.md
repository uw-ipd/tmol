# GPU batching

Use this compact recipe when an application needs to score many structures on
one device. The linked tutorial provides the deeper walkthrough, measurements,
and interpretation.

> - **Prerequisites:** {doc}`Quickstart </quickstart>` and a CUDA-enabled TMol
>   installation.
> - **Deep tutorial:** {doc}`02 — GPU Batching with TMol
>   </tutorial/02_gpu_batching>`.
> - **Related workflows:** {doc}`Packing </workflows/packing>` and
>   {doc}`developer benchmarking </user_guide/benchmarking>`.
> - **API reference:** {doc}`Pose </api/pose>` and {doc}`Scoring </api/score>`.
> - **Rosetta mapping:** {doc}`GPU batching and external orchestration
>   </tutorial/rosetta_crosswalk>`.

TMol gets application throughput by evaluating several structures in one
`PoseStack` on one device. Build each input pose from the same chemical
database, combine them, and render the scorer for the resulting batch layout:

```python
from time import perf_counter

import torch

from tmol.pose import PoseStackBuilder

batch = PoseStackBuilder.from_poses(poses, device)
scorer = sfxn.render_whole_pose_scoring_module(batch)

with torch.no_grad():
    scores = scorer(batch.coords)
```

`from_poses()` accepts heterogeneous sizes but pads the batch to its largest
member. Group similarly sized structures when practical, and use a score
function built from the same parameter database as poses with custom residue or
ligand types.

## Understand first-use latency

The wheel ships precompiled scoring extensions, but constructing databases,
poses, and packing metadata still performs CPU setup. Importing ``tmol`` is now
lazy at the public-API boundary: after importing Torch, the top-level import took
9--14 ms instead of 1.89 s in the H200 test environment. The relevant modules
load when their public objects are first requested, so benchmark complete
workflows rather than treating import time as eliminated work.

Across eight order-balanced processes in the same environment, median default
database construction decreased from 1.950 to 0.710 s, a cysteine-rich PPI pose
load from 6.653 to 1.446 s, and scorer rendering from 1.824 to 0.766 s. Peak host
RSS decreased from about 1.48 to 1.35 GiB; GPU peak memory was unchanged.
Serialized scoring tables are memory-mapped while loading; an isolated repeated
Dunbrack table load decreased from 24.15 to 11.94 ms without changing the
resulting objects. In an order-balanced protein-packing run, the first pack
decreased from 4.66--4.75 to 2.28--2.55 s while warmed medians stayed at about
51--53 ms. These are one-time process costs. Reuse databases, pose metadata,
rendered scorers, and packers when the input chemistry and layout permit it.

## Accelerate repeated fixed-layout scoring

For a fixed pose layout, opt into CUDA Graph capture when the scorer will be
evaluated repeatedly. Capture only the path that the workflow needs to limit
retained static buffers:

```python
inference_scorer = sfxn.render_whole_pose_scoring_module(
    batch, cuda_graph="forward"
)
scores = inference_scorer(batch.coords)

gradient_scorer = sfxn.render_whole_pose_scoring_module(
    batch, cuda_graph="forward_backward"
)
coords = batch.coords.detach().clone().requires_grad_(True)
scores = gradient_scorer(coords)
(coord_grad,) = torch.autograd.grad(scores.sum(), coords)
```

New coordinate values are allowed; shape, dtype, and device must stay fixed.
Forward-only replay reuses its output buffer, so clone a score tensor that must
survive the next scorer call.
Passing `cuda_graph=True` captures both paths for a mixed workload. Non-default
requests for unsummed or unweighted terms remain eager. Capture adds a one-time
setup cost and retains static graph buffers, so leave it off for a scorer used
only once. Keep minimization eager by default: small score and gradient rounding
differences can send a nonlinear line search to a different local minimum.

### H200 decision data

The production-wheel benchmark used Torch 2.13.0, CUDA 13.2, and one H200, with
runtime JIT disabled. Whole-scorer capture changed median batch-1 latency as
follows:

| Input | Path | Eager (ms) | Graphed (ms) | Capture (s) |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | forward | 0.627 | 0.498 | 0.010 |
| 1UBQ | forward + backward | 1.166 | 0.740 | 0.451 |
| 1S78 PPI | forward | 2.013 | 1.929 | 0.014 |
| 1S78 PPI | forward + backward | 3.207 | 3.116 | 0.404 |
| 1BNA | forward | 3.899 | 1.017 | 0.026 |
| 1BNA | forward + backward | 8.750 | 1.970 | 0.435 |

At protein batch 64, which is compute-bound, the corresponding gains were only
2.7% forward and 1.5% forward plus backward. Forward capture retained about 50
MiB for the protein/PPI fixtures and 82 MiB for 1BNA; backward capture retained
about 50 MiB and 178 MiB, respectively. Score and coordinate-gradient maximum
absolute differences were at most 7.8e-3 and 7.7e-6 in this matrix. Capturing
inside minimization remains rejected: the small numerical change can alter the
LBFGS trajectory and final local minimum.

For independent variants, batch packing is the safer high-throughput operation.
Four protein--ligand or protein--DNA mutations packed about 3--4 times faster
than four sequential calls, with the requested residue identities preserved.
FP32 Cartesian minimization can amplify sub-milliscale score differences into
different local minima; compare final energies and constraints, and split cases
when exact trajectory reproducibility is required.

The block-pair interaction reduction now sums through a broadcast mask instead
of materializing expanded boolean-index results. On an eight-complex PPI batch,
this changed median interface-score latency from 18.798 to 16.313 ms and peak
allocated memory from 5.20 to 2.72 GiB, without changing the scoring kernels.

## Time CUDA work correctly

CUDA launches are asynchronous. Warm up the rendered module, synchronize on both
sides of each timed call, and report multiple repeats:

```python
for _ in range(warmup):
    scorer(batch.coords)
torch.cuda.synchronize(device)

elapsed = []
for _ in range(repeats):
    torch.cuda.synchronize(device)
    start = perf_counter()
    scorer(batch.coords)
    torch.cuda.synchronize(device)
    elapsed.append(perf_counter() - start)
```

Measure total batch latency and throughput separately. First-call compilation
time should not be mixed into steady-state kernel timing.

## Chunk larger workloads

Batch size is limited by padding and operation-specific intermediates. Choose a
conservative chunk size, render and score one chunk, move only the needed
results off device, and then release that chunk before continuing. Monitor both
the live CUDA allocation after construction and the incremental peak during the
operation.

TMol does not provide a multi-GPU scheduler. Shard independent chunks outside
TMol, normally with one process owning one GPU. This application workflow is
separate from the {doc}`developer benchmark harness </user_guide/benchmarking>`.
