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

## Accelerate repeated nucleic-acid scoring

DNA and RNA torsion scoring consists of many small tensor operations. For a
fixed pose layout, opt into CUDA Graph capture when the scorer will be evaluated
repeatedly:

```python
scorer = sfxn.render_whole_pose_scoring_module(batch, cuda_graph=True)

# New coordinate values are allowed; shape, dtype, and device must stay fixed.
scores = scorer(coords)
scores.sum().backward()  # gradients remain supported
```

Capture adds a one-time setup cost and retains static graph buffers, so leave it
off for a scorer used only once. The option captures the pure-PyTorch
nucleic-acid torsion and carbohydrate `sugar_bb` terms when present; TMol's C++
score operators remain eager. Keep minimization eager: small score and gradient
rounding differences can send a nonlinear line search to a different local
minimum.

### H200 decision data

The production-wheel benchmark used Torch 2.13.0, CUDA 13.2, and one H200, with
runtime JIT disabled. For 1BNA, graphing the nucleic-acid term reduced median
whole-score latency from 3.936 to 1.188 ms forward and from 8.754 to 2.240 ms
forward plus backward at batch 1. At batch 64, the corresponding measurements
were 4.585 to 2.265 ms and 11.766 to 6.009 ms. Output and coordinate-gradient
maximum absolute differences were at most 1.7e-4 and 7.7e-6 in this matrix.

A cold-process capture took about 3.0 seconds, so this option pays off after
roughly 500 repeated forward-plus-backward evaluations or 1,100 forward-only
evaluations at batch 1. It is not a faster first call. Capturing inside
minimization was rejected even though it ran faster: the small numerical change
altered the LBFGS trajectory and final local minimum.

On the ten-sugar 4BYH fixture, graphing `sugar_bb` reduced full carbohydrate
forward latency from 2.094 to 1.441 ms and forward plus backward from 4.778 to
2.018 ms at batch 1. Capture took 0.172 seconds after normal scorer warmup;
output and gradient maximum absolute differences were 1.6e-5 and 4.8e-7.

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
