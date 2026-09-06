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

For repeated default weighted scoring with a fixed coordinate shape, dtype,
and device, capture the launches once and replay them:

```python
graphed_scorer = sfxn.render_whole_pose_scoring_module(
    batch, cuda_graph="forward"
)
with torch.no_grad():
    scores = graphed_scorer(batch.coords)
```

Forward-only replay reuses its output buffer. Clone `scores` when it must remain
unchanged across the next call. CUDA graphs mainly reduce launch overhead; they
do not replace batching or improve kernels that already dominate runtime.

## Chunk larger workloads

Batch size is limited by padding and operation-specific intermediates. Choose a
conservative chunk size, render and score one chunk, move only the needed
results off device, and then release that chunk before continuing. Monitor both
the live CUDA allocation after construction and the incremental peak during the
operation.

TMol does not provide a multi-GPU scheduler. Shard independent chunks outside
TMol, normally with one process owning one GPU. This application workflow is
separate from the {doc}`developer benchmark harness </user_guide/benchmarking>`.
