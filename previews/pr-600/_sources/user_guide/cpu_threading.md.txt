# CPU threading

TMol uses PyTorch's process-wide CPU thread budget. Set it before rendering
scorers or starting autograd work.

## Inspect the allocation and active budget

On Linux, process affinity is the most useful answer to “how many CPUs may this
process use?” It respects scheduler and container restrictions:

```python
import os
import torch

available_cpus = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
print(f"CPUs available to this process: {available_cpus}")
print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
```

Log both values: scheduler affinity can limit visible CPUs, and an inherited
`OMP_NUM_THREADS` can set a smaller thread budget.

## Override the budget

Set the budget before rendering TMol scorers or starting autograd work:

```python
torch.set_num_threads(8)
```

For a command-line job, set the equivalent launch-time controls:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python workflow.py
```

`MKL_NUM_THREADS` controls MKL independently and takes precedence for MKL
operations. `torch.get_num_threads()` is the authoritative value for PyTorch
intra-op work after startup. `torch.set_num_interop_threads()` controls a
different, process-wide pool and generally does not need changing for TMol; if
an application changes it, PyTorch requires doing so before inter-op work starts.

For Slurm, request CPUs and either rely on the launcher's affinity or set the
same number explicitly:

```bash
srun --cpus-per-task=8 \
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python workflow.py
```

## How TMol uses the budget

When rendering a CPU scorer, TMol uses `torch.get_num_threads()` to plan workers
and pair-traversal shards. It parallelizes score terms, poses, and pair traversal,
but uses fewer workers for small workloads. The thread budget is an upper bound.

Changes to the thread budget should happen before rendering a scorer because
the scorer's internal execution plan is selected at construction time. Render
a new scorer after changing the budget.

Packing, minimization, and FastRelax call the same scoring machinery and inherit
the same process budget. CUDA kernels do not use this CPU thread count, although
structure preparation and other host-side work still can.

## Choose a practical value

Start with all CPUs in the process affinity mask for one TMol process. Then
measure representative structures at smaller values; memory bandwidth,
workload size, and the mix of score terms can produce a plateau before every
logical CPU is useful. Prefer physical cores before adding sibling hardware
threads when topology is known.

For multiple processes or data-loader workers, divide the allocation between
them instead of giving every process the full count. For example, four worker
processes in a 32-CPU allocation should normally start near eight intra-op
threads each. Record the affinity, `torch.get_num_threads()`, pose sizes, batch
size, and process count with benchmark results so comparisons remain
reproducible.
