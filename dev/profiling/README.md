# TMol GPU workflow profiling

This directory benchmarks complete TMol workflows and produces self-contained
Nsight Systems captures for performance investigations and NVIDIA handoff. It
complements the pytest microbenchmarks in `tmol/tests`: setup and warm-up happen
before capture, while one synchronized steady-state invocation is enclosed by
case-specific NVTX ranges.

## Coverage

The checked-in matrix spans:

| Workflow | Systems | Batch coverage |
|---|---|---|
| eager and CUDA-graph scoring | 15-, 100-, and 400-residue proteins; DNA; RNA; protein–DNA; protein–ligand | one pose and a system-appropriate batch |
| eager and CUDA-graph score + coordinate gradient | the same seven systems | one pose and a system-appropriate batch |
| Cartesian minimization | all seven systems | one pose plus selected batches |
| torsion-space minimization | proteins, DNA, RNA, and protein–DNA | one pose plus selected batches |
| FastRelax | 15- and 100-residue proteins | one pose and a selected batch |

Minimization uses ten LBFGS iterations. FastRelax uses one complete default
four-stage ramp, with ten Cartesian LBFGS iterations per stage. These fixed
budgets make implementations comparable; they are not convergence claims.

## Run one case

Use an AOT build on a CUDA node for stable results:

```bash
export TMOL_USE_JIT=0
python dev/profiling/workflows.py \
  --workflow score_grad --system protein100 --batch 16 \
  --warmup 5 --iterations 30 --output artifacts/result.json
```

Valid workflow names are `score`, `score_grad`, `score_graph`,
`score_graph_grad`, `cart_min`, `kin_min`, and `fast_relax`. The runner records
the Git revision, imported TMol path and version, GPU, driver, CUDA and PyTorch
versions, container path, Slurm job, system dimensions, synchronized
system/workload setup times, steady-state timings, and allocated-memory
high-water mark. CUDA-graph capture time is part of workload setup rather than
replay timing.

## Run the matrix

Regular timings use more repetitions than trace captures:

```bash
python dev/profiling/matrix.py --output-dir artifacts/baseline
```

Select workflows or exact case slugs when iterating:

```bash
python dev/profiling/matrix.py --output-dir artifacts/baseline \
  --workflow score --workflow score_grad

python dev/profiling/matrix.py --output-dir artifacts/baseline \
  --case cart_min-dna24-b1
```

Completed cases are skipped, so the command is safe to resume. Pass `--force`
to replace them.

## Capture Nsight Systems traces

`nsys` must be available in the GPU runtime. The trace command uses the CUDA
Profiler API to exclude imports, topology construction, score-function
rendering, and warm-up from the capture:

```bash
python dev/profiling/matrix.py \
  --trace --output-dir artifacts/nsys
```

Each case produces:

- `results/<case>.json`: timing, memory, topology, and environment metadata;
- `traces/<case>.nsys-rep`: the source trace to open in Nsight Systems;
- `traces/<case>.sqlite`: an exported queryable trace database;
- `traces/<case>.stats.csv`: kernel, CUDA API, and NVTX summaries;
- `logs/<case>.log`: complete runner and profiler output;
- `results/summary.csv`: one-row-per-case overview;
- `manifest.json`: case inventory, sizes, and SHA-256 checksums.

The `.nsys-rep`, matching JSON, and manifest are the minimum NVIDIA handoff.
Keep the SQLite and CSV files for analysis without the GUI.

## Verify and summarize a run

Turn the full result bundle into a readable table, a log-scale latency plot,
and a compact Nsight summary:

```bash
python dev/profiling/analyze.py artifacts/nsys
```

The analyzer first verifies every size and SHA-256 checksum in `manifest.json`,
then writes `analysis/timings.md`, `analysis/timings.svg`, and
`analysis/trace_summary.csv`. The trace summary includes the profiled iteration
time, total GPU-kernel time, kernel-launch count and API time, and dominant
kernel. Compare regular runs from two revisions with:

```bash
python dev/profiling/analyze.py artifacts/current \
  --baseline artifacts/baseline
```

This additionally writes `analysis/comparison.csv`; negative `delta_percent`
is faster. Run regular timings and traces into separate directories because
Nsight interception intentionally changes wall time.

## Slurm and Apptainer example

Run the container itself inside the GPU allocation. Replace the two paths with
the local environment and image:

```bash
sbatch --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=16 \
  --mem=128G --time=04:00:00 --wrap="
    apptainer exec --nv <tmol.sif> bash -lc '
      source <tmol-venv>/bin/activate
      export TMOL_USE_JIT=0
      cd <tmol-checkout>
      python dev/profiling/matrix.py --trace --output-dir <artifact-dir>
    '"
```

Use an exclusive GPU when comparing revisions. Nsight interception adds host
overhead, so use the regular matrix for latency comparisons and the trace
matrix for causal analysis.

## Interpreting the traces

Start at `<case>/measure`, then inspect `<case>/iteration-0`. TMol's nested NVTX
ranges identify score-term, water-generation, kinematics, packing, and optimizer
work. The CSV reports answer three first-pass questions:

1. `cuda_gpu_kern_sum`: which kernels account for device time?
2. `cuda_api_sum`: is the workload launch- or synchronization-bound?
3. `nvtx_sum`: which TMol phase owns the elapsed time?

Record conclusions only after comparing the regular timing matrix with these
traces; a profiled wall time is not a benchmark result.
