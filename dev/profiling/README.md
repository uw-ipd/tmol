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

## H200 reference result

The 2026-08-27 reference run used one NVIDIA H200, driver 595.71.05, Python
3.12.3, PyTorch 2.11.0+cu128, CUDA runtime 12.8, the
`latent-dev-cuda13-26.06-tmol0.1.42.sif` image, AOT extensions, and runtime JIT
disabled. These are synchronized steady-state medians; setup and warm-up are
excluded.

| Workflow and input | Batch 1 (ms) | Larger batch | Batch time (ms) | Throughput gain |
|---|---:|---:|---:|---:|
| score, 15-residue protein | 0.527 | 32 | 0.557 | 30.3x |
| score, 400-residue protein | 0.743 | 4 | 1.374 | 2.2x |
| score, 24-nt DNA | 2.771 | 32 | 2.898 | 30.6x |
| score, protein–ligand | 0.809 | 8 | 2.294 | 2.8x |
| Cartesian min, 15-residue protein | 41.492 | 32 | 42.342 | 31.4x |
| Cartesian min, 24-nt DNA | 174.446 | 8 | 170.993 | 8.2x |
| torsion min, 24-nt DNA | 245.590 | 4 | 253.509 | 3.9x |
| FastRelax, 15-residue protein | 349.066 | 8 | 521.468 | 5.4x |
| FastRelax, 100-residue protein | 1238.964 | 4 | 2251.062 | 2.2x |

CUDA graphs pay off for repeated fixed layouts, especially nucleic acids. At
batch 1, graph replay changed DNA forward scoring from 2.771 to 0.863 ms (3.21x)
and forward plus coordinate gradient from 7.392 to 1.676 ms (4.41x). The RNA
gains were 2.94x and 3.99x. A 15-residue protein gained 1.28x and 1.82x, while a
protein–ligand case gained only 1.08x for both forward and gradient because its
larger kernels were already device-bound. Graph construction remains a one-time
setup cost.

### Optimization and trace conclusions

The nucleic-acid term previously rebuilt immutable atom indices, masks, base
identities, ring indices, and predecessor links on every score evaluation.
Caching them on `PoseStack` removed 67 launches per forward call. A sequential
same-H200 A/B across DNA, RNA, and protein–DNA reduced eager forward latency by
11–19%, eager forward plus gradient by 4–6%, and graph replay by about 11%.
The final DNA trace has 339 launches versus 406 before the change and a 5.24 ms
profiled wall time versus 6.11 ms. The cache adds less than 0.5 MiB in the
largest measured NA batch and did not regress protein-only setup or scoring.

The full trace matrix points to three distinct regimes:

- Small protein eager scores are launch-sensitive; larger protein and
  protein–ligand batches spend most device time in `lk_ball` and are
  compute-bound. Batch similarly sized poses for throughput.
- Eager NA scoring and gradients remain host/launch-bound: only 12–30% of the
  profiled wall time is GPU activity, with 339 forward and 823–829 gradient
  launches. CUDA graphs raise GPU occupancy to roughly 70–80% of the profiled
  wall time without changing the fixed-layout public API.
- NA Cartesian and torsion minimization still enqueue about 19k–28k kernels in
  ten LBFGS iterations. FastRelax shifts from host-sensitive at a tiny protein
  to 75–85% GPU activity for 100 residues; simulated annealing is its dominant
  kernel. Further gains there require larger fusion or optimizer/packer changes,
  not another small Python cleanup, and should preserve minimization trajectory
  semantics.

Treat the regular matrix as the timing source. Nsight interception materially
inflates host-heavy workflows; use its traces to explain time, not to publish
latency. `analyze.py` produces the complete 83-row table and SVG rather than
embedding a stale full matrix here.

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

Nsight's default graph-level mode keeps complete matrix captures fast and
compact. Use the matching eager case to inspect the kernels inside a scorer.
For a focused CUDA-graph investigation, node-level tracing is available but can
add minutes of profiler overhead to one replay with Nsight Systems 2026.3.1:

```bash
python dev/profiling/matrix.py --trace --graph-node-trace \
  --output-dir artifacts/graph-nodes --case score_graph-dna24-b1
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
wall time, NVTX enqueue-range time, kernel and CUDA-graph device time, ordinary
and graph launch counts/API time, largest cumulative NVTX range, and dominant
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
