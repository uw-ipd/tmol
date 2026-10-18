# Where the mixed relax cycle stands, and what is worth doing next

Supersedes the four-step plan. All four were executed or resolved; see the
"Four-step optimization pass" section of MPS_PORT_NOTES.md for the full record
including what was measured and reverted.

## State

- 100 poses: host 65442 ms, mixed 26514 ms, **2.47x** (was 1.72x).
- 300 poses: host 191563, mixed 77106, **2.48x**.
- Mixed stages at 100: calcRPEs 15230, build IG 3090, run SA 3840,
  minimization 3514, build rots 790.
- All 8 Metal shaders compile at MSL 3.2. Full suite green.

The ratio understates the session: the build-IG fix is in shared code, so the
host baseline improved too. The mixed cycle itself went 39674 -> 26514 ms.

## The one thing that changed the ranking

calcRPEs is now **device-bound**. The host-only rotamer terms run underneath
the device half rather than after it, so host work there is nearly free:
dropping all four of them moves calcRPEs by only 1290 ms. Anything that adds
device work to the 11.8 s rotamer-kernel pole makes the stage slower even if
it removes host work.

Re-measure this before acting on any of the below -- it is the assumption they
all rest on.

## Ranked

0. **4x is not reachable by tuning.** At 26.5 s the cycle is calcRPEs 15.2
   (device pole ~10.9, coalesce 2.4, rest ~1.9), build IG 3.1, SA 3.8,
   minimization 3.5, build rots 0.8. Reaching 16.4 s means removing about
   10 s, and there is no 10 s anywhere except the device pole plus the two
   host-only packer stages. Concretely: perfect elimination of build IG, SA
   and the coalesce together lands near 3.8x, and only then. That is the
   whole-packer-on-device project, not a tuning pass -- plan it as such.

1. **The device pole, ~10.9 s.** lk_ball 3642, ljlk 3422, elec 2668,
   hbond 2097. lk_ball's occupancy gap is fully diagnosed (MPS_PORT_NOTES.md:
   the energy function accounts for all of 1024 -> 576, split 192 between the
   water-matrix load and 256 across `V`, with no single callee responsible).
   Closing it means restructuring the evaluation in a CUDA-shared header.
   Measure a candidate with `msl_occupancy` before building it.

2. **build IG, now 3.1 s, host.** The big serial pass is fixed (see step 5 in
   MPS_PORT_NOTES.md); it runs at 4.08x on 6 threads. About 1.2 s of serial
   work remains across the other `forall` sites. Same rule applies: a body
   that folds into a shared cell must not be converted, and the two-pass
   split is the pattern to copy where one does.

3. **SA, 3.8 s, host.** A real port, not a redesign -- CUDA's annealer is
   already parallel, so backend trajectory divergence is accepted. Needs
   Philox (~40 lines, counter-based) and host-side ranking in place of
   `mgpu::segmented_sort`. A session of its own.

4. **build IG on Metal.** ~30 entry points and a device scan primitive that
   does not exist yet. Only worth starting if the goal is to move the whole
   packer onto the device, not for this stage's seconds.

## Do not

- **Do not port omega / rama / gen_torsions / cartbonded for speed.** Measured
  net negative or neutral: see step 3 in MPS_PORT_NOTES.md.
- **Do not chase the exclusive scan in `sphere_overlap.impl.hh`.** It costs
  4 ms. The sync that mattered was the count `.item()`, and it is gone.
- **Do not re-try these**, all measured negative this session: `fast_divide`
  on elec's Coulombic divide, merging/coalescing the two sparse halves on the
  GPU, an early `has_water` exit in lk_ball, widening the Metal threadgroup
  past 32 (already recorded in workgroup_lane.hh), and making
  `fast_exp`/`fast_log` the default for all terms (elec regresses).

## Measurement notes that cost time to learn

- `TMOL_MPS_TIME_KERNELS` waits on every launch, so it makes any host-side
  sync look free. Use **`TMOL_MPS_TIME_HOST`** to time host phases with
  launches left asynchronous; use the former only for relative kernel cost.
- Benchmarks need an idle machine -- never concurrent with a build or a test run.
- `msl_probe` and `msl_occupancy` MUST be passed the explicit `32`.
- Three runs, take the median; run-to-run spread at 100 poses is ~2%.

## Commands

```bash
export PATH="/Users/dimaio/miniforge3/envs/tmol/bin:$PATH"
MAX_JOBS=8 ninja -j8

python dev/mps_scratch/relax_mixed_full.py 100 20 mixed   # the benchmark
python dev/mps_scratch/relax_mixed_full.py 100 20         # host+mixed ratio
python dev/mps_scratch/rpe_mps.py <term> <n>              # rotamer correctness+time
python dev/mps_scratch/pose_check.py <term> <n>           # pose score+gradient
TMOL_MPS_TIME_KERNELS=1 ...                               # per-kernel tally
TMOL_MPS_TIME_HOST=1 ...                                  # host phases, async launches

bash dev/mps_scratch/expand.sh <x.metal.in> /tmp/x.metal
./dev/mps_scratch/msl_probe     /tmp/x.metal 32
./dev/mps_scratch/msl_occupancy /tmp/x.metal 32
```

## Standing constraints

- `MAX_JOBS=8` on every build.
- No test run over ~2 minutes between steps; full suite only at the end.
- "Relax" means all four operations, never minimization alone.
- Do not fuse different energy terms.
- Keep the CPU and CUDA implementations tidy and bit-unchanged where possible.
- Leave `LAUNCH_BOX_32_OCC(16)` occupancy targets alone.
