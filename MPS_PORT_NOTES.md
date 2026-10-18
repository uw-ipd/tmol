# tmol MPS (Apple Metal) port — working notes

Status as of 2026-10-15. Uncommitted working tree: 122 files changed
(+1610/−3165), 33 new files. Host build clean, full score suite green,
`beta2016` on 100×1UBQ unchanged at **E = 26099.805** (pre-refactor 26099.789;
the ~6e-7 relative drift is float32 reduction-order/FP-contraction noise).

---

## 1. Why: the profiling that motivates this

Measured on Apple M2 Pro (6 perf + 4 efficiency cores, 16 GB), torch 2.14.1,
100 × 1UBQ = 123,100 atoms, full torsion relax via `run_kin_min`.

**Total 61.8 s wall, 319 score evaluations.**

| bucket | time | share |
|---|---:|---:|
| **Scoring** | 53.9 s | **87%** |
| Kinematics (`forward_kin_op` fwd+bwd) | 3.2 s | 5% |
| L-BFGS (`bmm`, `solve_triangular`, copies) | ~3.5 s | 6% |

Per-term, measured with **terms run serially** (`_MAX_CPU_SCORE_TERM_WORKERS=1`,
which takes the `cpu_workers < 2` path and bypasses the thread pool, making
cProfile attribution exact — 100% of forward wall time attributed):

fwd+bwd, 4 threads, 100×1UBQ, kernels total 243.5 ms:

| op | ms | % |
|---|---:|---:|
| `elec_pose_scores` | 102.6 | **42.2** |
| `ljlk_pose_scores` | 82.0 | **33.7** |
| `lk_ball_pose_score` | 16.4 | 6.7 |
| `cartbonded_pose_scores` | 15.5 | 6.4 |
| `hbond_pose_scores` | 8.4 | 3.4 |
| `genbonded_pose_scores` | 5.1 | 2.1 |
| `build_compact_block_neighbors` | 4.7 | 1.9 |
| `gen_pose_waters` | 3.4 | 1.4 |
| `dunbrack_pose_scores` | 2.4 | 1.0 |
| `gen_hbond_bases` | 1.7 | 0.7 |
| `backbone_torsion_pose_score` | 1.3 | 0.5 |

**elec + ljlk = 76% of scoring.** Bonded terms total ~14%.

### Measurement traps hit (do not repeat)

- **cProfile under the thread pool is wrong.** `_ParallelScoreTerms.forward`
  submits every term to a `ThreadPoolExecutor`, and cProfile only instruments
  the calling thread, so it times wall-clock-under-contention. An early table
  claimed cartbonded 21.3% / hbond 14.2% as the top two; isolated, cartbonded
  is 11.8 ms not 29.8 ms. **Always force serial terms before attributing.**
- **Leave-one-out is also wrong** for per-term cost: it measures critical-path
  displacement. It said lk_ball 23.2% (really 6.7%) and ljlk 15.0% (really
  33.7%). It *did* get the bonded/nonbonded split right (15% vs 85%), which
  the direct serial attribution confirms at 14.1%.
- **Single-run A/B is noise.** An early claim of "12.6% regression" from the
  step-2 refactor collapsed to "2–8%, noise-limited" with 3 runs per side.

### Transfer cost is NOT a blocker

```
N= 100  1.48 MB each way   CPU<->MPS round trip  0.821 ms
N= 200  2.95 MB each way                         0.778 ms
```
0.82 ms against a 173 ms scoring step = 0.5%, and flat in N. Unified memory.
An earlier concern that round-trips would eat the win is **disproven**.

### Amdahl

Scoring is 87% → ceiling 7.7×. Nonbonded (elec+ljlk+lk_ball+waters+neighbors)
= 85.9% of kernel time. If only elec+ljlk are ported (76% of scoring):

| if those terms get | overall relax speedup |
|---|---|
| 3× | ~2.2× |
| 5× | ~3.0× |
| ∞ | ~5.9× |

---

## 2. The three-step plan

1. **Portable math layer** — replace Eigen + C++ stdlib in device-side math so
   the same headers compile for host, nvcc, and Apple's Metal compiler.
   **DONE.**
2. **Lambda → context struct** — kernel bodies are `[=]` device lambdas, which
   Metal cannot use as entry points. Bodies become function templates taking an
   explicit context. **DONE for elec.**
3. **MPS backend** — `Device::MPS`, `DeviceOperations<MPS>`, `.metal` entry,
   host dispatch. **IN PROGRESS.**

Kinematics deliberately stays on Eigen (4×4 transform algebra, not an MPS
target). Portable `Mat` is layout-identical to Eigen so both can view the same
tensors.

---

## 3. Step 1 — portable layer (COMPLETE)

### New: `tmol/utility/portable/`

| header | contents |
|---|---|
| `qualifiers.hh` | `TMOL_HD` (= `__host__ __device__ __forceinline__` under nvcc, `inline` otherwise), `TMOL_PORTABLE_METAL` |
| `math.hh` | 16 scalar fns dispatching to `std::` or `metal::`; `pi<T>()` replaces `EIGEN_PI` |
| `matrix.hh` | `Mat<Real,R,C>` column-major, `Vec<Real,N> = Mat<Real,N,1>`, row/col proxies |
| `tuple.hh` | fixed-arity tuple, `get`/`make_tuple`/`tie`, portable `index_sequence` |
| `reduce_ops.hh` | `plus_t`/`maximum_t`/`minimum_t`, `enum class scan_type` |
| `device.hh` | `enum class Device { CPU, CUDA, MPS }`, `PtrTag` |
| `metal_view.hh` | `MetalView<T,N>` — same 3-pointer shape as `TView`, address-space qualified |
| `eigen_interop.hh` | `from_eigen`/`to_eigen` + `sizeof` assertions |
| `pybind.h` | casters for `portable::Mat` and `portable::tuple` |

### Key design decisions

- **`Mat` is an aggregate** (no user-provided ctors) → trivially copyable,
  layout-compatible with Eigen, `Real3({0,0,0})` still parses.
- **Column-major**, matching Eigen's default, so `TView` reinterpretation of
  torch storage is byte-identical.
- **`row()`/`col()` return `const` prvalues.** This is load-bearing — see
  "Bugs found" below.
- **Eigen over-aligns**: `Eigen::Matrix<float,4,1>` has `alignof` 16, not 4.
  `sizeof` matches (so `TView` strides are fine) but a struct embedding one may
  pack differently. Alignment is deliberately *not* asserted.

### Enabling moves (these are what made per-term migration possible)

Shared helpers became **generic over the vector/view type with an Eigen or
portable default**, so unmigrated callers compile untouched:

- `distance<Real>::V/V_dV` — templated on the 3-vector type (`geom.hh`)
- `compute_rot_spheres` / `compute_block_spheres` — `CoordTView` deduced
  (`sphere_overlap.impl.hh`)
- `coord_from_shared` — vector type is a parameter, defaults to portable
  (`data_loading.hh`)
- `resolve_local_atom_ind_from_uaid` — `ConnTView` deduced (`uaid_util.hh`)

### Decoupling that had to happen at the source

- **`TensorUtil.h` → Eigen.** Its `enable_tensor_view<Eigen::Matrix>`
  specialization made the *whole tensor layer* pull in Eigen. Moved to new
  **`TensorUtilEigen.h`**, included only by kinematics, bspline, and pybind.
- **`Device`/`PtrTag` lived in `TensorAccessor.h`.** Device-templated but
  view-free machinery was dragging in torch just for the enum. Moved to
  `portable/device.hh`.
- **12 params headers split**: `params_types.hh` (plain POD structs, portable)
  + `params.hh` (keeps `enable_tensor_view`, includes the types header).
  **Zero existing include sites changed** — this was the user's chosen option
  over `#if` guards or renaming.
- **`bspline.hh` split**: `bspline_coeffs.hh` holds host-only coefficient
  construction (needs `Eigen::Dynamic` scratch buffers); `bspline.hh` keeps
  interpolation (portable). Only `bspline.pybind.cpp` changed, calling
  `ndspline_coeffs<...>` instead of `ndspline<...>`. `computeCoeffs` was only
  ever instantiated at `Device::CPU` — never device code.
- **39 stale `#include <Eigen/...>`** removed; 9 stale `<cmath>`; 4 unused
  pybind includes in math headers.

### `TMOL_PORTABLE_ONLY` was introduced and then DELETED

A preprocessor escape hatch for excluding host-only code. User preferred no
hatch. Eliminated by splitting `bspline.hh` by file instead, consistent with
`params_types.hh` and `TensorUtilEigen.h`. **There are now zero preprocessor
escape hatches** — portability is structural, decided by which header you
include. Don't reintroduce one.

### The guard (`tmol/tests/utility/portable/`)

Two tiers + negative control, runs in ~2 s:

1. `portable_cone.cc` — 9 headers that compile against **nothing but the
   portable layer** (no Eigen, no torch, no std containers).
2. `portable_cone_tensor.cc` — 14 headers that still use `TView` but are
   **Eigen-free**.
3. `test_guard_trips_when_eigen_is_reintroduced` — proves the guard can fail.
4. `test_portable_matrix_matches_eigen` — byte-layout exact, arithmetic to 1e-12.

Mechanism: a poison `Eigen/Core` that `#error`s, placed ahead of the real one
on the include path.

### Numerical equivalence (characterised, accepted)

Elementwise ops (`+ - * / cross`, unary `-`): **bit-exact**.
Reductions (`dot`, `norm`, `squaredNorm`, `normalized`): differ at epsilon.

| | max abs diff f32 | f64 |
|---|---:|---:|
| elementwise | 0 | 0 |
| reductions | 1.9e-06 | 3.6e-15 |

Two causes, both isolated: **FP contraction** (99.8% of it — `-ffp-contract=off`
took 68k mismatches to 114) and **association order** (Eigen's redux computes
`a₀ + (a₁ + a₂)`; a loop computes `(a₀ + a₁) + a₂`). User decided bit-exactness
is **not** required.

---

## 4. Step 2 — lambda → context (COMPLETE for elec)

### 2a. Portable reduction vocabulary

`mgpu::plus_t` (94 uses), `scan_type_exc/inc` (63), `maximum_t`/`minimum_t`
appeared **inside kernel bodies** and in the `DeviceOperations` interface — even
the CPU path templated on `mgpu::scan_type_t`. Converted at ~180 sites across
24 files. CUDA translates at its own boundary via `mgpu_scan_type()`.

**Result: moderngpu headers reaching the elec CPU TU went 3 → 0.**

All `using namespace mgpu` eliminated (4 files). Worst offender was
`LAUNCH_BOX_32`, which injected the directive into *every scope that launches
work*.

> **Trap:** an automated "qualify mgpu names" pass wrecked `kernel_segscan.cuh`
> — it qualified template parameters (`type_t`, `op_t`), local typedefs
> (`storage_t`, `reduce_t`), member names (`storage.reduce`) and comment text.
> Reverted and redone with the file's own declarations excluded. **Any such
> pass must exclude template params, local decls, members, and comments.**

### 2b. Bodies extracted

**Technique that worked: let the compiler enumerate the captured set.**
Temporarily replace `[=]` with `[]`, compile with `-ferror-limit=0`, and grep
`variable 'X' cannot be implicitly captured`. Exact, no guessing.

Result: **24 fields**, where the dispatch entry points take **94 tensors**.

- `elec_ctx.hh` — `ElecCtxT`, templated on a view template-template param, so
  the field list exists **once** for both TView and MetalView. Parameterized on
  `OutRank` because the rotamer dispatch writes rank-2 energy tensors where
  pose scoring writes rank-4.
- `elec_ctx_tensor.hh` — binds it to `TView` for host/CUDA via `TViewBinder`.
- Shared-memory union hoisted to namespace scope (`ElecSharedMem`) and passed
  by reference — **Metal only permits `threadgroup` declarations at kernel
  scope**.
- 5 bodies → `TMOL_HD void elec_eval_*(Ctx const& ctx, Shared& shared, int cta)`.

### Performance cost (measured, accepted)

Isolated elec fwd+bwd, 100×1UBQ, 3 runs each side:

```
BEFORE (lambdas):  medians 76.3 / 81.9 / 82.6 ms
AFTER  (ElecCtx):  medians 83.5 / 84.8 / 83.7 ms
```

**2–8%, noise-limited.** User said acceptable.

Cause chased down: **not** inlining (`TMOL_HD` changed nothing), **not**
by-value vs by-reference. It was the **inner helper lambdas** — with `[=]` each
copied the whole 24-field context where it used to capture 2–3 views.
`[=, &ctx]` took it from 91 ms back to ~83 ms. Residual is context indirection.

---

## 5. Step 3 — MPS backend (IN PROGRESS)

### Toolchain finding — IMPORTANT

**The offline Metal toolchain is NOT installed.** `xcrun metal` is a stub:
`missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain`.

**Runtime compilation works and is the chosen path.** Verified end-to-end:

```
device: Apple M2 Pro
runtime MSL compile: OK  (maxThreadsPerThreadgroup=1024, simdWidth=32)
result check: out[10]=30.0  PASS
```

So: compile the shader at load time via `newLibraryWithSource`, **no `.metallib`
CMake lane needed**, no multi-GB download. simdgroup width is **32**, same as a
CUDA warp, so `LAUNCH_BOX_32` tiling maps across directly.

**Runtime compilation cannot resolve filesystem `#include`s.** Working method:
pre-expand with clang, then prepend Metal's own header:

```bash
# stub dir with empty metal_stdlib, metal_math, metal_atomic, metal_simdgroup
clang++ -E -x c++ -std=c++17 -D__METAL_VERSION__=320 \
  -I<stubs> -I. -Itmol/extern file.metal.in | grep -v '^#' > out.metal
printf '#include <metal_stdlib>\nusing namespace metal;\n' | cat - out.metal > full.metal
```
Probe harness: `scratchpad/msl_probe.mm` (compile with
`clang++ -ObjC++ -fobjc-arc -framework Foundation -framework Metal`).

### Done

- `Device::MPS` added to the enum; whole tree rebuilds clean.
- `MetalView<T,N>` written (same 3-pointer layout as `TView`).
- **Shared kernel machinery freestanding: 7/7** — `tile_atom_pair_evaluation`,
  `accumulate`, `count_pair`, `warp_segreduce`, `warp_stride_reduce`,
  `data_loading`, `diamond_macros`. They had **zero** tensor-view uses between
  them; they were dragging in the tensor layer for the `Device` enum and stale
  includes.
- **`elec.hh` freestanding** — its 25 `TView` params became deduced view types.
  Cascade: `Int`/`Real`/`D` stopped being deducible, so **13 call sites** needed
  explicit template args — and **parameter order differs per helper**
  (`<…,nt,Int,Real>` vs `<…,nt,TILE_SIZE,Real,Int>`). Extract each signature's
  real order; don't assume.
- **`elec_kernels.hh` — 1054 lines, compiles standalone** with poison Eigen and
  no torch. This is the Metal-compilable artifact.
  - Deliberately does **not** include `device_operations.hh`: bodies reach
    primitives through their `DeviceDispatch` template param, resolved at
    instantiation. The interface declares two `TPack`-returning host helpers
    and so needs torch.
- `device_operations.mps.impl.hh` written — only **4 primitives** are needed by
  elec's whole chain: `synchronize_workgroup` (16 uses),
  `for_each_in_workgroup` (6), `reduce_in_workgroup` (2),
  `copy_contiguous_data` (2).
- **PROVEN: tmol's portable math and elec's real potential compile as MSL.**

### MSL restrictions discovered (by actually compiling)

| finding | fix applied |
|---|---|
| derived classes unsupported | `make_index_seq` rewritten with member typedef |
| program-scope vars must be `constant` | `static constexpr int value` → enumerator |
| `<cmath>` leaking | removed from 9 headers already on the shim |
| **address space is part of a reference's type** | `constant&` will not bind to `thread const&`; kernel entries must copy params into thread space first |

That last one matters for the real entry point: every `const&` in tmol's math is
implicitly `thread const&`.

### Remaining

1. Push `elec_kernels.hh` itself through the Metal compiler (expect more
   restrictions like the four above).
2. `.metal` entry point: declare `threadgroup ElecSharedMem`, build a
   `MetalView`-backed `ElecCtxT`, call `elec_eval_energies<...>`.
3. Host side (`.mm`): buffer binding + dispatch. **Design agreed with user:**
   NOT an X-macro — the context struct is declared once templated on the view,
   with a `visit_views()` member for the host-side encode loop. Metal argument
   buffers need **GPU virtual addresses** (`MTLBuffer.gpuAddress`), not host
   pointers, so a straight `memcpy` of the context will not work.
4. Python device routing; validate vs CPU; benchmark.

### Scope decision already made

`sphere_overlap.impl.hh` (45 view uses) **stays on CPU** for the first Metal
path — it's the block-neighbor computation and measures only 1.9% of kernel
time. Hand the kernel a precomputed neighbor list.

---

## 6. Environment / workflow

- Build: `MAX_JOBS=8 TMOL_BUILD_TESTS=ON pip install -e . --no-build-isolation`
  — **always cap MAX_JOBS**, 16 GB machine thrashes at full width (it only
  *just* thrashes; 8 is enough, don't over-correct).
- Keep individual pytest runs **under ~1 minute**; full suite only at the end.
- Python: `/Users/dimaio/miniforge3/envs/tmol/bin/python` (3.12 conda env
  `tmol`). Build backend (`scikit-build-core`, `ninja`, `pybind11`, `cmake`,
  `packaging`) was installed into the env — required for
  `--no-build-isolation`, and it also fixes
  `test_indexed_sdist_metadata_retains_public_candidate_version`.
- clang-format 23.1.2 at `.../envs/tmol/bin/clang-format`.
- **No CUDA toolchain on this machine.** Everything is host-compile + CPU
  numerics only. The nvcc path is unverified by explicit user decision —
  deferred to CI.

---

## 7. Bugs found and fixed (worth remembering)

1. **Silent no-op writes.** `row()` returned by value, so `m.row(i) = v;`
   *compiled and did nothing* — lk_ball scored exactly zero on DNA because its
   water matrices were never populated. 4 sites in lk_ball, 1 in dunbrack.
   Fixed by making `row()`/`col()` return **const prvalues**, turning every such
   site into a compile error; writes go through `row_ref()`/`col_ref()`. The
   guard immediately caught the dunbrack one.
2. **pybind caster ignored the `convert` flag.** pybind resolves overloads in
   two passes; always using `forcecast` made the `float` overload greedily
   accept `double` coefficients and truncate them —
   `test_polynomial_gradcheck` failed. Casters must honor `convert`.
3. **Casters must live with their types.** `_cubic_hermite_polynomial.cpp`
   didn't include the tensor binding header → compiled fine, failed at runtime
   with "Unregistered type". Moved both casters to
   `tmol/utility/portable/pybind.h`.
4. **`astuple()` is device code, not just pybind.** `tie(...) = ....astuple()`
   runs in kernels. Portable tuple needed an element-wise **converting
   constructor** because `return {double_expr, ...}` into a float field is a
   narrowing error under aggregate init but fine through `std::tuple`'s
   converting ctor.

---

## 8. CPU threading finding (separate from MPS; user deferred)

Two independent parallelism layers that **multiply**:

1. `_ParallelScoreTerms` submits terms to a `ThreadPoolExecutor`;
   workers = `min(_MAX_CPU_SCORE_TERM_WORKERS=4, torch.get_num_threads(), n_terms)`
2. each term internally calls `at::parallel_for` → OpenMP team of
   `torch.get_num_threads()`

Default here: **4 workers × 6 OMP threads = 24 threads of demand on 10 cores.**
Scales the wrong way — demand is `4 × torch.get_num_threads()`, so a 64-core
node sees 256.

Measured (full beta2016 fwd+bwd, 100×1UBQ, torch threads 6):

| W × L | ms | cores busy | demand |
|---|---:|---:|---:|
| 1 × 6 | 195.1 | 5.39 | 6 |
| 2 × 6 | 167.4 | 7.66 | 12 |
| 3 × 6 | 163.7 | 7.94 | 18 |
| 4 × 6 (default) | 164.9 | 7.93 | 24 |

**W=4 buys nothing over W=2 at double the demand.** Changing
`_MAX_CPU_SCORE_TERM_WORKERS` 4 → 2 costs ~2% and halves thread demand
(one-line change, `_score_function.py:39`).

`OMP_THREAD_LIMIT` does **not** give a process-wide budget — it is
per-contention-group, and each executor worker is its own pthread master, so
total demand stays `W × L`. Proven: 4 workers + limit 1 → 1.99 cores busy;
1 worker + limit 1 → 1.00.

Also: the **fused ljlk+elec module is disabled** when
`device.type == "cpu" and torch.get_num_threads() > 1 and n_poses != 1`
(`_score_function.py:909-913`). Thread-count changes silently swap which
kernels run — confounds any threading A/B.

---

## 9. Open decisions / risks

- **nvcc unverified.** 122 files changed including every energy term's device
  code. `TMOL_HD` expands to `__host__ __device__ __forceinline__`; deduced
  templates now sit inside extended device lambdas, where nvcc is pickiest.
  Expect at least one round of CUDA-specific fixes in CI.
- `elec/potentials/potentials.hh` is clean but **not in the guard** — it
  includes `params_types.hh`… which is now portable, so it may now qualify.
  Re-check and add it.
- The std-container guard only checks the 4 portable headers, not the term math
  headers. Extending it would have caught the 9 stale `<cmath>` includes.
- Shared source ≠ shared performance: tile widths tuned for NVIDIA CTAs aren't
  right for Apple's 32 KB threadgroup memory limit. `LAUNCH_BOX_*` is the lever.

---

# Working MPS backend for elec (session 2)

The Metal path is end-to-end working and validated. Numbers are at the end.

## What runs on the GPU

`ElecPoseScoreDispatch<..., Device::MPS, float, int32_t>::forward`, the
whole-pose scoring path, plus the two sphere-overlap kernels it depends on.
The kernel bodies are the *same source* the CPU and CUDA backends compile --
`elec_kernels.hh` and `sphere_overlap_kernels.hh` -- instantiated with
MetalView and DeviceOperations<MPS>.

Five elec entry points are compiled (energies, energies_by_block,
energies_by_block_2, derivs, derivs_2); only the forward whole-pose path is
wired into the dispatch so far. Backward and the rotamer paths still fall to
CPU/CUDA.

## Architecture

- `tmol/score/common/mps_layout.hh` -- metadata buffer layout, shared by host
  and shader.
- `tmol/score/common/mps_launch.{hh,cpp}` -- shader cache (one library per
  source, compiled lazily), view-to-buffer binding, dispatch.
- `tmol/score/common/tview_binder.hh` -- binds a view-generic context to TView.
- `<term>_ctx.hh` / `<term>_ctx_metal.hh` -- the field list, once, with
  `visit_views` walked by *both* the host (to bind and to write shapes) and
  the shader (to rebuild the views). This is what keeps the positional buffer
  binding from drifting.
- `<term>_mps_kernels.metal.in` -- entry points only; pre-expanded at build
  time and embedded as a string.
- `cmake/TmolMetalShader.cmake` -- flatten + embed, **with a depfile**.

## The decisive findings

1. **The stable ABI has a complete MPS path.** `torch/mps.h` and all of
   `at::mps::` are `#error`'d out under `TORCH_TARGET_VERSION`, but
   `torch/csrc/inductor/aoti_torch/c/shim_mps.h` provides
   `aoti_torch_mps_create_shader_library` (runtime MSL compilation),
   `set_arg_tensor`, and `dispatch_array_with_group_size` on torch's own
   command buffer. No full-ABI operator package, no MTLBuffer bit-casting, no
   separate command queue. Symbols confirmed present in libtorch_cpu.dylib.

2. **Torch compiles shaders at MSL 3.2, not "latest".** This is the single
   biggest source of surprise. Under 3.2 every pointer and reference must name
   its address space explicitly -- `T&` is an error, not a thread-space
   reference. Under the default (Metal 4) version, `thread` is inferred and
   explicit `thread` on a member accessor *fails* to bind. The two are in
   tension, so the portable headers target 3.2 via `TMOL_THREAD`, and
   `dev/mps_scratch/msl_probe <file> 32` is the correct local check. Testing
   with the default version is misleading.

3. **An MPS tensor's data pointer is not host-readable.** It is the MTLBuffer
   object. Any host-side kernel that dereferences it reads garbage -- this is
   how the first failure showed up, as a bounds assert deep in sphere_overlap.
   It also means "run this part on the CPU for now" is not available without
   real tensor copies.

4. **The shader build needs a depfile.** `add_custom_command` depending only
   on the `.metal.in` leaves a stale shader embedded when a kernel *header*
   changes. The symptom is a runtime "shader library compilation failed" that
   contradicts a clean local probe. Fixed with `-MD -MF` + `DEPFILE`.

## MSL restrictions found (beyond session 1)

- Address space is part of a deduced type: `make_tuple(T const&)` binding a
  thread int deduces `T = thread int`, and a struct field may not be
  address-space qualified. Value parameters avoid it.
- `tuple<T&...>` cannot exist. `tie()` returns a `ref_tuple` of pointers,
  typed from `decltype(&v)`.
- A `threadgroup` variable may not have a constructor -- it becomes a guarded
  dynamic initialization and fails to link on `__cxa_guard_acquire`.
- `as_type` is a builtin, not a member of namespace `metal`.
- A member of a thread-space object has the *default* address space, which
  does not match an explicitly `thread`-qualified parameter during deduction
  at Metal 4 -- but does at 3.2. Target 3.2.

## Known shortcuts, to revisit

- A context is the union of what its family of kernels reads, so any one
  kernel leaves some views unset. `mps_launch` binds the metadata buffer as a
  placeholder in those slots to keep the positional numbering intact.
- The metadata buffer is rebuilt on the host and copied to the GPU on *every*
  launch (three launches per forward). It is pure overhead and explains why
  small workloads lose; it should be cached.
- Backward, rotamer scoring, and the compact-block-neighbor paths are
  CPU/CUDA-only.
- CUDA is still unverified -- no nvcc on this machine.

## Numbers

Apple M2 Pro, 10 cores / 16 GB, torch 2.14.1, 6 torch threads. Whole-pose
elec forward, N copies of 1UBQ, median of 5. CPU baseline is the normal
multithreaded CPU backend.

| poses | CPU (ms) | MPS (ms) | speedup | rel. energy error |
|------:|---------:|---------:|--------:|------------------:|
|    10 |     4.16 |     5.45 |   0.76x |           1.8e-07 |
|   100 |    35.62 |    12.07 |   2.95x |           7.3e-08 |
|   400 |   138.44 |    45.51 |   3.04x |           7.3e-08 |
|  1000 |   389.28 |   112.32 |   3.47x |           2.3e-07 |

The error is float32 accumulation-order noise, as expected; it does not grow
with the workload.

Small workloads lose to per-launch overhead (the metadata copy above, plus
three dispatches). The crossover is somewhere between 10 and 100 poses.

Regression state at this point: `1104 passed, 907 skipped, 0 failed` across
tmol/tests/score, tmol/tests/utility, tmol/tests/numeric.

---

# Plan: finish ljlk, hbond, lk_ball; then metadata; then benchmark

## Current state

- **Host build green.** Last full verification: `1104 passed, 907 skipped,
  0 failed`. Since then only address-space-macro edits landed, and all four
  affected TUs (elec/ljlk/lk_ball/hbond `*_pose_score.cpu.cpp`) are
  syntax-clean, so the host path is sound. Re-run a full build to be certain.
- **elec + sphere_overlap on MPS: done and validated.** 2.9-3.5x on 25+
  poses, relative energy error ~7e-8 (float accumulation order).
- **ljlk: two known errors from done.** Details below.
- **hbond, lk_ball: not started.**

## Measured baseline (elec forward, median of 7)

| poses | CPU ms | MPS ms | speedup |
|------:|-------:|-------:|--------:|
|    10 |   4.19 |   2.52 |   1.66x |
|    25 |  10.20 |   3.60 |   2.83x |
|    50 |  19.61 |   6.44 |   3.05x |
|   100 |  34.82 |  11.97 |   2.91x |
|   200 |  69.26 |  23.14 |   2.99x |

Marginal cost is linear at 0.111 ms/pose, implying ~1.4 ms fixed per-call
overhead -- ~13% at 100 poses, negligible past 400. This is why metadata
caching is scheduled last: it is a ~10% effect, while the unported terms are
worth 1.4x -> 2.0x -> 2.5x on scoring by Amdahl (elec 42%, ljlk 34%, hbond
~14%, lk_ball remainder).

## STATUS UPDATE (after the compaction): ljlk is DONE

Both errors were as diagnosed. `ljlk_load_tile_invariant_intrares_data` got
the trailing `WorkgroupLane<D> lane = {}` (it never uses it -- it reaches no
workgroup primitive -- but its call site is generated by the same macro as
its siblings'), and `lane` is now passed at both `tile_evaluate_rot_pair`
call sites in ljlk_kernels.hh.

Then the wiring, following the elec template exactly:

- `ljlk_pose_score.impl.hh`: the two inline lambda bodies (265 lines) were
  replaced by a `LJLKCtx<Real, Int, D> ctx` plus two four-line lambdas that
  call `ljlk_eval_energies_by_block` / `ljlk_eval_energies` from
  ljlk_kernels.hh. CPU, CUDA and Metal now run one copy of the source.
  MAX_N_CONN and TILE_SIZE come from ljlk_kernels.hh rather than being
  defined twice.
- The MPS launch branch is tested *first*, ahead of the compact-block-
  neighbor paths, for the same reason as in elec: those launch through
  DeviceOperations, which on this backend runs the kernel on the CPU.
- `build_compact_block_neighbors` gained an `else if constexpr (D == MPS)`
  that throws. Without it the CPU branch is instantiated for MPS and fails on
  `try_cpu_spatial_compact_block_neighbors`, whose spatial sweep reads host
  memory -- and an MPS tensor's data pointer is an MTLBuffer object. The
  branch is unreachable anyway, since the MPS test precedes it.
- New: `ljlk_mps_launch.hh`, `ljlk_pose_score.mps.cpp`, the CMake shader
  entry (`kLJLKMetalShaderSource`), and `TMOL_DISPATCH_FLOATING_DEVICE_MPS`
  on the whole-pose forward op. `max_dis` is the one non-integral launch
  constant and travels as a float bit pattern in the int64 metadata buffer.

Measured, 1UBQ copies, median of 5:

| poses | cpu | mps | speedup | rel err |
|-------|-----|-----|---------|---------|
| 10  | 5.79 ms  | 3.76 ms  | 1.54x | 4.9e-07 |
| 100 | 49.48 ms | 20.99 ms | 2.36x | 7.0e-07 |

`pytest tmol/tests/score`: **998 passed, 884 skipped, 0 failed.**

The superseded step-1 plan follows, kept for the diagnostic commands in it.

## Step 1: finish ljlk (two errors)

1. `ljlk_load_tile_invariant_intrares_data` -- "requires 10 arguments, but 11
   were provided". The blanket `, lane` append added a lane argument at this
   call site, but the function never got a lane parameter, because the pass
   that added them only touched functions whose *bodies* mention `lane`, and
   this one does not reach a workgroup primitive directly. Fix: give it the
   trailing `WorkgroupLane<D> lane = {}` for uniformity with its siblings.

2. `call to deleted constructor of WorkgroupLane<MPS>` at
   `tile_evaluate_rot_pair`'s default argument. ljlk's bodies call
   `tile_evaluate_rot_pair`, but the pass that appended `, lane` only matched
   `tile_evaluate_block_pair`. Fix: pass `lane` at the `tile_evaluate_rot_pair`
   call sites in `ljlk_kernels.hh`. The deleted default constructor is the
   intended guard firing correctly -- it is how an unthreaded kernel is caught.

Then, following the elec template exactly:

3. Launch branch in `LJLKPoseScoreDispatch::forward`, tested **before** the
   compact-block-neighbor branches (those launch through DeviceDispatch, which
   on MPS would run the kernel on the CPU). See the elec forward for the shape.
4. `ljlk_mps_launch.hh` -- device-templated `if constexpr (D == MPS)` wrapper
   that packs the six scalars and calls `mps_launch`.
5. `ljlk_pose_score.mps.cpp` -- instantiate `<DeviceOperations, MPS, float, int>`
   only; Metal has no double.
6. CMake: `tmol_embed_metal_shader` for `ljlk_mps_kernels.metal.in` +
   `kLJLKMetalShaderSource`, and add the TU to `_C_SOURCES`.
7. `TMOL_DISPATCH_FLOATING_DEVICE_MPS` on the ljlk pose-score forward op only.
8. Build, run `tmol/tests/score`, then validate and time with a ljlk clone of
   `dev/mps_scratch/elec_mps_bench.py`.

## Steps 2 and 3: hbond, then lk_ball

Same recipe. **Order within each term matters** -- this is the lesson that
cost the most time:

1. Split any `params.hh` that device code includes: POD structs into
   `params_types.hh`, `TView`-holding tensor structs stay in `params.hh`.
2. Make the term's `<term>.hh` loaders view-generic (TView parameters become
   deduced template parameters).
3. **Build immediately.** Making the views generic costs `Real`, `Int` and `D`
   their deducibility, because they previously appeared only inside
   `TView<T, N, D>`. Every call site must then name them, and **the order
   differs per function**. Get the real diagnostic from the single TU:

       clang++ -std=c++17 -fsyntax-only -ferror-limit=1 \
         -DTORCH_TARGET_VERSION=0x020b000000000000 -DWITH_MPS \
         -DTMOL_INTEROP_NAMESPACE='"x"' -DTMOL_MAIN_MODULE \
         -I. -Itmol/extern -Igenerated \
         -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
         tmol/score/<term>/potentials/<term>_pose_score.cpu.cpp

   ninja's output elides the candidate note inside lambda substitution and
   reports only "no matching function"; this prints "could not match 'int'
   against 'float'", which names the problem at once.
4. Extract the lambda bodies into `<term>_kernels.hh`, rewriting captures onto
   a `<Term>CtxT` with `visit_views`. Remove the inline `SHARED_MEMORY union`
   -- Metal allows `threadgroup` declarations only at entry scope, so the entry
   declares it and passes it down.
5. Address spaces: `TMOL_SHARED_PTR`/`TMOL_SHARED_REF` for anything pointing
   into the tile buffers, `TMOL_THREAD` for ordinary references, and
   `threadgroup`-qualified member-function overloads where a shared-memory
   object has accessors (see `Mat` in matrix.hh and `lj_params()` in ljlk's
   `params_types.hh`).
6. Entry points, launch wrapper, instantiation TU, CMake, dispatch opt-in.
7. Build, test, validate, time.

### Term-specific things to expect

- **hbond**: `hbond.hh` already reads element 0 of a `portable::array`; its
  donor/acceptor scratch structs will need the shared-pointer treatment.
- **lk_ball**: has `water.hh` and `gen_pose_waters.impl.hh` in its chain, and
  uses `forall_independent` as well as workgroup launches -- the water
  generation kernel is a separate entry point, like sphere_overlap's pair of
  kernels.
- Any CPU-only accumulator typed `double` (ljlk had `cpu_pose_accum`) cannot
  be a context field: Metal cannot name `double`. Make it a template
  parameter so `if constexpr (D == CPU)` discards it untyped.

## Step 4: metadata cleanup

`mps_launch` rebuilds the int64 metadata buffer on the host and copies it to
the GPU on every launch -- three launches per elec forward. The contents only
change when shapes change, which is once per pose stack, not per call.

- Cache keyed on the metadata contents (or on the context's view pointers plus
  shapes), holding the device tensor alive in the `ShaderCache` alongside the
  libraries.
- Fold elec's and sphere_overlap's private copies of the Metal context builder
  onto the shared `mps_ctx_builder.hh`.
- Consider whether the placeholder binding for unset views (currently the
  metadata buffer stands in) is worth a dedicated one-byte dummy.

Expected gain: ~1.4 ms per call, so ~13% at 100 poses and little past 400.
Measure once, against all four ported terms.

## Step 5: benchmark

- Per-term validate-and-time scripts, as `elec_mps_bench.py` does.
- Then the end-to-end number that motivated the project: the phase-0 torsion
  relax on 100 x 1UBQ, CPU vs MPS, with the per-term breakdown. Use
  `dev/mps_scratch/phase0*.py`. Expect ~2.5x on scoring if all four terms land
  at 3x, and less end-to-end since scoring is 87% of the relax.

## Still outstanding, independent of the above

- **CUDA is unverified.** No nvcc on this machine. `WorkgroupLane`, the
  `tie`/`make_tuple` value-parameter changes, `TMOL_THREAD`, and the
  per-function explicit template arguments all touch CUDA code paths.
- Backward, rotamer scoring, and the compact-block-neighbor paths are
  CPU/CUDA-only on every term.
- The change set is large and uncommitted. Committing the portable layer
  separately from the per-term Metal work would let a CUDA regression be
  bisected to one or the other.

# Handoff: build is GREEN; ljlk port is partially done

`1104 passed, 907 skipped, 0 failed`. elec on MPS still validates at 3.0x on
400 x 1UBQ. ljlk's host path is intact; its Metal path is not wired up yet.

## The deducibility trap, and how it was fixed

Making a loader view-generic -- replacing `TView<T, N, D>` parameters with
deduced template parameters -- costs `Real`, `Int` and `D` their
deducibility, because they used to appear *only* inside the view type. Every
call site then has to name them explicitly, and **the order differs per
function**: some loaders are
`<DeviceDispatch, D, nt, Real, Int, views...>`, others
`<DeviceDispatch, D, nt, Int, Real, ...>`, others
`<DeviceDispatch, D, nt, TILE_SIZE, MAX_N_CONN, Real, Int, ...>`. A blanket
`<..., Real, Int>` substitution compiles for some and silently fails to match
for others, and clang reports it as a bare "no matching function" with the
candidate note elided inside the lambda substitution.

Getting the real reason requires compiling the one TU directly:

    clang++ -std=c++17 -fsyntax-only -ferror-limit=1 \
      -DTORCH_TARGET_VERSION=0x020b000000000000 -DWITH_MPS \
      -DTMOL_INTEROP_NAMESPACE='"x"' -DTMOL_MAIN_MODULE \
      -I. -Itmol/extern -Igenerated \
      -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
      tmol/score/ljlk/potentials/ljlk_pose_score.cpu.cpp

That prints `candidate template ignored: could not match 'int' against
'float'`, which names the problem immediately. **Do this first on the next
term rather than guessing from the ninja output.**

The per-function orders for ljlk are recorded in the ARGS map that was used to
fix them; regenerate it by listing each loader's template parameter list
before touching call sites.

## Also worth knowing

- `lk_isotropic.hh` used `std::max`/`std::min` on Real values. Unqualified
  `min`/`max` only exist on the host (diamond_macros.hh pulls them from
  <algorithm>); the portable shim's `fmin`/`fmax` work on both, so that is the
  right replacement.
- `params_types.hh` gained `threadgroup`-qualified `lj_params()`/`lk_params()`
  overloads: those parameters are read out of the shared tile buffer, and MSL
  makes the address space part of the implicit `this`.
- ljlk's CPU-only `cpu_pose_accum` is a `double*`, and Metal cannot name
  `double` at all, so it is a template parameter on the kernel bodies rather
  than a context field -- `if constexpr (D == CPU)` then discards it without
  type-checking.

## ljlk: what is done, what is left

Done: `LJLKCtxT` (22 views + visit_views), host and Metal binders,
`ljlk_kernels.hh` with both forward bodies extracted from the lambdas, and
`ljlk_mps_kernels.metal.in`. `ljlk_kernels.hh` compiles standalone as MSL 3.2.

Left: instantiating the Metal entries still reports a few errors (a
`WorkgroupLane<MPS>` default-construction where a loader default is taken, and
loader signature mismatches). Then the launch branch in the forward, an
`ljlk_pose_score.mps.cpp` instantiation TU, the CMake shader entry, and
`TMOL_DISPATCH_FLOATING_DEVICE_MPS` on the op. Nothing includes the new ljlk
files yet, so the build is unaffected by their state.

## lk_ball and hbond

Not started. Recommended order per term, learned the hard way: make the
headers view-generic, **build immediately**, fix deducibility with the direct
clang invocation above, build again, and only then touch Metal.

## Reusable machinery

`tmol/score/common/mps_ctx_builder.hh` -- Metal-side view rebuilder, promoted
out of elec. elec and sphere_overlap still carry their own copies; folding
them onto the shared one is tidy-up.

---

# hbond: a real blocker, with numbers

hbond stores its polynomial tables as **float64** and the type is not
templated: `TView<HBondPolynomials<double>, 2, Dev>` appears in hbond.hh,
hbond_pose_score.hh, hbond_pose_score.impl.hh and potentials.hh, and
`_params.py` builds the table with `dtype=torch.float64`. MPS has no float64
at either level -- the tensor cannot exist on the device and Metal cannot
name the type -- so the table has to become float32 to port the term at all.

**Rounding the coefficients to float32 is not acceptable.** Measured over
each polynomial's own range, restricted to the region where the polynomial is
physically meaningful (|V| < 10):

| polynomial | max |V64 - V32| |
|---|---|
| hbpoly_ahdist_aASN_dARG_9gt3_hesmooth_min1p6 | 1.9e-01 |
| hbpoly_ahdist_aTYR_dLYS_9gt3_hesmooth_min1p6 | 2.1e-02 |
| hbpoly_ahdist_aASN_dASN_9gt3_hesmooth_min1p6 | 1.6e-02 |

The worst case sits at x = 2.66 A where V = 6.03, and the error is still
1.1e-01 inside the ordinary x = [1.6, 2.4] hydrogen-bond window. A single
hbond is worth about -1 to -2, so this is a 10-20% error per bond -- three to
four orders of magnitude worse than the 7e-07 that elec and ljlk achieve.

**Re-centering the basis does not fix it.** Converting each polynomial to
u = (x - mid)/half, exactly, in float64, and then rounding, leaves the worst
error at 1.7e-01. The conditioning problem is intrinsic: these are
degree-10 least-squares fits that reach |V| ~ 1e7 inside their nominal range,
so even the centered coefficients run to 1.5e+06. The cancellation is in the
evaluation, not just the storage.

### The route that works

`poly<POrd, Real, EvalReal>` in score/common/polynomial.hh already separates
the coefficient type from the accumulator type. So:

- add a compensated float pair (`hi`, `lo` -- classic double-float, ~1e-14
  relative) to the portable layer, with add and multiply;
- store the coefficients as two float32 tensors instead of one float64 one;
- instantiate `EvalReal` as that type on Metal only.

CPU and CUDA numerics are untouched -- they keep the float64 table and the
float64 accumulator. The cost is roughly 3x the flops in polynomial
evaluation, on a term that is a small share of the total.

### Decision: hbond stays on CPU

Not pursued. The compensated-float route above would work, but it is a
numerics change to a shared evaluator for a term whose Metal payoff is
unproven, and the alternative -- shipping 0.1 kcal/mol errors -- is not
acceptable.

Nothing had to be torn out: hbond never got Metal scaffolding, and all seven
of its dispatch sites still use the CPU/CUDA-only `TMOL_DISPATCH_FLOATING_DEVICE`,
so the term is CPU and CUDA only by construction. The port-preparation that
remains -- the params_types.hh split and the portable-layer migration of
hbond.hh and potentials.hh -- is shared with every other term and stays.
`params_types.hh` carries a comment recording why no Metal kernel includes it.

---

# Session result: lk_ball ported, metadata cached, measured

## Per-term numbers (N copies of 1UBQ, median of 7, M2 Pro)

| term    | poses | cpu      | mps      | speedup | rel err |
|---------|-------|----------|----------|---------|---------|
| elec    | 10    | 4.19 ms  | 1.40 ms  | 2.99x   | --      |
| elec    | 100   | 35.21 ms | 11.44 ms | **3.08x** | 7.3e-08 |
| ljlk    | 10    | 5.78 ms  | 2.28 ms  | 2.53x   | --      |
| ljlk    | 100   | 48.34 ms | 20.54 ms | **2.35x** | 1.3e-06 |
| lk_ball | 10    | 2.52 ms  | 3.03 ms  | 0.83x   | --      |
| lk_ball | 100   | 19.76 ms | 22.41 ms | **0.88x** | 1.3e-07 |

`pytest tmol/tests/score`: 998 passed, 884 skipped, 0 failed.

## lk_ball is correct but not a win, and the split says why

LK-ball needs **two** Metal entries, because water generation is a separate op
whose output is an input to scoring; leaving it on the host would mean copying
the pose to the CPU and back on every call. Timing them separately at 100
poses:

| stage    | cpu      | mps      |
|----------|----------|----------|
| watergen | 3.45 ms  | 0.81 ms  | 4.3x
| scoring  | 17.98 ms | 22.71 ms | 0.79x

So the water generator is the best speedup of anything ported so far, and the
block-pair scoring kernel is a net loss. The likely cause is occupancy rather
than overhead: a threadgroup is one 32-wide simdgroup, and an LK-ball block
pair has far fewer polar/occluder pairs than an LJ/LK or elec block pair has
atom pairs, so most lanes idle. Its shared tile block is also the largest of
the three (two `TILE_SIZE * MAX_N_WATER * 3` coordinate arrays, ~4.5 KB).

Worth trying next, in order: a wider threadgroup for this kernel alone
(`TMOL_MPS_THREADGROUP_SIZE` is currently global, and
`shuffle_reduce_and_broadcast_in_workgroup` static_asserts a single simdgroup,
so this is not a one-line change); then fusing the two entries.

Until then lk_ball's MPS dispatch is enabled -- it is correct, and both the
energies and the tests agree -- but it is the one term where the Metal path
is slower than the host.

## Metadata caching (step 4) paid off most on small stacks

The per-launch metadata buffer is now cached on its contents in
`mps_launch.cpp` (`MetaCache`). Shapes and launch scalars change only when the
pose stack's shape changes, not per call, so a relax trajectory no longer pays
a host allocation, a memcpy and a host-to-device copy per launch. Keyed on
contents rather than on the caller because the scalars share the buffer: two
calls with identical shapes but a different `output_block_pair_energies` must
not share an entry.

Effect, ljlk at 10 poses: **1.54x -> 2.53x**. lk_ball at 10: 0.66x -> 0.83x.
At 100 poses the change is within noise, as expected -- it is fixed cost.

Also folded: elec's and sphere_overlap's private copies of the Metal context
builder now call the shared `mps_build_ctx` / `mps_scalars` in
`mps_ctx_builder.hh`. All five shaders still compile at MSL 3.2.

Separately, `ljlk_kernels.hh` and `lk_ball_kernels.hh` now `#undef` their
macros at the end. Both files previously redefined the same macro names that
their `.impl.hh` keeps for the rotamer and derivative paths -- 16 redefinition
warnings per ljlk TU, silently relying on include order.

## End-to-end torsion relax: not reachable yet, and why

A full `beta2016_score_function` cannot score on MPS, so the phase-0
comparison cannot be run. Two distinct blockers, both pre-existing:

1. `ZeroTermPoseScoringModule` in `tmol/score/common/_scoring_module.py`
   allocates `torch.zeros((), dtype=torch.float64, device=device)`, which MPS
   rejects outright. Generic plumbing, unrelated to any term.
2. Even past that, the unported terms' ops reach
   `TMOL_DISPATCH_FLOATING_DEVICE`, which `AT_ERROR`s on MPS by design.

So end-to-end numbers wait on the remaining terms. The per-term table above is
what can honestly be measured today.

## New things this session that are worth knowing

- **`tmol::portable::math::isnan` is a bit-pattern test on Metal**, not
  `metal::isnan` and not `x != x`. The Metal compiler may assume finite
  operands and fold either spelling to false. LK-ball marks absent waters with
  NaN and relies on the test, so it checks the exponent and mantissa bits,
  which no float-contraction rule can elide.
- **A discarded `if constexpr` branch is still parsed.** `reinterpret_cast<Real*>`
  inside a `Dev == CPU` branch fails on Metal even though the branch is never
  instantiated, because the cast names a pointer type. Spell it
  `TMOL_DEVICE_PTR(Real)`.
- **A static data member of a class template is program scope in MSL** and must
  live in `constant`. `lkball_globals<Real>`'s four `static constexpr Real`
  members became static member functions returning the literal; integral cases
  (`AcceptorHybridization`) became enumerators.
- **Macro-defined constants collide with template parameter names.**
  `constants.hh` defines `TILE_SIZE`, `MAX_N_WATER` and `MAX_N_CONN`; including
  it before `lk_ball.hh` turns `template <..., int TILE_SIZE, ...>` into
  `template <..., int 32, ...>`. The kernel headers include it last, with a
  comment saying so.
- **`bonded_atom.hh`'s `RotamerCentricIndexedBonds` is now view-templated**, and
  its two pre-sliced members became full-rank views plus a `pose_ind`. A slice
  has a different type per backend (`TensorAccessor` vs `MetalView`), and
  naming that difference would cost a second template parameter everywhere the
  struct travels. `IndexedBonds` and `BlockCentricIndexedBonds` are guarded off
  on Metal -- they are host and CUDA only.
- **MPS has no float64, so the hbond polynomial tables are pinned to the host**
  (`poly_device` in `tmol/score/hbond/_params.py`). LK-ball pulls the hbond
  resolver onto the GPU but reads only the donor/acceptor listings.
- **Thirty views is the ceiling.** `WaterGenCtxT` binds 30 buffers plus the
  metadata buffer, which is exactly the 31-binding per-kernel limit on Apple
  GPUs. Another field means packing the small per-block-type tables first.

---

# Full energy function and relax: measurements and the real bottleneck

## No CUDA on this machine

A CPU-vs-CUDA comparison cannot be run here: `torch.cuda.is_available()` is
False and `torch.backends.cuda.is_built()` is False -- this torch build has no
CUDA at all, and there is no nvcc. Everything below is CPU vs MPS.

## What had to be fixed before a score function would run on MPS at all

Three blockers, all pre-existing and none term-specific:

1. `ZeroTermPoseScoringModule` allocated `torch.zeros((), dtype=torch.float64,
   device=device)`. MPS rejects float64 outright. The f64 buffer now lives on
   the host there; a float64 pose cannot reach this backend anyway, since the
   dispatch macros reject the dtype first.
2. `HBondParamResolver` and `CompactedHBondDatabase` built their float64
   polynomial tables on the requested device. Those are pinned to the host
   (`poly_device`). LK-ball is what pulls the hbond resolver onto a GPU, and it
   reads only the donor/acceptor listings, never the polynomials.
3. The shared compact block-neighbor list is built by
   `build_compact_block_neighbors`, which has no Metal kernel. MPS now opts out
   and the ported kernels read the dense `scratch_rot_neighbors` grid that
   sphere_overlap produces on the device.

**And one latent bug that opting out exposed.** `_fused_ljlk_elec_module_index`
indexes `_execution_modules`, but `unweighted_scores` enumerates
`term_modules` whenever the shared neighbor list is absent. On CPU and CUDA
those two conditions never co-occur -- the fused module exists only when both
ljlk and elec are present, which also means two shared-neighbor terms -- so the
mismatch was unreachable. Opting MPS out of the shared list created exactly
that combination and `forward_weighted` was called on a
`TermWholePoseScoringModule`, which has no such method. Both the shared-list
decision and the fused-weight decision now go through one predicate,
`_shares_block_neighbors`, so they cannot disagree.

## A. Full beta2016 energy, all-CPU vs ported-terms-on-MPS

Machine idle, median of 7, N copies of 1UBQ (1231 atoms each). ljlk and elec on
Metal, the other nineteen score types on the host. Energies agree to 4e-07.

| poses | forward cpu | forward mixed | | fwd+bwd cpu | fwd+bwd mixed | |
|-------|-------------|---------------|------|-------------|---------------|------|
| 10  | 11.30 ms  | 5.82 ms  | **1.94x** | 20.97 ms  | 12.06 ms  | **1.74x** |
| 100 | 97.58 ms  | 34.30 ms | **2.85x** | 177.88 ms | 74.49 ms  | **2.39x** |
| 300 | 289.68 ms | 94.82 ms | **3.06x** | 508.68 ms | 216.99 ms | **2.34x** |

Half-by-half at 100 poses:

| | fwd+bwd |
|---|---|
| ljlk + elec on MPS | 49.60 ms |
| ljlk + elec on CPU | 140.53 ms (2.83x) |
| the other 19 score types, on CPU | 59.53 ms |
| sum of halves | 109.13 ms |
| measured mixed total | 74.49 ms |

The measured total is *below* the sum of halves, which is the overlap working.

## B. Relax

Works end to end. `dev/mps_scratch/relax_mixed.py`.

**The kinematic layer stays on the host.** `tmol/kinematics/compiled/compiled_ops.cpp`
has thirteen dispatched ops (four floating, nine index) and none has a Metal
backend -- the first one a relax hits is `calculate_ff_edge_delays`. So the
network's coordinates are CPU-resident and the composite copies up for the
Metal half rather than down. The transfer is the same 0.2-0.7 ms either way.

Integration is one line: build the network against the host pose stack with a
score function holding only the Metal-backed terms, then assign the composite
to `net.whole_pose_scoring_module`. `KinForestSfxnNetwork` uses the score
function for nothing else.

100 x 1UBQ, full torsion relax:

| | cpu | mps |
|---|---|---|
| 1 fwd+bwd (median) | 191.65 ms | 87.29 ms (**2.2x**) |
| network forwards during relax | 330 | **540** |
| full relax wall | 65.95 s | 63.05 s (**1.05x**) |
| final E | -6127.77 | -6081.27 |

So the per-evaluation speedup is real and the relax barely moves, because the
MPS run needs 64% more line-search evaluations.

## C. What is actually costing performance

Ranked by how much they cost, with numbers.

### 1. Non-deterministic accumulation breaks the line search -- the big one

Twelve identical scoring calls, same coordinates, no gradient:

| | distinct results | spread |
|---|---|---|
| CPU | 1 | 0 (bit-identical) |
| MPS | 2 | 9.8e-04 on -11673 |

Gradients, six identical calls: CPU deviation exactly 0, MPS 7.6e-06 on a
gradient whose largest component is 67.

The cause is `accumulate<Device::MPS>`, which is
`metal::atomic_fetch_add_explicit`. Float addition is not associative and the
order in which workgroups land their atomics varies per launch, so the pose
total jitters. The CPU path accumulates in a fixed order, and in double.

This is what the relax's extra 210 evaluations are. `LBFGS_Armijo` validates
its analytic derivative against a finite difference, and the minimizer printed

    Inaccurate G! Segment= 20 Step= 4.5e-13 Deriv= -6858.87 Finite= 3.69e+08

At a step of 4.5e-13 the true energy change is many orders of magnitude below
the 1e-3 jitter floor, so the finite difference is pure noise, the step is
rejected, and the search retries.

**Fix:** give each workgroup its own output slot and reduce deterministically
in a second pass -- the block-pair scoring mode already writes per-block-pair
cells, so the machinery exists. Costs one extra small kernel and
`n_poses * n_workgroups` floats. Integer fixed-point atomics would also be
exactly associative, but Metal's 64-bit atomic support is not dependable and
32 bits has too little dynamic range for these totals.

Until this is fixed, the per-evaluation speedup does not reach a minimizer.

### 2. Enqueue is not submission -- worth 1.5x on its own, already fixed

Torch's MPS backend records work into a command buffer and commits it only
when something forces it. The entire ljlk+elec forward and backward returns to
Python in **0.20 ms** and then runs ~49 ms on the GPU -- but it does not *start*
until a commit. Without a nudge the GPU sat idle through the host half and
began only when the final add waited on it:

| | host fwd | wait at the add | total fwd+bwd |
|---|---|---|---|
| no commit | 35.6 ms | 49.2 ms | 110.9 ms |
| `torch.mps.Event().record()` after enqueue | 35.4 ms | 13.5 ms | **73.7 ms** |

`torch.mps` exposes no explicit `commit()`; recording an event is the handle.
This took the 100-pose full energy from 1.60x to 2.39x and is in the composite.

### 3. Host-to-device transfer is not the problem

0.19 ms at 10 poses, 0.19 ms at 100, 0.70 ms at 300 -- **0.3% to 1.6%** of the
mixed call. It is worth saying clearly because it is the intuitive suspect: a
(300, 1231, 3) float32 copy is 4.4 MB over unified memory and costs nothing.

What *is* worth care is where the copy sits. A device-to-host copy blocks until
the queue drains, so taking it before anything is enqueued costs its own
bandwidth, and taking it afterwards costs the whole queued GPU pipeline. That
is a scheduling property, not a bandwidth one.

### 4. The host half is now the floor

At 100 poses the host terms are 59.53 ms against the Metal half's 49.60 ms, so
the host is the longer pole. Within it, **lk_ball is 34.30 ms -- 31% of the
whole mixed call**, more than the next five groups combined (cartbonded 6.83
ms, the rest below noise). lk_ball is on the host by choice: its forward is
ported and correct but 0.88x, and its backward has no Metal kernel.

Finishing lk_ball properly -- the threadgroup-width work in the previous
section, plus its backward -- is the second-largest lever after determinism.
The leave-one-out numbers for the smaller host groups come out negative and
should not be read as term costs: the host terms run concurrently across a
thread pool, so removing one changes the sharding.

### 5. The backward lags the forward

At 100 poses the forward is 2.85x but forward+backward is 2.39x; at 300, 3.06x
against 2.34x. The forward overlaps because the composite enqueues, commits,
and then does host work. The backward does not: the autograd engine decides the
order, so the Metal backward is not committed early the way the forward is. A
custom autograd Function for the composite could do for the backward what the
event nudge does for the forward.

---

# Determinism: what the CUDA warp-reduce trick buys, and what it does not

## The warp-level part was already there

`DeviceOperations<MPS>::reduce_in_workgroup` is the CUDA shape already:
`metal::simd_sum` across the simdgroup (the direct analogue of a warp
shuffle-reduce), per-simdgroup partials combined through threadgroup scratch,
and then `if (tid == 0)` one `accumulate<MPS>` atomic per threadgroup. At
`TMOL_MPS_THREADGROUP_SIZE == 32` a threadgroup *is* one simdgroup, so a pose
total already costs exactly one atomic per block pair rather than one per
thread.

So the amelioration was in place, and the jitter was what remained after it:
roughly 2900 block pairs per 1UBQ pose, each contributing one atomic, arriving
in an order the GPU is free to vary, into a non-associative float32 add.

## What was done: remove the remaining atomics for the energy

The store already had a no-atomic mode. `output_block_pair_energies` writes
each threadgroup's total to its own cell, `output[term][pose][b1][b2]`, instead
of atomically adding into `[term][pose][0][0]`. It was just unreachable
together with derivatives, because entry selection keyed off the *output
shape* rather than off whether gradients were wanted.

Now, on MPS whole-pose scoring: allocate the cell-shaped buffer, let the store
write cells, choose the entry on `compute_derivs` alone, and reduce the cells
afterwards in a fixed order. The CPU keeps its serial double accumulator and
CUDA keeps its atomics; neither path changes. Cost is one float per block pair
per pose (7 MB at 100 poses, 21 MB at 300 for ljlk's three terms) plus one
reduction.

Result:

| | before | after |
|---|---|---|
| distinct energies over 12 identical calls (MPS) | 2 | **1** |
| spread | 9.8e-04 | **0** |
| `Inaccurate G!` during a 100-pose relax | fires | **silent** |
| full energy 100 poses, fwd+bwd | 2.39x | 2.34x |
| full energy 300 poses, fwd+bwd | 2.34x | 2.30x |

So the energy is now bit-reproducible, the line search's derivative-validation
warning is gone, and it cost nothing measurable.

## But it did not fix the relax, and the reason is precision, not ordering

Relax forwards at 100 poses went 540 -> 525 against the CPU's 330. Still ~60%
more evaluations, and the final energy still varies run to run (-6069, -6081,
-6149).

A finite-difference probe along a fixed random unit direction says why.
Relative error between the analytic directional derivative and
`(E(x + s*d) - E(x)) / s`:

| step | cpu | mps |
|------|-----|-----|
| 1e-04 | 1.000e+00 | 2.83e+01 |
| 1e-05 | 1.000e+00 | 5.87e+02 |
| 1e-06 | 1.000e+00 | 5.86e+03 |
| 1e-07 | 1.000e+00 | 5.86e+04 |

A relative error of exactly 1.000 means the finite difference came out as
**zero** -- the energy did not change at all. That is the CPU: the true change
over such a step is ~1e-8 on a total of 26100, far below a float32 ULP
(~0.002), and because the CPU sums the pose total in *double* and rounds once,
the returned float32 is identical. The step is cleanly rejected as "no change".

On MPS the error grows as 1/step, which is the signature of the energy moving
by about one ULP when it should not move at all. The pose total is summed in
float32 across ~2900 cells, so it carries roughly 11 ULP of summation error
(MPS -11673.2412 vs CPU -11673.2520, a 0.0108 gap), and that error shifts
discontinuously under a tiny coordinate perturbation. The line search reads
those shifts as real slope.

**Determinism was necessary but not sufficient.** The conclusion drawn here
at the time -- that the remaining gap was the float32 pose accumulator -- was
wrong, and the section below records what the gap actually was. Keep reading
before acting on this paragraph.

Two further things are still non-deterministic and worth naming separately:
the gradient, which still goes through an atomic scatter into `dV_dcoords`
(7.6e-06 deviation across six identical calls, against a largest component of
67), and therefore the relax trajectory.

## What to do next, in order

1. **Compensated reduction of the cells.** A Kahan or two-float (hi/lo) sum
   brings the pose total to roughly double accuracy without needing a float64
   type. Only ~2900 cells per pose, so a single threadgroup per pose is
   enough. This is the same portable compensated-float machinery hbond's
   polynomials would need -- one piece of work would serve both, which is an
   argument for building it properly rather than inline.
2. **Deterministic gradient accumulation**, which is the harder one: atoms are
   shared between block pairs, so the scatter genuinely contends. Options are
   per-block-pair gradient storage (the existing code comment calls the memory
   prohibitive, and it is), evaluating each pair from both sides so each
   workgroup owns its writes (deterministic, no atomics, 2x the pair work), or
   integer fixed-point atomics, which are exactly associative but need either
   dependable 64-bit atomics in MSL or a dynamic range argument for 32 bits.
3. Only after those does the 2.2x per-evaluation speedup reach a minimizer.

---

# The relax gap, found: it was the gradient, not the energy

Three hypotheses were tested and discarded before the right one. Recording all
of them, because each rules out a direction someone would otherwise retry.

**Accumulator width is not it.** Building with ljlk's serial CPU pose
accumulator changed from `double` to `float` left the host total bit-identical
(-11673.251953 either way) and the finite-difference behaviour unchanged. elec
on CPU already accumulates in float32. So the GPU's lack of float64 was never
the problem -- which is worth knowing, because it is the first thing one
assumes.

**The composite is not it.** Running the same two-half composite with *both*
halves on the host took 331 line-search evaluations, matching the plain host
score function's 330. Two separately weighted partial scores summed per pose
is not what cost the evaluations.

**Energy roughness is not it either.** The compensated reduction below did
make the Metal energy as smooth as the host's -- zero change for perturbations
below 1e-05, one ULP at 1e-04, matching the host exactly -- and the relax
still took 526 evaluations.

**It was the gradient.** Substituting the host's gradient while keeping the
Metal energy gave **331 evaluations and a final energy of -5927.979, matching
the all-host control's -5927.976 to five figures**. That is as clean as a
bisection gets.

## Why the gradient and not the energy

Energies land in per-block-pair cells, one threadgroup each, no atomics.
Derivatives cannot work that way -- an atom is shared between every block pair
its block participates in, so the scatter genuinely contends, and per-block-pair
derivative storage is the memory the existing code comments already call
prohibitive. So derivatives went through atomic float adds, whose arrival order
varies per launch, making the gradient neither reproducible nor a smooth
function of the coordinates.

## The fix: an exactly-representable grid

Two measurements set the design. First, a deterministic gradient *coarsened*
to 1e-03 still gave 336 evaluations, and to 6.1e-05, 325 -- so the minimizer
does not need a fine gradient, it needs a consistent one. Second, Metal has no
64-bit atomics (`atomic_fetch_add_explicit` rejects `ulong`; 32-bit `int`
works), so the obvious fixed-point route needs either a carry protocol or a
dynamic-range argument.

Neither is necessary. Rounding each contribution to a multiple of 2^-13 makes
the float32 sum *exact*: a float32 represents an exact multiple of 2^-13 up to
a magnitude of 2^23 * 2^-13 = 1024, so every partial sum below that is
unrounded and the additions commute. Order-independence then falls out of
ordinary float arithmetic -- no integer atomics, no second buffer, no
conversion pass.

It lives in `accumulate_grad`, a new seam alongside `accumulate` in
score/common/accumulate.hh. On CPU and CUDA it *is* `accumulate` -- inheritance,
no code, no numeric change. Only the Metal specialization snaps. The seam also
documents at each call site which accumulations are derivatives and which are
energies, which the previous code could not distinguish (both went through
`accumulate<D, Real>`).

Past a component magnitude of 1024 exactness lapses and that component is
approximate again. That is a graceful degradation; derivative components here
reach tens.

Results, 100 poses:

| | before | after |
|---|---|---|
| gradient deviation over 6 identical calls | 7.6e-06 | **0** |
| gradient vs host, max abs (on a max component of 67) | 5.5e-06 | 1.6e-03 |
| relax line-search evaluations | 526 | **332** (host: 330) |
| relax wall | 61.5 s | **36.0 s** (host: 65.3 s) |
| final energy | -6069 to -6149 across runs | -6130.287 (host -6127.769) |

## Also landed: compensated pose-total reduction

`tmol/utility/portable/compensated.hh` -- Neumaier summation, compiles for all
three backends. Neumaier rather than plain Kahan because block-pair energies
are frequently larger than the running total early on, which is the case plain
Kahan gets wrong.

`tmol/score/common/pose_total_reduce*` -- a device-generic reduction of the
cell buffer to one total per pose, one workgroup per (term, pose), each lane
walking a fixed stride and lane 0 merging in lane order. CPU and CUDA reach it
through `DeviceOperations` like any other kernel; MPS through a named entry.
It replaced a torch tree-sum.

This did not change the evaluation count, but it is what makes the Metal
energy a correctly-rounded function of the coordinates rather than one that
jumps by several ULP under a sub-ULP perturbation, and it costs 1.2% of GPU
time. It is also the machinery hbond's degree-10 polynomials would need.

---

# Performance: fixed-step torsion relax

A converged relax mixes evaluation speed with evaluation *count*, so
`dev/mps_scratch/relax_steps.py` runs a fixed number of optimizer iterations
instead, using `LBFGS_Armijo(fixed_iterations=True)`. Both backends performed
43 energy evaluations in 20 iterations at every size, so the comparison is
equal work.

| poses | cpu / step | mps / step | speedup |
|-------|-----------|-----------|---------|
| 10  | 48.43 ms   | 30.84 ms  | **1.57x** |
| 100 | 408.67 ms  | 191.23 ms | **2.14x** |
| 300 | 1180.33 ms | 556.33 ms | **2.12x** |

Converged relax, 100 poses: 65.28 s -> **36.01 s (1.81x)**, 330 vs 332
evaluations, final energy -6127.77 vs -6130.29.

Whole-pose energy, for reference:

| poses | fwd cpu | fwd mps | | fwd+bwd cpu | fwd+bwd mps | |
|-------|---------|---------|------|-------------|-------------|------|
| 10  | 11.35 ms  | 6.29 ms  | 1.81x | 20.75 ms  | 13.13 ms  | 1.58x |
| 100 | 97.55 ms  | 35.08 ms | 2.78x | 177.51 ms | 76.00 ms  | 2.34x |
| 300 | 292.06 ms | 95.99 ms | 3.04x | 507.38 ms | 221.95 ms | 2.29x |

## Where the remaining time goes

Per evaluation at 100 poses: kinematics forward 4.75 ms, scoring forward and
backward 79.70 ms, whole network 89.94 ms. **Scoring is 89% of an evaluation**,
so the kinematic layer staying on the host costs almost nothing.

Inside the scoring call:

| phase | ms |
|---|---|
| coords host->device | 0.70 |
| Metal enqueue | 0.31 |
| host forward (Metal forward runs concurrently) | 39.40 |
| waiting for the Metal forward at the gather | 10.16 |
| backward (both halves, overlapped -- final sync is 0.00) | 26.90 |
| **total** | **77.44** |

Both halves are already overlapped as far as the work split allows: the Metal
forward spans 49.6 ms against the host's 39.4, so the 10.16 ms is the Metal
half being the longer pole, and the backward's final sync is zero because the
Metal backward finishes inside the host backward's window.

## Optimizations tried, with outcomes

**Partition.** Measured all three: both terms on Metal 189.72 ms/step, ljlk
only 313.49, elec only 276.84. The current partition is already optimal -- the
Metal half has to carry both to be worth its latency.

**Wider threadgroups: rejected, with a reason.** The workgroup primitives were
generalized to any width -- `shuffle_reduce_and_broadcast_in_workgroup` now
falls back to a threadgroup-scratch pass past 32 simdgroup lanes, component by
component so it serves `Vec` as well as scalars, which lifted an artificial
`static_assert`. But built at 128 the energies came out 7.2e-03 wrong and the
kernels were *slower* (114.83 ms against 75.72). The tile evaluators assume one
thread per tile slot and TILE_SIZE is 32. Widening means auditing
tile_atom_pair_evaluation.hh and the term kernels first; the constant is not
the obstacle. The generalization is kept -- it is correct, and the 32-wide path
is unchanged -- and `TMOL_MPS_THREADGROUP_SIZE` now records why it cannot move.

**Sharing sphere_overlap between terms: not worth it.** Each term computes its
own block spheres and neighbours on MPS, since MPS opts out of the shared
compact list. A new opt-in timer (`TMOL_MPS_TIME_KERNELS=1`, off by default)
attributed the GPU time:

| kernel | share | per call |
|---|---|---|
| ljlk_eval_energies | 64.8% | 33.6 ms |
| elec_eval_energies | 31.3% | 16.2 ms |
| compute_block_spheres | 1.5% | 0.38 ms |
| detect_block_neighbors | 1.3% | 0.35 ms |
| pose_total_reduce | 1.2% | 0.31 ms |

Sharing the neighbour work would save about 0.7 ms per evaluation, under 1% of
a step. Settled: not worth the plumbing.

## What is left, in order of expected payoff

1. **The ljlk scoring kernel is 65% of GPU time** (33.6 ms per evaluation at
   100 poses) and elec another 31%. Everything else is noise. Occupancy is the
   first thing to look at: `LJLKBlockPairSharedData` is about 5 KB per
   threadgroup, which at 32 KB of threadgroup memory per core caps residency
   near seven simdgroups. Shrinking the shared tile block, or the threadgroup
   audit above, is where the next factor comes from.
2. **lk_ball**: 31% of the *host* half. It should stay on the host as things
   are -- the Metal half is already the longer pole, and moving work there
   makes the stall worse -- but a faster lk_ball kernel plus its backward would
   change that arithmetic.
3. Fusing elec and ljlk into one Metal entry, the way
   `_FusedLJLKAndElecWholePoseModule` does on the host, would share the tile
   loads the two kernels currently each perform.

---

# Optimizing the ljlk and elec kernels

## The probe that mattered, and the one that lied

Inflating the threadgroup block with a dummy array and timing the kernel is
the cleanest way to ask whether occupancy binds. The first attempt passed the
pad size through `CXXFLAGS` and showed **no effect** -- which was wrong, and
wrong in the most dangerous direction. `cmake/TmolMetalShader.cmake` flattens
the shader with its own fixed `clang -E` invocation that never sees
`CXXFLAGS`, so the define reached the host build and not the kernel. Spelled
as a literal in the header instead, the answer was emphatic:

| ljlk threadgroup block | kernel time |
|---|---|
| 5032 B (as shipped) | 677.6 ms / 20 calls |
| +4096 B | 943.9 ms (+39%) |
| +12288 B | 1680.9 ms (+148%) |

**Anything that measures the Metal kernels has to put the knob in a header.**

## ljlk: tile the atom type, not the parameters

Two thirds of ljlk's block -- 3328 B of 5032 -- was a copy of
`LJLKTypeParams` per tile atom: 13 floats, of which six are booleans stored as
floats and two (`lk_dgfree`, `lk_lambda`) are read by nothing in the scoring
path at all. The tile now holds a 16-bit atom type index and the three readers
take the device parameter table, which is a couple of kilobytes shared by
every workgroup and sits in cache.

Block: **5032 B -> 1832 B**. ljlk kernel **33.6 -> 26.5 ms** per call at 100
poses (-21%); the term standalone **20.54 -> 15.04 ms** (-27%). Energies and
gradients unchanged within the usual 1e-07.

A 16-bit index rather than 8-bit: there are 122 ljlk atom types today, which
fits a byte, but not with headroom, and the loader needs the -1 "no atom here"
sentinel it already relied on.

## elec: charges do not need tiling at all

elec's block was already lean (1960 B), and its only fat item was 256 B of
per-tile partial charges. Those live contiguously in the block type's own row
of the charge table, so the tile can point at the row instead of copying it --
no index, no shared storage, one fewer cooperative copy per tile.

Block **1960 B -> 1704 B**; kernel 16.2 -> 15.83 ms (-2%). Small, as the size
change predicts, but it is strictly less work and less memory.

**It also broke the intra-residue path, and the bug is worth remembering.**
When both halves of an intra-residue pair are the same tile, only tile 1 is
loaded and `elec_load_intrares_data_from_shared` aliases r2's pointers to the
"1" arrays. Coordinates were aliased there; charges used to be aliased by the
same mechanism and now are not loaded at all for r2 in that case. Segfault.
The fix is three lines aliasing the charge pointer alongside the coordinate
one. Anything that moves data out of the tile has to check that function.

## Persistent threadgroups: tried, measured, reverted

Scoring dispatches one workgroup per block pair -- 292,600 for a hundred 1UBQ
poses -- and only about a third of pairs are neighbours, so roughly two thirds
exit immediately. Measured by forcing every workgroup to exit:

| ljlk, all workgroups exit | per call |
|---|---|
| exit after reading the neighbour grid | 8.2 ms |
| exit without the read (dispatch + index math only) | 5.9 ms |

against a full kernel of 27 ms. So about 30% of the kernel is spent getting
workgroups started, and roughly two thirds of that is on pairs with no work.
elec is worse: 7.3 ms of its 16.2 ms, 45%.

That looked like a clear 14-21% sitting there, so: cap the grid and have each
group stride over the remaining pairs, turning a skipped pair from a
threadgroup launch into a loop iteration. Barrier between iterations, since
the group's block is reused.

**It made ljlk slower -- 27.0 to 29.4 ms -- at every cap tried (4096, 16384,
65536).** The premise was wrong. That 5.9 ms is not an overhead a software
loop can amortize; it is what it costs to stream 9.4 million threads through
the machine, and Apple's scheduler retires a dead workgroup faster than a
loop can skip one. Striding also costs load balance: a persistent group that
draws many live pairs becomes a straggler, where independent workgroups let
the hardware fill the gaps. Reverted in full.

The remaining dead-workgroup cost is real but needs a genuinely compacted job
list to collect -- `nonzero()` on the dense neighbour grid would do it, but it
requires the count on the host, and a mid-call synchronize costs the
host/device overlap that is currently worth about 10 ms.

## Where this leaves things

| | before | after |
|---|---|---|
| ljlk kernel, per call at 100 poses | 33.6 ms | **26.5 ms** |
| elec kernel, per call | 16.2 ms | **15.8 ms** |
| relax, per optimizer step at 100 poses | 191.2 ms | **173.7 ms** |
| relax speedup over the host | 2.14x | **2.32x** |

Final, machine idle, after the probe scaffolding was removed. Fixed-step
relax, 20 optimizer iterations, 43 energy evaluations on both backends:

| poses | cpu / step | mps / step | speedup |
|-------|-----------|-----------|---------|
| 10  | 48.93 ms   | 30.59 ms  | **1.60x** |
| 100 | 403.64 ms  | 173.69 ms | **2.32x** |
| 300 | 1162.05 ms | 500.83 ms | **2.32x** |

Converged relax, 100 poses: 64.50 s -> **31.39 s (2.05x)**, final energy
-6127.77 against -6130.29.

Whole-pose energy:

| poses | fwd cpu | fwd mps | | fwd+bwd cpu | fwd+bwd mps | |
|-------|---------|---------|------|-------------|-------------|------|
| 100 | 103.16 ms | 34.54 ms | 2.99x | 179.02 ms | 68.29 ms  | **2.62x** |
| 300 | 305.75 ms | 97.16 ms | 3.15x | 507.52 ms | 195.46 ms | **2.60x** |

`pytest tmol/tests`: 3647 passed, 2143 skipped, 4 xfailed, 0 failed.

Next, in order: ljlk's remaining block is 1832 B, of which 768 B is coords and
768 B path distances -- the count-pair data would need a representation change
to shrink, and the padding probe says the next kilobyte is worth about 5% on
ljlk and 16% on elec. After that, fusing elec and ljlk into one entry is the
structural win: they tile the same coordinates and the same count-pair data,
and they pay the 5.9 ms dispatch floor twice.

---

# A relax cycle is four operations, and minimization is the small one

Everything above optimized the score+derivative evaluations and the optimizer
driving them. That is two of the four operations in a relax cycle, and
measurement says they are not the expensive two.

`dev/mps_scratch/relax_full_profile.py`, host, one pack-and-minimize cycle:

| | 10 x 1UBQ | 100 x 1UBQ | share |
|---|---|---|---|
| **rotamer-pair energies (calcRPEs)** | 5630 ms | 49200 ms | **71%** |
| build interaction graph | 650 ms | 6620 ms | 9.5% |
| SA over rotamer assignments | 350 ms | 3600 ms | **5.2%** |
| build rotamers | 340 ms | 1110 ms | 1.6% |
| minimization (both operations) | 979 ms | 8597 ms | 12.4% |
| **cycle total** | **7949 ms** | **69233 ms** | |

The split is the same at both sizes. Within minimization, measured separately
(`relax_profile.py`): scoring forward 54%, backward 38%, kinematics forward
5%, **the optimizer's own bookkeeping 2.9%**.

## What that means for the work already done

Minimization is 2.32x faster on Metal, but it is 12.4% of a cycle, so the
whole cycle moves **1.05x**. The honest headline for relax is that number, not
2.32x.

## Answering the annealing question directly

Simulated annealing over rotamer assignments is **5.2% of a cycle**. A perfect
Metal port of the annealer -- and `tmol/pack/compiled/` is a substantial body
of code -- buys at most 5%, and realistically three or four. It is not where
the time is.

Likewise the minimization-update operation: the Armijo and LBFGS ops are 2.9%
*of the minimization phase*, which is itself 12.4% of a cycle. Porting
`armijo_compiled` is worth about a third of one per cent.

## Where the time actually is

Rotamer-pair energies are 71% of a cycle, so that is the only phase whose port
changes the picture. Splitting it by term (`rpe_terms.py`, 10 poses, real
packing path):

| score function | calcRPEs |
|---|---|
| full beta2016 | 5960 ms |
| ljlk + elec only | 4010 ms (**67%**) |
| everything except ljlk + elec | 2130 ms (36%) |

So **ljlk and elec rotamer scoring alone is about 48% of a relax cycle** --
four times the whole of minimization.

## How much of it is already built

More than I expected. The rotamer kernels are the same tiled block-pair
evaluators as whole-pose scoring, differing in output rank: whole-pose writes
`[term, pose, block, block]`, rotamer scoring writes `[term, block_pair]`.
That is exactly the `OutRank` parameter the context template already carries,
and **elec's rotamer entries are already written and compiled as MSL**:

    TMOL_ELEC_ENTRY(elec_eval_energies_by_block_f32, ..., RotCtx)
    TMOL_ELEC_ENTRY(elec_eval_derivs_f32, ..., RotCtx)

What is missing for elec is the wiring, the same shape as the whole-pose port:
an MPS branch in `ElecRotamerScoreDispatch::forward`, a launch wrapper, and
`TMOL_DISPATCH_FLOATING_DEVICE_MPS` on `elec_rotamer_score_op`, which today is
the CPU/CUDA-only macro. ljlk needs its rotamer kernels extracted the same way
its whole-pose ones were -- a bigger job, but a recipe that has now been run
three times.

Both will also need the cells-plus-fixed-order-reduction treatment and
`accumulate_grad`, for the same determinism reasons as whole-pose scoring.

## Projected payoff

If rotamer scoring for ljlk and elec reached the 2.6-3.0x their whole-pose
kernels get, at 100 poses: calcRPEs 49.2 s -> 28.2 s, cycle 69.2 s -> 43.4 s,
**1.60x on the whole relax** against today's 1.05x. The remaining terms'
rotamer scoring (23% of a cycle) would be the next tranche after that.

Ranked, by what each is worth on a full cycle:

| work | ceiling |
|---|---|
| ljlk + elec rotamer scoring | ~31% of cycle time removed |
| other terms' rotamer scoring | ~15% |
| build interaction graph | ~9.5% |
| further minimization work | ~7% remaining |
| SA on Metal | ~5% |
| armijo / LBFGS ops on Metal | ~0.4% |

---

# lk_ball: the tile, and whether its backward needs porting

## Its threadgroup block is the largest of the three, which explains 0.88x

| | bytes | share |
|---|---|---|
| water_coords (2 tiles) | 3072 | 43% |
| lk_ball_params (2 tiles) | 2304 | 32% |
| pose_coords (2 tiles) | 768 | 11% |
| path_dist (2 tiles) | 768 | 11% |
| the rest | 236 | 3% |
| **total** | **7148 B** | |

That is 40% larger than ljlk's *original* 5032 B, and the occupancy probe put
5 KB -> 9 KB at +39%. lk_ball benchmarking at 0.88x was never a statement
about its arithmetic; it is sitting deep in the occupancy-bound regime.

Both reductions are ones already validated on the other two terms:
`lk_ball_params` -> a 16-bit atom type index (the ljlk fix, -2176 B) and
`water_coords` -> point at the device row rather than tiling, since they are
contiguous per tile (the elec charges fix, -3072 B). **7148 B -> ~1900 B**, in
line with ljlk's 1832 and elec's 1704. It should move lk_ball from a loss to a
win, and it helps whole-pose and rotamer-pair scoring alike.

## Its derivatives are not computed in the forward, unlike elec and ljlk

elec and ljlk fuse derivatives into the forward: with gradients requested they
run a derivative-capable kernel and return `dV_dcoords`, leaving the backward
a cheap weighting pass.

    elec forward -> {output_t, dV_dcoords_t, scratch_rot_neighbors_t}
    ljlk forward -> {output_t, dV_dcoords_t, scratch_rot_neighbors_t}
    lk_ball      -> {output_t, scratch_rot_neighbors_t}

lk_ball returns no derivatives and contains no reference to `compute_derivs`
or `require_gradient` at all. Its gradients come from a separate `backward`
dispatch that walks the block pairs a second time, and from the water
generator's own backward (`d_build_water_for_don` / `d_build_water_for_acc`),
since water positions depend on pose coordinates. So porting it is two more
kernels, not zero.

Measured on the host at 100 poses, forward against gradient-enabled:

| term | forward | fwd+bwd | difference |
|---|---|---|---|
| elec | 43.90 ms | 73.48 ms | 40.3% |
| ljlk | 55.30 ms | 67.75 ms | 18.4% |
| lk_ball | 20.05 ms | 41.52 ms | 51.7% |

Read the elec and ljlk numbers carefully: that difference is not a separate
backward kernel, it is the derivative-capable forward costing more than the
energy-only one plus a light backward. Only lk_ball's is a genuine second
pass.

## What follows for ordering

The packing phase -- rotamer-pair energies, 71% of a relax cycle -- needs
**energies only, no gradients**. So a forward-only lk_ball is fully useful
there, and its backward is not on the critical path for the large win. The
backward only unlocks lk_ball inside minimization, which is 12% of a cycle, of
which lk_ball is a part: worth a couple of per cent.

Ordering by value:

1. lk_ball tile shrink -- helps every lk_ball evaluation, forward and
   rotamer-pair, and is two fixes already proven twice.
2. Rotamer-pair scoring for elec, ljlk and lk_ball -- 62% of a cycle,
   forward-only, so it does not wait on lk_ball's backward.
3. lk_ball backward plus the water-generator backward -- a couple of per cent,
   and the last of the three.

---
---

# PLAN TO CARRY FORWARD — three steps, in order

Written to survive a compaction. Everything below is grounded in numbers
measured earlier in this file; re-read the relevant section before starting a
step rather than trusting the summary.

## State of the tree right now

Green: host build clean, all six shaders compile at MSL 3.2, `pytest
tmol/tests` = 3647 passed / 2143 skipped / 4 xfailed / 0 failed. ~221 files
modified, nothing committed.

Working on MPS today: elec, ljlk (whole-pose forward *and* backward), lk_ball
(whole-pose forward only), sphere_overlap block spheres + neighbour
detection, the compensated pose-total reduction.

**Half-finished, and the only loose end:** three context structs were added to
`tmol/score/common/sphere_overlap_ctx.hh` for step 2 --
`RotSpheresCtxT`, `BlockSpheresFromRotCtxT`, `RotPairListCtxT` (lines ~91,
~113, ~137). They compile but nothing uses them yet. Either build on them in
step 2 or delete them; do not leave them dangling.

## Why this order

A relax cycle measured on the host (`dev/mps_scratch/relax_full_profile.py`):
rotamer-pair energies 71%, build interaction graph 9.5%, minimization 12.4%,
SA 5.2%, build rotamers 1.6%. Minimization is already 2.32x on Metal, which
moves the whole cycle only 1.05x. Step 2 is where the cycle speeds up; step 1
is cheap and makes step 2 better; step 3 is worth a couple of per cent and
goes last.

---

# STEP 1 — shrink lk_ball's threadgroup tile (7148 B -> ~1900 B)

Smallest, highest confidence, helps whole-pose *and* rotamer-pair scoring.
Two patterns, both already applied twice in this tree; read the diffs for
ljlk (`tmol/score/ljlk/potentials/ljlk.hh`) and elec
(`tmol/score/elec/potentials/elec.hh`) first.

### Pattern A: `lk_ball_params1/2` -> 16-bit atom type index (-2176 B)

In `tmol/score/lk_ball/potentials/lk_ball.hh`:

1. `LKBallSingleResData`: replace
   `TMOL_SHARED_PTR(LKBallTypeParams<Real>) lk_ball_params` with
   `TMOL_SHARED_PTR(short) atom_types`.
2. `LKBallBlockPairSharedData`: replace
   `LKBallTypeParams<Real> lk_ball_params1[TILE_SIZE]` (and `2`) with
   `short atom_types1[TILE_SIZE]` (and `2`).
3. The loader (`lk_ball_load_block_coords_and_params_into_shared`, ~line 726)
   writes the index instead of copying the record. Note it has *two* branches
   -- a CPU `if constexpr` branch that assigns pointers and a GPU branch that
   calls `copy_contiguous_data`. Both change.
4. The readers index the device table instead:
   `type_params[score_dat.r1.atom_types[i]]`. They need the table as a new
   trailing template parameter + argument, exactly as ljlk's three readers
   got `TypeParamsView type_params`. Call sites are in
   `lk_ball_kernels.hh` and `lk_ball_pose_score.impl.hh`.
5. Every `r?.lk_ball_params = shared_m.lk_ball_params?` assignment becomes
   `atom_types`.

Use `short`, not `unsigned char`: there are 122 ljlk atom types today (no
headroom in a byte) and the loader relies on a negative "no atom here"
sentinel.

### Pattern B: `water_coords1/2` -> point at the device row (-3072 B)

`water_coords` is already contiguous per tile, and **the CPU branch of the
loader literally does this today**:

    r_dat.water_coords = reinterpret_cast<TMOL_DEVICE_PTR(Real)>(
        &water_coords[r_dat.rot_coord_offset + start_atom][0]);

So: change `LKBallSingleResData::water_coords` to `TMOL_DEVICE_PTR(Real)`,
delete `water_coords1/2` from the shared struct, delete the
`copy_contiguous_data<nt, MAX_N_WATER * 3>` call, and let both branches take
the CPU branch's assignment.

### The trap — read this before building

Both patterns bite in the same-tile intra-residue path, where only tile 1 is
loaded and r2's pointers are aliased to the "1" arrays. This segfaulted elec.
In `lk_ball_load_intrares_data_from_shared` (~line 1326) `water_coords` is
*already* aliased, so pattern B is covered -- but `atom_types` will need the
same:

    if (same_tile) intra_dat.r2.atom_types = intra_dat.r1.atom_types;

Also check `lk_ball_load_intrares2_tile_data_to_shared` (~line 1272), which
resets r2's pointers before loading.

### Validate and measure

    clang++ -std=c++17 -fsyntax-only -ferror-limit=4 -w \
      -DTORCH_TARGET_VERSION=0x020b000000000000 -DWITH_MPS \
      -DTMOL_INTEROP_NAMESPACE='"x"' -DTMOL_MAIN_MODULE \
      -I. -Itmol/extern -Igenerated \
      -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
      tmol/score/lk_ball/potentials/lk_ball_pose_score.cpu.cpp
    bash dev/mps_scratch/expand.sh \
      tmol/score/lk_ball/potentials/lk_ball_mps_kernels.metal.in /tmp/t.metal
    dev/mps_scratch/msl_probe /tmp/t.metal 32
    CMAKE_BUILD_PARALLEL_LEVEL=6 TMOL_BUILD_TESTS=ON pip install -e . --no-build-isolation
    python dev/mps_scratch/lk_ball_mps_bench.py 100 5        # energy + speed
    TMOL_MPS_TIME_KERNELS=1 python dev/mps_scratch/relax_steps.py 100 mps 5

Expect the lk_ball block near 1900 B and its 0.88x to become a win. Energies
must still agree with the host to ~1e-7. If it does become a win, reconsider
putting lk_ball back in `MPS_SCORE_TYPES` in `dev/mps_scratch/mixed_device.py`
-- but only after step 3, since minimization needs its backward.

---

# STEP 2 — rotamer-pair energies on Metal (the 62%)

Forward only; no gradients are needed for packing. Order: infrastructure,
then elec (cheapest, kernels already exist), then ljlk, then lk_ball.

### 2a. Infrastructure: four kernels plus a scan

The rotamer dispatch list is built in `tmol/score/common/sphere_overlap.impl.hh`
by three structs, none with an MPS path:

| struct | line | shape |
|---|---|---|
| `compute_rot_spheres` | 60 | one workgroup per rotamer |
| `compute_block_spheres_from_rot_spheres` | 1057 | one thread per (pose, block) |
| `rot_neighbor_indices_from_block_neighbors` | 1130 | count pass, scan, fill pass |

Follow the existing port exactly: extract each lambda body into
`sphere_overlap_kernels.hh` next to `compute_block_spheres_body` and
`detect_block_neighbors_body`, bind the contexts already added to
`sphere_overlap_ctx.hh`, add entries to
`sphere_overlap_mps_kernels.metal.in`, and add `*_mps_launch_if` helpers
beside the existing ones. Then an `if (D == tmol::Device::MPS)` branch in
each struct's `f()`, tested *before* the `forall_independent` call.

`compute_rot_spheres` uses `shuffle_reduce_and_broadcast_in_workgroup` on a
`Vec<Real,3>`; that now works at any threadgroup width (component-wise
scratch path), so it needs no special handling.

**The scan.** `rot_neighbor_indices_from_block_neighbors` calls
`DeviceDispatch<D>::scan<exclusive>` on raw pointers, which MPS cannot do --
an MPS tensor's data pointer is an MTLBuffer, and there is no cumsum in the
stable ABI. Do it on the host: copy `pair_counts` D->H, exclusive-scan on the
CPU, copy back. **This costs nothing extra** because the function already
synchronizes twice right there -- `pair_count_overflow_t.tensor.item<int64_t>()`
and `pair_counts_t.tensor.sum(...).item<int64_t>()`. The counts array is
n_poses * block_pairs_per_pose int32 = ~1.2 MB at 100 poses. Put it behind a
device-generic helper so CPU and CUDA keep `DeviceDispatch::scan` unchanged.

### 2b. elec — mostly wiring, the kernels exist

`elec_mps_kernels.metal.in` already declares and compiles:

    typedef ElecCtxMetal<float, int32_t, 2> RotCtx;          // line 32
    TMOL_ELEC_ENTRY(elec_eval_energies_by_block_f32, ..., RotCtx)
    TMOL_ELEC_ENTRY(elec_eval_derivs_f32, ..., RotCtx)

Missing: an MPS branch in `ElecRotamerScoreDispatch::forward`
(`elec_pose_score.impl.hh` line ~520; the launch is at the end, a
`foreach_independent_workgroup` over `dispatch_indices.size(1)`), a launch
wrapper, and swapping `elec_rotamer_score_op` in
`tmol/score/elec/potentials/compiled.ops.cpp` (~line 318) from
`TMOL_DISPATCH_FLOATING_DEVICE` to `TMOL_DISPATCH_FLOATING_DEVICE_MPS`.

Determinism is already fine here: the rotamer store writes
`ctx.output[0][cta]`, one cell per pair, no atomics. No cells-reduction
needed, unlike whole-pose.

### 2c. ljlk, then lk_ball

Same recipe as their whole-pose ports. Extract the rotamer kernel bodies,
add `RotCtx` typedefs and entries to each `.metal.in`, add launch branches to
`LJLKRotamerScoreDispatch::forward` (`ljlk_pose_score.impl.hh` ~line 1423) and
lk_ball's equivalent (~line 1416 returns `{output_t, dispatch_indices_t}`),
and flip the dispatch macros. **Do step 1 before lk_ball's rotamer port** so
it inherits the smaller tile.

The ordering lesson that cost the most time, three terms running: make the
loaders view-generic, **build immediately**, fix deducibility with a direct
single-TU clang invocation (ninja elides the candidate note inside lambda
substitution), build again, and only then touch Metal.

### Validate and measure

    python dev/mps_scratch/rpe_terms.py 10          # calcRPEs per term
    python dev/mps_scratch/relax_full_profile.py 100 20

Rotamer energies must match the host. Projection: if the three terms reach
the 2.6-3.0x their whole-pose kernels get, calcRPEs 49.2 s -> ~17 s at 100
poses and the cycle 69.2 s -> ~33 s, about **2.1x on a whole relax** against
today's 1.05x.

---

# STEP 3 — lk_ball's backward (and the water generator's)

Worth a couple of per cent; do it last, and only to let lk_ball join the
minimization composite.

lk_ball is the only one of the three whose forward does **not** compute
derivatives -- it returns `{output_t, scratch_rot_neighbors_t}` with no
`dV_dcoords`, and contains no `compute_derivs`/`require_gradient` at all.
Gradients come from a separate `backward` dispatch that walks the block pairs
again, plus the water generator's own backward (`d_build_water_for_don` /
`d_build_water_for_acc` in `water.hh`), because water positions depend on
pose coordinates. So this is two kernels.

Note `lk_ball_atom_derivs_full` in `lk_ball.hh` is currently guarded
`#if !TMOL_PORTABLE_METAL` -- it names `TView` directly. Making it
view-generic is the first task, the same edit the forward loaders got.

Use `accumulate_grad`, not `accumulate`, for every derivative accumulation,
or the gradient will be order-dependent and the line search will thrash --
this cost 60% more evaluations when it was missed on elec and ljlk.

Then add lk_ball to `MPS_SCORE_TYPES` in `dev/mps_scratch/mixed_device.py`
and re-measure `relax_steps.py` at 10/100/300.

---

# Standing rules, learned the hard way

- **Build flags do not reach the shader.** `cmake/TmolMetalShader.cmake`
  flattens with its own fixed `clang -E`; `CXXFLAGS` reaches the host build
  only. Any knob a kernel must see goes in a header as a literal. A probe
  passed through `CXXFLAGS` silently measured nothing and nearly sent this
  work down the wrong path.
- Keep `CMAKE_BUILD_PARALLEL_LEVEL=6` on every build.
- A discarded `if constexpr` branch is still *parsed* on Metal, so pointer
  types inside it still need an address space.
- Check the same-tile intra-residue aliasing after any tile change.
- Shared-memory size is what bounds these kernels: +4 KB cost ljlk 39%,
  +12 KB cost 148%. Use the padding probe (as a header literal) to decide
  before optimizing anything else.
- `TMOL_MPS_TIME_KERNELS=1` gives a per-kernel breakdown; it synchronizes per
  launch, so read it as relative cost.
- CUDA remains unverified -- no nvcc on this machine. `accumulate_grad`,
  `WorkgroupLane`, the portable layer and the per-function explicit template
  arguments all touch CUDA paths.
- The change set is large and uncommitted. Committing the portable layer
  separately from the per-term work would let a CUDA regression bisect.

---

# The three steps, implemented

All three landed. `pytest tmol/tests` = **3647 passed, 2143 skipped, 4
xfailed, 0 failed** -- identical to the baseline before this work.

## STEP 1 -- lk_ball's threadgroup tile: 7148 B -> 1772 B

Both tiled arrays turned out to need the *same* treatment, not the two
different patterns the plan predicted. `block_type_tile_lk_ball_params` is
already indexed `[block_type][tile][i]`, so its tile is a contiguous device
row exactly like `water_coords` -- there was no reason to tile the atom type
and re-index a global table the way ljlk had to. Both fields became
`TMOL_DEVICE_PTR` and both staging arrays went away (-3072 B of waters,
-2304 B of parameters), which also deleted two `copy_contiguous_data` calls
per tile load.

The CPU branch of the loader was already doing this; the change is that every
backend now takes that branch, and the GPU branch shrank to coordinates only.
The same-tile intra-residue aliasing the plan warned about is now identical on
all three backends, which is one fewer thing to get wrong.

One supporting change: `coord_from_shared` needed a `device`-address-space
overload (`tmol/score/common/data_loading.hh`), guarded `#if
TMOL_PORTABLE_METAL` because on CPU and CUDA the two pointer spellings are
the same type and it would be a redefinition.

    100 x 1UBQ, whole-pose lk_ball forward
      cpu   20.06 ms
      mps   13.51 ms     0.88x -> 1.49x, energies agree to 1.3e-07

## STEP 2 -- rotamer-pair energies on Metal

### Infrastructure

Four new kernel bodies in `sphere_overlap_kernels.hh`:
`compute_rot_spheres_body`, `compute_block_spheres_from_rot_body`,
`rot_pair_counts_body`, `rot_pair_fill_body`, plus `rot_spheres_overlap_view`
and the shared candidate-decoding prologue.

Unlike the two kernels ported earlier -- whose host launchers still carry a
second copy of the body as a device lambda -- these four are called from both
paths. The generic launcher builds a TView-backed context and passes it to the
same function the Metal entry point calls, so the body exists once. The three
contexts left dangling in `sphere_overlap_ctx.hh` are now used;
`RotPairListCtxT` gained a tenth view for the overflow flag.

**The scan.** `DeviceDispatch::scan` takes raw pointers, which an MPS tensor
cannot provide. It runs on the host for that backend, inside
`rot_neighbor_indices_from_block_neighbors::exclusive_scan_counts`. As
predicted this costs no extra synchronization: the caller has already blocked
on the overflow flag and on the count sum two lines above.

### Per term

| term | change |
|---|---|
| elec | launch branch + `TMOL_DISPATCH_FLOATING_DEVICE_MPS`; the `RotCtx` entry already existed |
| ljlk | new `ljlk_eval_rotamer_energies_by_block` body, `RotCtx` typedef and entry, `dispatch_indices` added to `LJLKCtxT` (22 -> 23 views) |
| lk_ball | new `lk_ball_eval_rotamer_energies_by_block` body and a ctx-based `STORE_ROTAMER_CALCULATED_ENERGIES`; `dispatch_indices` and `block_ind_for_rot` added to `LKBallCtxT` |

Each needed its `*RotamerScoreDispatch` explicitly instantiated in the
`.mps.cpp` translation unit -- the dispatch macro alone does not pull it in,
and the omission shows up as a link-time missing symbol at import.

### Correctness

Rotamer-pair tables agree with the host (`dev/mps_scratch/rpe_mps.py`), with
one honest caveat worth recording:

    100 x 1UBQ, 3.59M rotamer atoms
      elec     63,314,200 shared pairs   rel 9.2e-07   cpu-only 100, mps-only 0
      ljlk     67,981,600 shared pairs   rel 4.2e-07   cpu-only 0,   mps-only 100
      lk_ball  67,981,600 shared pairs   rel 5.5e-07   cpu-only 0,   mps-only 100

The dispatch lists differ by about one pair in 600,000. **Every pair in the
difference carries exactly zero energy** -- they sit on the bounding-sphere
cutoff, where the backends disagree because the CPU takes the square root
before the reduction and the parallel backends after. The same divergence
exists between CPU and CUDA; it is a property of the cutoff test, not of this
port.

### Speed

    100 x 1UBQ, rotamer-pair energies
      term      cpu        mps      speedup
      elec     21.2 s     2.59 s      8.2x
      ljlk     35.7 s     3.54 s     10.1x
      lk_ball   9.7 s     3.79 s      2.6x
      total    66.7 s     9.92 s      6.7x

**What this does not yet buy.** `pack_rotamers` still cannot run against an
MPS pose stack: `PackerPalette.block_types_from_original` asserts, and the
rotamer builder, the interaction graph and the annealer have no Metal
backends. The kernels above are the calcRPEs work and are exercised on real
rotamer sets through `render_rotamer_scoring_module`, but turning 6.7x on that
phase into 6.7x on that phase *of a packing run* needs the packer's setup
layer to accept a device pose stack. That is a separate piece of work and was
not part of these three steps.

## STEP 3 -- lk_ball's backward, and the water generator's

Two kernels, as expected.

`lk_ball_atom_derivs_full` lost its `#if !TMOL_PORTABLE_METAL` guard: the two
output tensors became template view parameters, and its two inner lambdas
needed `TMOL_THREAD` on their reference parameters. Its accumulations use
`accumulate_grad`, so the gradient is reproducible -- verified bit-identical
across three runs at 20 poses.

`lk_ball_eval_derivs` joins the two energy bodies in `lk_ball_kernels.hh`;
`LKBallCtxT` grew `dTdV`, `dV_d_pose_coords`, `dV_d_water_coords` and a
`block_pair_scoring` scalar, reaching 24 views.

The water generator's backward needed its **own** context.
`WaterGenCtxT` already binds 30 views, which with the metadata buffer is
Metal's 31-binding limit, and the backward needs two accumulators the forward
does not. `WaterGenDerivCtxT` drops `water_coords` and folds the three
torsion tables into one `[3, max_len]` table whose row lengths travel as
launch scalars, landing at 29 views. The packed table is built in the
backward op wrapper with `torch::stable::copy_`, where the tables are still
Tensors rather than views, and only when the coordinates are on MPS.

    5 x 1UBQ, whole-pose lk_ball
      energy      cpu -30.549126   mps -30.548950
      gradient    max abs 1.1e-03, rel 3.4e-04, no NaN

The gradient is coarser than elec's or ljlk's because `accumulate_grad` snaps
every contribution to a 1/8192 grid and lk_ball accumulates many small ones.
That is the determinism trade already accepted for the other two terms.

### Three bugs worth remembering

1. **An unread scalar reads as garbage.** `block_pair_scoring` was added to
   the context and set by the launcher, but `lk_ball_build_metal_ctx` was
   never taught to read it back out of the metadata. The kernel then indexed
   a per-pose `dTdV` as if it were per-block-pair, walked off the end, and
   produced NaN for *every* real water -- while leaving the zeros exactly
   right, which is what made it look like a math bug. **Adding a scalar slot
   means editing three places: the context, the launcher, and the builder.**

2. **Metal wants a 16-byte-aligned buffer offset.** An upstream gradient is a
   slice of the score function's concatenated per-term scores, so its data
   pointer carries that slice's offset -- for lk_ball, 8 bytes past a
   16-byte boundary. Binding it faults inside `objc_retain` in the Metal shim
   with no error return. `tmol::native::realigned_copy` fixes it at the
   source, and `mps_launch` now *checks* the alignment and throws a message
   naming the kernel and the view instead of crashing. Only lk_ball hits this
   today, because elec's and ljlk's whole-pose backwards are still
   `TMOL_DISPATCH_FLOATING_DEVICE` -- their derivatives come out of the fused
   forward.

3. **Encoding is not thread-safe.** `start_encoding`, the argument binds and
   the dispatch act on one command encoder owned by torch's MPS stream.
   `mps_launch` now holds a mutex across all three. This was not the cause of
   the crash above -- single-threaded autograd still crashed -- but it is a
   real hazard the moment a backward graph holds two Metal terms.

## Where this leaves the relax cycle

Fixed-step torsion relax, 20 optimizer steps, per energy evaluation:

    poses    host only    ljlk+elec on Metal    + lk_ball on Metal
       10      27.5 ms           16.9 ms              26.7 ms
      100     229.7 ms           92.5 ms             118.0 ms
      300     627.1 ms          243.7 ms             345.3 ms

**Adding lk_ball to the minimization composite makes it slower**, and
`MPS_SCORE_TYPES` in `dev/mps_scratch/mixed_device.py` has been left at the
four terms that measure fastest. The reason is lk_ball's *backward*, not the
composite's balance:

    100 x 1UBQ, whole-pose, median of 5 (ms)
      term         dev   forward   backward   total
      ljlk+elec    cpu     125.9        4.5   130.4
      ljlk+elec    mps      43.0        2.1    45.1
      lk_ball      cpu      21.5       21.9    43.4
      lk_ball      mps      13.8       36.4    50.2

lk_ball's forward is 1.56x faster on Metal and its backward is 1.7x slower,
so the term costs more on the GPU than on the host. Moving it makes the GPU
pole 45.1 + 50.2 = 95.3 ms, which overtakes the roughly 92 ms host pole, and
the composite flips from host-bound to GPU-bound with a longer critical path
than it started with.

Per-kernel, with `TMOL_MPS_TIME_KERNELS=1` (synchronized per launch, so read
the ratios rather than the absolute times):

      lk_ball_eval_derivs_f32              27.8 ms/call
      lk_ball_eval_energies_by_block_f32   17.5
      water_gen_build_water_derivs_f32      8.9
      water_gen_build_waters_f32            0.6

Both derivative kernels were priced by rebuilding with the atomic add
replaced by a plain one -- wrong results under contention, but it separates
contention from arithmetic. Neither is contended:

| kernel | with atomics | without | verdict |
|---|---|---|---|
| `lk_ball_eval_derivs_f32` | 27.8 ms | 31.1 ms | not atomics |
| `water_gen_build_water_derivs_f32` | 8.9 ms | 9.0 ms | not atomics |

**Read the next paragraph before trusting any experiment of this shape.**
The first version of the water-generator measurement reported 0.43 ms and
"95% atomic contention". It was wrong. The stand-in accumulator was written
as a template taking `T target` *by value*, so it was handed a copy of the
`Vec<Real,3>` rather than a reference into `dE_d_pose_coords`; every write
was dead, and the compiler deleted the derivative chain feeding it. The
kernel being timed was nearly empty. Writing `dE_d_pose_coords[ind][i] +=
...` directly, through the view, gives the 9.0 ms above. **A non-atomic
stand-in has to write through the same reference the real one does, or it
measures an empty kernel.**

What actually costs the time, then, is arithmetic and the registers it needs:

- `d_build_water_for_don` and `d_build_water_for_acc` run a full derivative
  chain through `build_coordinate::dV` where the forward runs only `::V`.
  That is most of the forward-to-backward gap in the water generator.
- *Registers.* Metal reports `maxTotalThreadsPerThreadgroup` of 384 for the
  water-generator backward against 896 for its forward, and 384 for the
  scoring derivatives against 512 for the energy kernel -- at identical
  threadgroup memory in both pairs (592 B and 1776 B). So the derivative
  kernels need 2.33x and 1.33x the registers and get correspondingly fewer
  resident threadgroups to hide latency with. `weighted_dV` is the worst
  offender: it keeps `dI`, `dJ` and two 4x3 water matrices live, plus
  `frac_desolv` and `frac_bridge` with their own, on the order of 77 live
  floats against a handful for `weighted_V`.

The CPU shows neither effect -- it has a stack and a cache hierarchy -- so
its backward costs about what its forward does (24.5 vs 22.4 ms) while
Metal's is 2.7x its forward.

### A fix that was tried and reverted: one thread per acceptor

Written, measured, and backed out. An acceptor's waters all derive from the
same three atoms, so giving one thread all of them lets their contributions
sum in registers, turning twelve-or-more adds per acceptor into three. It is
correct -- the tests pass and the gradient is unchanged -- and it also hoists
the bond walk that resolves the base atoms out of the per-water work.

It is 2.4x slower: **8.9 ms -> 21.3 ms** per call.

The reason is that it trades away parallelism to buy a reduction in
contention that was not there. Work units per tile drop from
`n_donH + n_acc * MAX_N_WATER` to `n_donH + n_acc`, which for a 32-atom tile
leaves most of a 32-wide threadgroup idle while the few active lanes run up
to four `build_coordinate::dV` calls back to back. Re-running the non-atomic
experiment on that version confirms it: 21.3 ms with atomics, 22.1 without.
Same conclusion, from the other side.

Reordering the per-water index mapping so that adjacent lanes take different
acceptors does not help either, and is worth not trying: the four waters of
one acceptor are concurrent whichever lanes they land on.

This is not an argument against the step 3 work: lk_ball's backward is what
makes the term usable end to end on Metal, and lk_ball's kernels are where
the *rotamer-pair* speedup comes from, where no backward runs at all. It does
say where the next work is. The lever is register pressure, not contention:

  - Cut the live state in `weighted_dV` and in the water derivative
    builders. Accumulating each contribution into the output as it is
    computed, rather than building full `dWI`/`dWJ` matrices and returning
    them in one struct, is what would buy back resident threadgroups. The
    occupancy numbers bound the prize: 384 -> 896 max threads would be
    2.3x on the water-generator backward.
  - Note this argues *against* naively fusing lk_ball's derivatives into
    its forward the way elec and ljlk do. Fusion removes the second
    traversal but moves all of that live state into the kernel that is
    currently the fast one.
  - Only then look at moving more host terms across, hbond first.

At 10 poses nothing helps: the work per launch is too small for the GPU to
matter, and that has been true at every stage of this port.

---

# Contracting `build_coordinate::dV`: tried, measured, reverted

Written, verified bit-identical, measured to do nothing, and backed out. The
reasoning is recorded because the probe that motivated it was misread, and
that mistake is easy to repeat.

## The probe, and how it misled

`dev/mps_scratch/msl_occupancy.mm` (kept) reports each Metal entry point's
`maxTotalThreadsPerThreadgroup`, which the compiler derives from the kernel's
register and threadgroup-memory needs. Compile-only: no build, no run. It
said:

| variant of `build_coordinate::dV` | max threads |
|---|---|
| baseline, 27 outputs | 384 |
| `dgp`/`dggp` zeroed, 9 outputs | 576 |
| also `build_don_water::dV` cut to 9 | 576 |

Read as "27 live outputs cost 1.5x the occupancy". **That reading was wrong.**
Zeroing `dgp` and `dggp` does not remove eighteen result slots, it removes
eighteen whole *expressions* -- and with them a large share of the
subexpression web the symbolic derivation shares across all twenty-seven. The
probe measured less arithmetic, not fewer live results.

**A probe that deletes work cannot price storage.** This is the second
measurement in this file to fail that way; the other is the non-atomic
stand-in that was handed its target by value. Both produced a large, exciting
number by accidentally deleting the kernel's work. Before believing a probe
that reports a big win, check what it removed.

## What was built, and what it measured

`build_coordinate::dV_terms` -- the twenty-seven expressions written once and
emitted one at a time through three callbacks -- with `dV()` storing them into
matrices and a new `dV_dot(..., v)` contracting each against `v` as it was
produced. Both hot callers (`d_build_water_for_acc`, and the leaf-atom
backward in `io/details/compiled/gen_pose_leaf_atoms.impl.hh`) took the
contracted form; neither ever wanted the matrices.

It was bit-identical, not merely close -- the contraction accumulates down the
inner index in ascending order, which is what `operator*(Mat, Mat)` in
`utility/portable/matrix.hh` does. CPU energy -30.549126 and gradient max-abs
1.092e-03 / rel 3.431e-04, unchanged to every printed digit, 693 tests green.

| | before | after |
|---|---|---|
| `water_gen_build_water_derivs_f32` | 26.5 ms / 3 calls | 27.3 |
| max threads per threadgroup | 384 | **384** |
| lk_ball backward, MPS, 100 poses | 36.8 ms | 36.5 |
| lk_ball backward, CPU, 100 poses | 21.9 ms | 21.8 |

Noise in every column. The twenty-seven expressions have to be evaluated
either way, and what bounds the kernel is their shared subexpressions, not the
slots their results land in.

## If anyone reconsiders this for CUDA

There was one argument left for keeping it, and it is worth writing down so
the next person can weigh it rather than rediscover it:

- Metal applies no cap. The compiler picked a register count, reported 384
  max threads, and ran at lower occupancy. Nothing spilled, so removing
  eighteen live floats changed nothing.
- CUDA's derivative kernels carry `LAUNCH_BOX_32_OCC(16)`, whose comment
  already calls them "latency- and register-bound". That is a
  `__launch_bounds__`, a *hard* ceiling of roughly 128 registers per thread
  at 16 warps, and past it nvcc spills to local memory instead of lowering
  occupancy. Eighteen fewer live values is around 14% of that budget.

That remains untested -- there is no nvcc on this machine. It was backed out
rather than carried on speculation: the emitter layer is not worth having for
a redundant matrix-vector product, and reinstating it is a small, mechanical
change if a CUDA profile ever justifies it. The occupancy targets were left
alone throughout; re-tuning `OCC(16)` is a separate experiment with its own
register trade.

## Where that leaves the backward

Unchanged from the measurements above: neither derivative kernel is
atomic-bound, and the register pressure in the water generator is not
addressable by reshaping `dV`'s outputs. The untried lever is `weighted_dV`
in the scoring derivative kernel, where zeroing the returned `dWI`/`dWJ`
moved max threads 384 -> 512, exactly matching the energy kernel. That one is
a real redesign -- the water matrices cannot be contracted, because each
element lands in its own slot of `dV_d_water_coords`, so it means scaling and
writing them out as they are produced and paying a second pass over the
waters inside `lk_fraction::V_dV` and `lk_bridge_fraction::V_dV`. Note the
same probe caveat applies: zeroing those outputs also deleted the work that
produced them, so 512 is an upper bound on what the restructure could
recover, not a prediction.

---

# The real reason the backward was slow: IEEE division on Metal

This is the largest single find of the port, and it was sitting under two
failed optimizations. Read this before optimizing any Metal kernel.

## The measurement that found it

The whole-term numbers confounded tile walking with math, so the water
generator was timed on its own -- one job, no block pairs, nothing else in the
kernel:

| device | forward | backward | ratio |
|---|---|---|---|
| cpu | 3.39 ms | 5.97 ms | **1.8x** |
| mps | 0.51 ms | 8.74 ms | **17.0x** |

Same source, same algorithm, one backend 9.4x worse in the *ratio*. The
forward is 6.6x faster on Metal and the backward 1.5x slower. That is not
"GPUs expose arithmetic the CPU hides" -- a 9x discrepancy in a ratio is a
defect, and naming it that way is what led to the cause.

`dev/mps_scratch/wgsplit.py` is that harness; keep it. Isolating one kernel
pair was worth more than three rounds of probing the composite.

## The cause

`build_coordinate::dV` is 576 lines of machine-generated symbolic derivative
that shares nothing between its twenty-seven expressions. It contained:

- **135 copies** of the identical loop-invariant `sqrt(abxcb2 / a_m_b2)`
- **250 divisions**, every one by a product of loop invariants -- about nine
  distinct values (`a_m_b`, `a_m_b2`, `a_m_b4`, `norm_abxcb`,
  `abxcb2 * norm_abxcb`, and products of those)

**Metal compiles with `MTLMathModeSafe` by default**, so each `/` is an
IEEE-correct division: a multi-instruction reciprocal-plus-refinement
sequence, not one op. The host has a hardware divider and out-of-order
execution and barely noticed. The GPU paid for all 250, per water.

And there is no flag to reach for. `aoti_torch_mps_create_shader_library`
takes only the source string -- no `MTLCompileOptions`, so fast math cannot
be enabled from here. **It has to be fixed in the source.**

## The two fixes, separately measured

**Hoisting the 135 sqrts** into one `norm_abxcb`:

| | before | after |
|---|---|---|
| water-gen op backward, cpu | 5.97 ms | **4.63** |
| water-gen op backward, mps | 8.74 ms | 9.05 |

So the *host* compiler was not fully eliminating them (22% there) and Metal's
already was. Worth keeping, but not the answer.

**Hoisting the reciprocals** of the nine invariant divisors -- 212 divisions
became multiplies:

| | before | after |
|---|---|---|
| `water_gen_build_water_derivs_f32` | 8.9 ms/call | **0.8** |
| water-gen op backward, mps | 8.74 ms | **2.33** |
| mps backward/forward ratio | 17.0x | **2.6x** (cpu 1.4x) |

**11x on the kernel.** The asymmetry that started this is essentially gone.

Reciprocal-then-multiply is not bit-identical to division -- it is accurate
to about one ulp, far below the 2^-13 grid the Metal gradient is snapped to
anyway. The host energy stayed bit-identical at -30.549126 and the gradient
metrics unchanged to four digits.

## The audit

Every Metal translation unit was flattened and its divisions attributed per
function (`#include` graph walked from the six `.metal.in` files). After the
fix, nothing Metal-reachable carries a comparable load:

| divisions | file | function |
|---|---|---|
| 29 -> 10 | gen_coord.hh | `build_coordinate::dV` |
| 9 -> 1 | geom.hh | `distance::V_dV` |
| 8 | tile_atom_pair_evaluation.hh | `n_remain2` (integer) |

Two more were worth fixing by hand:

- `distance::V_dV` returned `{V, delta / V, -delta / V}` -- six scalar
  divisions, and it runs **per atom pair** in the derivative kernels. Now one
  reciprocal.
- `interior_angle::V_dV` divided twice by each of two norms; now two
  reciprocals.
- The nineteen divisions left in `build_coordinate::dV` were by three more
  invariant products; hoisted too.

The scoring derivative kernel's own helpers are clean -- `lk_fraction::V_dV`
has 4 divisions, `lk_bridge_fraction::V_dV` 7, `lk_isotropic_pair::V_dV` 1,
`lj::V_dV` 1. So `lk_ball_eval_derivs_f32` is genuine arithmetic, not this
pathology, and it is now what remains of the backward.

## The standing rule

**A division in a Metal device function is not one instruction.** Under the
default math mode it is an IEEE-correct sequence. Generated or
symbolically-derived code is the dangerous case, because it divides by the
same invariant dozens of times and no compiler on this path will hoist it for
you. When a Metal kernel is slower than its host counterpart and the
arithmetic looks comparable, count the divisions before theorising.

The corollary for this project's design goal: the fix is device-generic. One
reciprocal is at least as good as one division on CPU and CUDA -- the host
gained slightly as well -- so none of this needed a backend guard.

---

# hbond's double precision: what actually needs it

hbond evaluates degree-10 polynomials through `bound_poly<11, double>`, and
Metal has no `double` at all. Before porting anything, the question was
measured on the real coefficients -- all 58 of them from the database, 193
points across each one's own range, against a float64 Horner reference:

| scheme | max rel error |
|---|---|
| float32 coeffs, float32 x | 1.96 |
| float32 coeffs, **float64** x | 1.96 |
| **float64** coeffs, float32 x | **2.2e-03** |

**The coefficients need the precision; the argument does not.** Rounding x to
float32 costs nothing measurable, and the whole 1.96 relative error -- a
200% error, at points where the polynomial nearly cancels -- comes from
rounding the *coefficients*. That is worth knowing because it says the fix
lives entirely inside the polynomial evaluation and does not touch how
distances or cosines are computed.

Two schemes were tried, both in float32 with coefficients split into an
unevaluated hi + lo float32 pair:

| scheme | max abs | max rel |
|---|---|---|
| plain float32 Horner | 8.34e+00 | 1.960 |
| compensated Horner (Graillat et al.) | 7.32e+00 | **2.201e-03** |
| double-float accumulator | 7.32e+00 | **2.201e-03** |

Both land on *exactly* the float64-coefficient number, 2.201e-03, so both are
optimal: the residual is the floor set by float32 x, not by the scheme. An
890x improvement over plain float32, and enough -- the error it was rejected
for was 0.19 kcal/mol on a total.

`poly<POrd, Real, EvalReal>` in score/common/polynomial.hh **already** has the
seam: the accumulator type is a separate template parameter from the
coefficient type, which is why hbond can say `bound_poly<11, double>` today.
A portable double-float type slots into `EvalReal` with no backend guard. The
coefficient *storage* does have to change on Metal, because a float64 tensor
cannot be read there: the host splits each coefficient into hi/lo float32
once at setup.

**What does not work:** recentring the polynomials on their own range,
`t = (x - mid)/half`, which is the usual conditioning fix. These are fitted
polynomials with values in the thousands and heavy cancellation, so the
monomial coefficients in t get *larger*, not smaller -- 4.59e+04 before,
6.34e+06 after -- and the error gets worse (max rel 13.7). A Chebyshev basis
with a Clenshaw recurrence would condition properly, but it changes the
kernel's evaluation and its derivative, where the double-float accumulator
changes neither.

---

# Relax cycle, measured: 1.55-1.67x

The whole four-operation cycle, `dev/mps_scratch/relax_mixed_full.py`, 20
fixed minimization steps, 43 energy evaluations in both columns (so this is
work per evaluation, not a different trajectory).

| poses | host | ported terms on Metal | speedup |
|---|---|---|---|
| 10 | 8268 ms | 4948 | **1.67x** |
| 30 | 21927 | 13837 | **1.58x** |
| 100 | 71143 | 44370 | **1.60x** |
| 300 | 207220 | 133401 | **1.55x** |

Stage breakdown at 100 poses:

| stage | host | mixed | |
|---|---|---|---|
| build rotamers | 1180 ms | 800 | host-only |
| calcRPEs | 51430 | 29810 | 1.73x |
| build interaction graph | 6510 | 6550 | host-only |
| run SA | 3690 | 3760 | host-only |
| minimization | 8279 | 3388 | 2.44x |
| **CYCLE** | **71143** | **44370** | **1.60x** |

## Two fixes this needed

**The fused LJ/LK+Elec rotamer op has no Metal kernel**, and the score
function was selecting it over the two separate rotamer dispatches that are
ported, so the mixed path raised on the first call. MPS now opts out of that
fusion in `RotamerScoringModule.__init__`, keyed on `weights.device` -- the
same shape as the existing `_shares_block_neighbors` opt-out. A library fix,
not a harness one.

**Merging the two devices' tables.** The packer consumes one sparse table.
Sparse-*adding* the halves capped calcRPEs at 1.38x, because addition
coalesces and the packer coalesces again immediately afterwards.
Concatenating the indices and values and letting the downstream pass do the
merge took it to 1.73x.

Also tried and reverted: `forward_sparse_entries` on both halves, to keep
int32 indices and halve the bytes crossing the bus. Slower -- 4220 ms against
3180 at 10 poses -- because the host half's `__call__` runs
`_try_coalesce_cpu_rotamer_layouts`, which merges duplicate layouts before
the packer's pass, and bypassing it hands the packer more entries than the
int64 promotion costs.

The rotamer split is deliberately not the minimization split
(`MPS_RPE_TYPES` against `MPS_SCORE_TYPES`): lk_ball's rotamer kernel belongs
on the device, because packing runs no backward, while in minimization it
still lengthens the GPU pole past the host's.

## The ceiling, and what lifts it

Of the 44.4 s mixed cycle at 100 poses, roughly **47% is host work with no
Metal path at all**:

- build interaction graph 6.55 s + SA 3.76 s + build rotamers 0.8 s = 11.1 s
  (25%). The annealer and the graph builder have no kernels, and PackerTask
  cannot even be constructed against an MPS pose stack.
- hbond's rotamer energies, about 19% of calcRPEs by the earlier
  leave-one-out, so roughly 9.8 s (22%).

hbond is the tractable half, and its blocker is now solved on paper: see the
polynomial-precision section above. A double-float accumulator through
`poly`'s existing `EvalReal` seam reaches float64-coefficient accuracy
(2.2e-03 relative, the floor set by float32 x) with no backend guard. That
port -- the polynomial type, then the hbond rotamer kernel -- is the next
work, and it is worth about 22% of the cycle.

## Landed: the portable double-float accumulator

`tmol/utility/portable/double_float.hh` -- `DoubleFloat<Real>`, an
unevaluated hi + lo pair with the three operations a Horner recurrence needs
(df x plain, df + plain, df + df), built on Knuth's two-sum and an fma-based
two-product. `tmol::portable::math::fma` was added alongside it, via the
existing `TMOL_MATH_TERNARY` pattern, so the two-product stays exact even
where contraction is permitted.

Verified in C++ against float64 on the worst of the 58 database polynomials,
`hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6` at x = 1.3268:

| | value | rel error |
|---|---|---|
| float64 reference | 0.0005214686 | -- |
| plain float32 Horner | **-0.0007888372** | 2.51 |
| double-float float32 | 0.0005224819 | **1.9e-03** |

Plain float32 returns the **wrong sign**. The double-float lands on the
predicted floor. Compiles for the host and for Metal; the Metal probe reports
1024 max threads per threadgroup for a double-float degree-10 Horner, so the
accumulator is register-cheap and will not constrain whatever kernel uses it.

**Not yet done, and the relax time is therefore unchanged at 1.60x:** hbond
itself is not ported. What remains is

1. coefficient storage -- the host splits `HBondPolynomials<double>` into
   hi/lo float32 once at setup, because a float64 tensor cannot be read on
   Metal. This touches the four `TView<HBondPolynomials<double>, 2, Dev>`
   signatures in hbond_pose_score.hh and the packing in
   score/hbond/_params.py.
2. threading `DoubleFloat<float>` through `poly`'s `EvalReal` parameter for
   the Metal instantiation, and a two-vector coefficient overload.
3. the hbond rotamer kernel on Metal: ctx, kernel body extraction, entry
   point, launch wrapper, `TMOL_DISPATCH_FLOATING_DEVICE_MPS`, and the
   explicit instantiation in a `.mps.cpp` -- the same six-part pattern the
   other three terms took.

Step 3 is the bulk of it and the part with no numerical risk left.

## Step 2 landed: the poly seam widened

`poly` and `bound_poly` in score/common/polynomial.hh now take three types
where they took two:

    template <int POrd, typename Real, typename EvalReal = Real,
              typename CoeffReal = Real>

`Real` is the argument and the result, `EvalReal` the Horner accumulator,
`CoeffReal` the stored coefficients. All three default to `Real`, so
`poly<11, double>` means exactly what it meant before and the CPU and CUDA
instantiations are bit-unchanged -- the only edit to those paths is that the
coefficient vector is now spelled `Vec<CoeffReal, POrd>`, which resolves to
the same type.

Metal instantiates `bound_poly<11, float, DoubleFloat<float>,
DoubleFloat<float>>`: float in, float out, coefficients and accumulator
carried as hi + lo pairs. Verified to compile and build a pipeline, V and
V_dV both, at 1024 max threads per threadgroup.

`DoubleFloat` gained an `explicit operator Real()` so `poly`'s closing
`(Real)v` cast resolves; explicit so a double-float cannot silently collapse
to one float anywhere else.

## Step 1 remains, and its design is settled

`sizeof(DoubleFloat<float>) == sizeof(double)`, so
`HBondPoly<DoubleFloat<float>>` has the same layout footprint as
`HBondPoly<double>` -- 16 slots of 8 bytes -- which makes the storage swap a
change of element type rather than of shape. What it needs:

1. A trait beside the structs, `hbond_poly_scalar<Device>`, resolving to
   `double` everywhere and `DoubleFloat<float>` on MPS. The four
   `TView<HBondPolynomials<double>, 2, Dev>` signatures in
   hbond_pose_score.hh then spell
   `TView<HBondPolynomials<typename hbond_poly_scalar<Dev>::type>, 2, Dev>`
   and the CPU/CUDA instantiations resolve to what they are today.
2. An `enable_tensor_view` specialization for
   `HBondPolynomials<DoubleFloat<float>>` -- 96 float32 slots where the
   double version has 48 float64 ones.
3. `pack_polynomial` in score/hbond/_params.py emitting hi/lo float32 pairs
   in the struct's field order for the MPS device, the float64 tensor
   otherwise.

Then step 3, the hbond rotamer kernel itself, is the established six-part
pattern with no numerical risk left in it.

## Step 1 landed: hbond coefficient storage

Three pieces, and the CPU/CUDA paths come out unchanged:

- **`hbond_poly_scalar<Device>`** in params_types.hh, with
  `HBondPolysFor<D>` as the alias the signatures use. Resolves to `double`
  everywhere and `DoubleFloat<float>` on MPS. The eleven
  `TView<HBondPolynomials<double>, 2, Dev>` signatures across
  hbond_pose_score.hh, hbond_pose_score.impl.hh and hbond.hh became
  `TView<HBondPolysFor<Dev>, 2, Dev>`.
- **`enable_tensor_view<DoubleFloat<Real>>`**, two primitive slots, plus a
  correction to the two hbond struct specializations: they now count slots in
  `sizeof(PrimitiveType)` rather than `sizeof(Real)`. Identical for `double`
  (128/8 = 16); correct for a double-float (128/4 = 32).
- **potentials.hh templated on the coefficient type.** Its eight helpers took
  `HBondPoly<double> const&` and named `bound_poly<11, double>`; they now
  take `HBondPoly<PolyReal> const&` and `bound_poly<11, Real, PolyReal,
  PolyReal>`, with `PolyReal` deduced -- so not one call site changed.
- **`bound_poly`'s range and bound** now share `CoeffReal`, because they are
  stored beside the coefficients in the same struct. Converted to `Real` once
  at the comparison, which is where the explicit `operator Real()` on
  DoubleFloat earns its keep.
- **`pack_polynomial` in score/hbond/_params.py** splits the assembled
  float64 table into hi/lo float32 pairs for MPS only, and
  `pair_poly_table`'s annotation was loosened from `Tensor[torch.float64]`
  since the trailing dimension doubles there.

Verified: the table is `(12, 14, 48)` float64 on the host, exactly as before,
and `(12, 14, 96)` float32 on MPS. hbond, lk_ball and cartbonded tests pass.

**Step 3 -- the hbond rotamer kernel itself -- is not started.** The relax
time is therefore still 1.60x at 100 poses: nothing hbond does runs on Metal
yet. What remains is the six-part pattern the other three terms took, and it
now has neither numerical nor storage risk left in it:

1. `hbond_ctx.hh` / `_ctx_metal.hh` / `_ctx_tensor.hh` -- the field list once,
   templated on the view.
2. extract the rotamer kernel body out of
   `HBondRotamerScoreDispatch::forward` into `hbond_kernels.hh`.
3. `hbond_mps_kernels.metal.in` with the entry point.
4. `hbond_mps_launch.hh`.
5. flip `hbond_rotamer_score_op` to `TMOL_DISPATCH_FLOATING_DEVICE_MPS`.
6. explicit instantiation in `hbond_pose_score.mps.cpp` -- which does not
   exist yet, so it needs adding to CMakeLists.txt alongside the other
   `.mps.cpp` files.

Watch for: hbond's threadgroup block is the one unmeasured tile. Run the
padding probe through `msl_occupancy` before assuming it fits -- lk_ball's was
7148 bytes until it was shrunk, and shared memory is what bounds these
kernels.

## Step 3: done. hbond's rotamer-pair kernel and its pre-pass run on Metal.

Two kernels, not one. The rotamer-pair scoring kernel was the planned work;
the derived-atom pre-pass (`gen_hbond_bases`) turned out to be a hard
prerequisite, because `_hbond_score_args` calls it on every scoring call and
its two outputs are inputs to the scoring kernel. Both are ported.

### What the port touched

`hbond.hh` was the file with no portability work at all -- no view-generic
parameters, no address-space qualifiers, no lane handles. Converting it:

- The ten tile loaders lost their `TView` parameters for deduced view types,
  gained `TMOL_THREAD` on the per-residue scratch references, `TMOL_SHARED_REF`
  on the tile buffers, and a trailing `WorkgroupLane<Dev>`.
- `HBondSingleResData`'s seven tile pointers became `TMOL_SHARED_PTR`.
- `HBondRotPairData` **lost its four parameter views**. It used to carry
  `derived_coords`, `derived_atom_inds`, `pair_params` and `pair_polynomials`
  as TViews, which would have forced the struct itself to be view-templated and
  every `HBondScoringData<Dev, Real, Int>` spelling in the dispatch headers to
  name four more types. The only consumers were the three atom-energy
  functions, which already take every other input explicitly, so the views
  reach them as arguments instead and the structs collapsed to
  `HBondRotPairData<Real>` / `HBondScoringData<Real>`.
- Two template parameters turned out to be dead once the views were deduced:
  `typename Int` in the two tile-invariant loaders and in both
  `eval_*_don_acc_pair_energies`. That is the same `couldn't infer template
  argument 'Int'` the previous attempt hit -- the answer was not to reposition
  `Int` but to notice those four functions never used it.
- `interres_count_pair_separation` moved from `hbond_pose_score.impl.hh` into
  `hbond.hh`, the same move elec's made, because the kernel bodies call it and
  they are compiled by the Metal toolchain.

`potentials.hh` had 31 Metal errors in the four expected categories, plus one
that was not expected:

- `using std::exp` etc. (10) became `using tmol::portable::math::exp`, which
  leaves every call site below unchanged.
- `AcceptorHybridization`'s four `static constexpr int` members became
  enumerators. A static data member is a program-scope variable and Metal
  wants those in `constant`; an enumerator is not a variable.
- the two `throw std::runtime_error` guards grew `&& !TMOL_PORTABLE_METAL`.
- `hbond_poly_scalar`'s primary template names `double`, which Metal rejects
  even in a template body it will never instantiate, so under
  `TMOL_PORTABLE_METAL` the primary names the `DoubleFloat<float>` type too.
- `tmol/score/common/tuple_operators.hh` applied `TMOL_THREAD` to a
  **by-value** `ref_tuple` parameter. MSL rejects an address space on a
  by-value parameter; a `ref_tuple` is already a pack of pointers, so the
  qualifier was simply wrong and is gone. This is in a shared header, so it
  was latent for every term -- nothing had instantiated those two overloads
  through Metal before.

### The parameter tables are read in place, not copied

`hbond_score::V` / `::V_dV` take their two tables by **`TMOL_DEVICE_REF`**, a
new companion to `TMOL_DEVICE_PTR` in `metal_view.hh`: `device T const&` on
Metal and plain `T const&` everywhere else. They are elements of a parameter
view, so they live in device memory, and a thread-space copy would put the
whole 360-byte polynomial triple in registers.

The three component helpers below still take their polynomial in thread space,
because `bound_poly` indexes `coeffs` through a thread-space reference. So
`hbond_score` copies **one polynomial at a time** out of the device-side
table rather than the whole triple at once -- three sequential 120-byte copies
instead of one 360-byte one, which is far less live state. Measured not to
cost the host anything: the host relax cycle at 10 poses came in at 7900 ms
against 8268 ms recorded before the change.

### Scaffolding

Two sets, one per kernel, both the established six-part pattern:

| | rotamer scoring | pre-pass |
|---|---|---|
| ctx | `hbond_ctx.hh` (26 views) | `gen_hbond_bases_ctx.hh` (22 views) |
| bodies | `hbond_kernels.hh` | `gen_hbond_bases_kernels.hh` |
| shader | `hbond_mps_kernels.metal.in` | `gen_hbond_bases_mps_kernels.metal.in` |
| launch | `hbond_mps_launch.hh` | `gen_hbond_bases_mps_launch.hh` |
| TU | `hbond_pose_score.mps.cpp` | `gen_hbond_bases.mps.cpp` |

`HBondCtxT` is parameterized on `OutRank` **and** `DispRank` -- whole-pose
scoring writes `[term, pose, block, block]` and dispatches over
`[3, pose, block_pair]`, rotamer scoring writes `[term, rot_pair]` and
dispatches over `[3, rot_pair]`. It also carries `PolyReal`, the one field
whose element type is backend-dependent.

**The macros moved but did not fork.** All twelve tile macros now live in
`hbond_kernels.hh` with `ctx.` prefixes, and the whole-pose forward and
backward bodies -- which stay in `hbond_pose_score.impl.hh`, since only the
rotamer kernel is wired to Metal -- build the same context and expand the same
macros. One copy, four callers. That is why each of the four dispatch bodies
now opens with a context fill and a `WorkgroupLane<Dev> lane;`.

`HBondBasesCtxT` exposes `template <typename T, int N> using ViewT = View<T, N>`
so the bond walk can spell
`RotamerCentricIndexedBonds<Int, Ctx::template ViewT>` without knowing which
backend instantiated the context. `bonded_atom.hh` and `identification.hh`
were already view-generic from the lk_ball water-generator port, so the walk
itself needed no change.

Four ops flipped to `TMOL_DISPATCH_FLOATING_DEVICE_MPS`:
`gen_hbond_bases_op`, `hbond_rotamer_score_op`, `hbond_rotamer_spheres` and
`hbond_rotamer_dispatch_page`. The packer reaches hbond through
`iter_packing_rotamer_scores`, which uses all four; the last two delegate to
the sphere-overlap bodies ported earlier.

One Python change: `_hbond_energy_term._param_tables` used to force the
polynomial table to float64 unconditionally. On MPS `_params.py` has already
packed it as hi/lo float32 pairs, so the conversion would both fail and undo
the packing.

### Results

Isolated hbond rotamer-pair energies, 3 x 1UBQ, 1.9M shared pairs:

| | host | MPS |
|---|---|---|
| median | 184.8 ms | **104.2 ms** |

max abs error 1.07e-06, relative 3.33e-07 -- float32 round-off on the
arguments, with the double-float coefficients holding. Three pairs appear on
the host only and all three are exactly zero.

Moving hbond onto the device is worth about 7% of the whole cycle. Controlled
at 30 poses, same build, same machine, mixed column only:

| hbond on | calcRPEs | CYCLE |
|---|---|---|
| host | 9430 ms | 14131 ms |
| Metal | 8550 ms | **13232 ms** |

Less than the ~22% the earlier stage breakdown projected, for two reasons: the
isolated kernel is 1.77x rather than the 2.4x minimization sees, and hbond is a
smaller share of `calcRPEs` than the pose-score profile suggested.

Whole cycle, all four operations, 20 fixed minimization steps, 43 evaluations
in both columns:

| poses | host | mixed | speedup | before step 3 |
|---|---|---|---|---|
| 10 | 7997 ms | 4428 | **1.81x** | 1.67x |
| 30 | 21636 | 12462 | **1.74x** | 1.58x |
| 100 | 68505 | 40738 | **1.68x** | 1.60x |
| 300 | 203231 | 121019 | **1.68x** | 1.55x |

`calcRPEs` at 100 poses went 51430 -> 25930 against the host's 49020, which is
1.89x on the stage; it was 1.73x before step 3. Minimization is unchanged at
2.35x, as expected -- hbond has no Metal pose-score kernel, so it still runs on
the host there.

## Step 4: hbond's whole-pose kernels, and what the profiler said about them

The whole-pose forward and backward are ported too, by the same recipe. The
context grew a `rot_offset_for_block` view (27 now), two launch-shape scalars,
and a `DerivRank` parameter -- a forward pass accumulates into `[term, atom]`
and a backward into `[atom]`, and spelling that as a template parameter let the
backward passes stop carrying a default-constructed field.

The whole-pose path also takes elec's **cell-reduction** treatment: rather than
atomically adding each threadgroup's partial into one cell, each writes its own
block-pair cell and a fixed-order reduction sums them. One atomic per
threadgroup is thousands per pose arriving in an order the GPU may vary, and a
line search that validates its analytic derivative against a finite difference
cannot tolerate a non-reproducible energy.

**Accuracy is the best of the four ported terms.** At 30 x 1UBQ:

| term | energy rel | gradient max abs | peak \|g\| |
|---|---|---|---|
| **hbond** | **2.1e-07** | **1.0e-05** | 3.31 |
| elec | 5.7e-07 | 9.3e-04 | 5.29 |
| ljlk | 0.0 | 1.3e-03 | 67.0 |
| lk_ball | 1.4e-05 | 1.1e-03 | 3.18 |

### But it is slower than the host, and the reason is not what I assumed

hbond whole-pose forward+backward at 30 poses: host 5.8 ms, **MPS 7.5 ms --
0.78x**. So it is ported, correct, and deliberately *not* in the minimization
split; adding it there took minimization from 1143 ms to 1332 ms at 30 poses.
Its rotamer kernel is a different story and does run on the device.

Four probes, each replacing one candidate with nothing and re-timing:

| removed | kernel time | share of the kernel |
|---|---|---|
| nothing (baseline) | 6.7 ms/call | -- |
| the coordinate copy into shared | 6.53 | **2.5%** |
| the double-float Horner accumulator | 6.2 | **7%** |
| the entire hbond energy function | 5.6 (rotamer: 15%) | **15%** |
| every live block pair (dead grid only) | 1.72 | 26% is dead workgroups |

So **85% of the kernel is tile machinery**, not arithmetic. Three hypotheses
died here, and each was worth killing:

- *The double-float polynomial is the cost.* Tested by evaluating with a plain
  float accumulator and double-float coefficients -- two operations a Horner
  step against eighteen. It bought **7%** and cost four orders of accuracy
  (energy relative error 1.4e-07 -> 3.0e-03, gradient relative error 1.35).
  Reverted. The cancellation a degree-10 fit suffers is in the *accumulator*,
  not only in the stored coefficients, which sharpens the original step-1
  finding.
- *The tile loading is the cost.* The coordinate copy is 2.5%.
- *Compacting the block-pair grid is the win.* Only 34% of the upper triangle
  is live for 1UBQ (996 of 2926 pairs), so compaction can shrink the grid
  2.94x -- but the dead workgroups only cost 1.72 ms of 6.7, so the ceiling is
  about 17% of this kernel.

### The actual constraint: occupancy, from per-residue tile state

`msl_occupancy` reports `maxTotalThreadsPerThreadgroup`, which is the compiler
saying how many threads it can fit given the registers the kernel needs:

| entry | maxthr | threadgroup memory |
|---|---|---|
| elec_eval_energies_by_block | 1024 | 1712 B |
| hbond_bases_gen | 1024 | 496 B |
| every sphere_overlap entry | 1024 | 0 B |
| ljlk_eval_rotamer_energies_by_block | 768 | 1840 B |
| lk_ball_eval_rotamer_energies_by_block | 576 | 1776 B |
| **all three hbond scoring entries** | **384** | 2032 B |

hbond is the most register-hungry kernel in the tree, by a factor of 2.7
against elec. Two bisections, both compile-only and therefore cheap:

- Stubbing the entire energy function leaves it at **384**. So the math is not
  what costs the registers -- matching the timing probe that priced it at 15%.
- The pre-pass carries 22 views in its context and reaches **1024**. So the
  context is not what costs the registers either.

What is left is the tile machinery, and specifically `HBondSingleResData`:
hbond carries **seven** shared-memory pointers per residue -- coords, two tile
index arrays, two type arrays, hybridization, path distances -- against elec's
three, and the scoring data holds two of them. That state is live across the
whole tile loop.

This is the thing to fix, and it is a structural change to a struct the CPU and
CUDA backends share: the seven pointers all address one
`HBondBlockPairSharedData` at fixed offsets, so one base pointer plus
compile-time offsets would replace fourteen live pointers with two. Left
undone deliberately -- it wants the CUDA node to confirm it does not regress
there, since register pressure is the binding constraint on that backend too.

The in-place polynomial read landed anyway (`TMOL_DEVICE_REF` all the way down
into `poly`'s Horner loop, so nothing copies a 128-byte polynomial into
registers): it measured 1.4% better and is the honest expression of where the
parameters live. It did not move `maxthr`, which is how we know the copies were
not the problem.

### The fix that did work: stop compiling the dead derivative path

hbond reaches its atom-pair evaluation through one macro, `HBOND_ATOM_ENERGY`,
which branches on `accumulate_derivs` to pick between the energy-only and the
energy-and-derivatives function. That flag was a **runtime** bool, so every
hbond kernel compiled both halves -- including the rotamer kernel, which
packing never asks for derivatives from.

Making it compile-time is the whole fix. The two branches are both valid code,
so a plain `if` on a constant suffices for the optimizer to drop one; the body
takes `bool AccumulateDerivs` as a template parameter and the shader emits two
entries from it. CUDA already passed `std::true_type` / `std::false_type` here;
MSL has no `<type_traits>`, so `deriv_tag<B>` in hbond_kernels.hh is the
stand-in, with a constexpr conversion to bool.

| | before | after |
|---|---|---|
| `hbond_rotamer_eval_energies_f32` maxthr | 384 | **448** |
| that kernel, 10 poses | 1859 ms | **1198 ms** |
| hbond rotamer energies, MPS | 324 ms | **216 ms** |
| against the host | 1.84x | **2.75x** |

Accuracy is bit-for-bit what it was (max abs 1.073e-06). The wall-clock gain is
1.55x, well beyond what the 17% occupancy change alone would predict --
dropping the branch takes a large amount of instruction and scheduling pressure
out with it.

The same split is applied to the whole-pose entries, though it buys nothing
there in practice: minimization genuinely wants derivatives, so it launches the
`_derivs` entry either way. It helps block-pair-energy mode and costs nothing.

lk_ball and ljlk were checked for the same pattern and do not have it -- they
already keep separate rotamer-energy and derivative bodies. This was hbond's
alone, a consequence of reaching both through one macro.

### Where the cycle's GPU time actually goes

Per-kernel, one packing run at 10 poses:

| kernel | ms | share |
|---|---|---|
| lk_ball_eval_rotamer_energies_by_block | 2192 | 27.5% |
| ljlk_eval_rotamer_energies_by_block | 2054 | 25.8% |
| hbond_rotamer_eval_energies | 1885 | 23.7% |
| elec_eval_energies_by_block | 1495 | 18.8% |
| everything else (spheres, pair lists, waters, bases) | 343 | 4.3% |

The four rotamer kernels are 96% of GPU time and are evenly matched in
absolute cost, so no single one of them is the bottleneck. Their speedups
against the host are not evenly matched at all:

| term | host | MPS | speedup |
|---|---|---|---|
| ljlk | 3520 ms | 364 | 9.7x |
| elec | 2150 | 265 | 8.1x |
| lk_ball | 1183 | 391 | 3.0x |
| **hbond** | **603** | **327** | **1.8x** |

hbond is the weakest ratio because the host is *already fast* at it -- hbond
only evaluates donor-acceptor pairs, perhaps six per block pair, where elec
evaluates all 1024 atom pairs in a tile. On the GPU hbond costs about what elec
costs anyway, which is the same occupancy story from the other side: a
workgroup with six pairs of work cannot hide its own latency.

The remaining headroom in `calcRPEs` is not in these kernels. About half of
that stage is host work -- the terms with no Metal kernel at all (dunbrack,
rama, omega, cart_bonded, disulfide) plus the sparse assembly and coalescing.

### Cycle after the optimization

Whole cycle, all four operations, 20 fixed minimization steps:

| poses | host | mixed | speedup | before step 4 |
|---|---|---|---|---|
| 10 | 7813 ms | 4764 | 1.64x | 1.81x |
| 30 | 21459 | 12914 | 1.66x | 1.74x |
| 100 | 67998 | 39434 | **1.72x** | 1.68x |
| 300 | 201642 | 116740 | **1.73x** | 1.68x |

`calcRPEs` at 300 poses went 77.0 s -> 73.3 s, which is the hbond rotamer win
arriving at the cycle level: that kernel is about 8.6% of rotamer GPU time and
rotamer GPU is roughly half the stage.

The 10- and 30-pose columns moved the *other* way between runs. Those cycles
are short enough that run-to-run variance swamps a 5% stage change -- the same
10-pose configuration has measured anywhere from 4404 to 4764 ms across this
session with no code change in between. Read the 100- and 300-pose rows; they
are long enough to mean something.

### What is left on the host

At 100 poses the mixed cycle is 40.7 s, of which **26% is packer machinery with
no device path at all**: build IG 6600 ms, run SA 3850 ms, build rots 830 ms.
Those are the next target, and they are a different kind of work from the
kernels ported so far -- the interaction graph and the annealer are not
tile-structured atom-pair loops. The remaining kernel-side work is hbond's
whole-pose forward and backward, which would lift minimization; the macros and
the context are already in place for it, so it is the cheapest remaining port.

### Tooling note, worth more than it looks

`dev/mps_scratch/msl_probe` takes an **optional language-version argument**,
and without it defaults to the newest MSL, which *infers* `thread` where it is
omitted. tmol's shaders are compiled through torch's MPS shader API, which
pins 3.2, where it does not. Run it as `msl_probe foo.metal 32`. Probing
without the `32` reports a wall of bogus
`could not match 'thread MetalView<T, N>' against 'MetalView<...>'` errors --
and reports them for elec, ljlk and lk_ball too, which is the tell. A clean
probe at the default version is not evidence of anything.


---

# Four-step optimization pass (dev/mps_scratch/NEXT_STEPS.md)

Executed unattended against the plan written before the previous compaction.
Each step was gated on the 100-pose median CYCLE, keep only on >2%.

Baseline, three runs: 40120 / 39350 / 39674 ms, **median 39674**.
Final, three runs: 27499 / 27290 / 27288 ms, **median 27290 (-31.2%)**.

Speedup against the host path: **1.72x -> 2.47x** at 100 poses, 2.48x at 300.
That ratio understates the work: step 5 is in code both paths share, so the
host baseline improved too (68659 -> 65442 ms at 100 poses). The mixed cycle
itself went 39674 -> 26514 in the final sweep, **1.50x faster than this
session started**.

## Step 1 -- KEPT. Two changes, both in the overlap, neither the planned one.

The plan blamed the host exclusive scan in `sphere_overlap.impl.hh` and
proposed a device scan plus an over-allocated dispatch list. **The premise was
wrong and the diagnosis that replaced it only became visible after fixing the
measurement.**

First measurement said the scan and the count syncs cost 7.5 ms of a 25260 ms
calcRPEs -- apparently nothing to win. That reading was invalid:
`TMOL_MPS_TIME_KERNELS` waits on every launch, so by the time a host-side
sync is reached the queue is already drained and the sync looks free. Added
`TMOL_MPS_TIME_HOST` (mps_launch.cpp) which records host phases while leaving
launches asynchronous. Re-measured: the exclusive scan is 4 ms; the
`.item<int64_t>()` overflow-and-sum pair is **2712 ms**. The scan was never
the problem; draining the queue was.

**1a. MPS may reuse a larger-cutoff dispatch layout** (`_score_function.py`,
`_rotamer_dispatch_cutoff_compatible`). CUDA already did this -- the consumer's
potential applies its own distance cutoff -- and the same holds on Metal, which
runs the same kernels. The four ported terms have two distinct cutoffs, so
this halves the sphere-overlap builds, 8 -> 4. calcRPEs 25360 -> 21790 ms; the
sync collapses to 32 ms. Validated: identical pair sets, shared-vs-exact
rel 1.19e-07 (float reassociation only), host-vs-MPS unchanged at 4.17e-07.

**1b. Commit the command buffer after launching the device half**
(`relax_mixed_full.py`, `MixedRotamerScoring.__call__`). Enqueuing is not
starting: PyTorch's MPS backend holds the buffer open until something waits
on it, so the queued kernels did not begin until the collect and the host half
overlapped nothing. `torch.mps.Event().record()` commits without waiting.
calcRPEs 21830 -> 16740 ms; the collect falls 10130 -> 4996 while the host
half is unchanged, which is the device half running underneath it.
`mixed_device.py` already did this for whole-pose scoring; the rotamer path
had not adopted it.

## Step 2 -- NOT ATTEMPTED. Scale, stated precisely.

`InteractionGraphBuilder` (`pack/compiled/compiled.impl.hh`, 1203 lines) has
**53 DeviceDispatch sites**: 19 `forall`, 7 `scan_and_return_total`, 6
`shuffle_reduce_in_workgroup`, 5 `store_idempotent`, 5 `forall_independent`,
4 `for_each_in_workgroup`, 2 `reduce`, 2 `foreach_workgroup`, 2 `bitwise_or`,
1 `foreach_grouped_workgroup`. A port is ~30 Metal entry points against the
5 hbond needed -- and `DeviceDispatch<MPS>::scan` **does not exist** (an MPS
tensor's data pointer is an MTLBuffer handle, not an address a kernel can take),
so all 7 scans need the two-level device scan built first. The 9
host-scalar-returning calls each size a tensor, so each is a sync. Fifteen
output tensors against a 30-buffer binding limit.

Note `DeviceOperations<MPS>` inherits `DeviceOperations<CPU>` on the host side
(`device_operations.mps.host.impl.hh`), so instantiating the builder for MPS
compiles and links but would run **CPU kernel bodies against MPS buffers**.
There is no partial port: it works only once every entry point exists.

The annealer is larger still (~700 lines plus Philox plus three
`mgpu::segmented_sort` calls). Both are multi-session efforts. Left undone
rather than half-done.

**Finding worth keeping**: build IG runs at **1.75x parallelism on 6 threads**
(wall 6545 ms, CPU 11478 ms over 4 calls at 100 poses). 19 of its dispatch
sites use the strictly-serial `forall`. Converting the independent ones to
`forall_independent` is a plausible host-side win worth several seconds and is
far cheaper than a Metal port -- but it is host optimization, not MPS work, and
changing `forall` where a body accumulates non-atomically would be wrong, so it
needs a site-by-site audit. Not done here.

## Step 3 -- KILLED BY MEASUREMENT. Step 1 inverted its premise.

The plan targeted 10.5 s of host-only rotamer terms (cartbonded 3604,
gen_torsions 2357, rama 2251, omega 2227). After step 1 the host half runs
*underneath* the device half, so that time is largely already free.

Measured directly: dropping **all four** host-only RPE terms moves calcRPEs
only **16760 -> 15470 ms, 1290 ms**. The device pole is 11.8 s of rotamer
kernels (lk_ball 3642, ljlk 3422, elec 2668, hbond 2097) against 5.5 s of
host terms. Porting those four would delete host work that costs ~1.3 s and
add device work to the pole that sets calcRPEs -- **net negative or neutral**.

Do not port them for speed. If they are ported it should be to let the packer
run entirely on the device, which is a different goal with a different
justification.

## Step 4 -- NO CHANGE KEPT. lk_ball's register pressure is diffuse.

Occupancy, all four rotamer kernels (threadgroup size is 32, so
`maxTotalThreadsPerThreadgroup` is a register-pressure proxy, not a launch
limit): elec 1024, ljlk 768, lk_ball 576, hbond 448.

Bisected `lk_ball_eval_rotamer_energies_by_block_f32` by stubbing the expanded
shader:

| variant | maxthr |
|---|---|
| baseline | 576 |
| `lk_ball_atom_energy_full` stubbed entirely | **1024** |
| geometry + water load kept, `lk_ball_score::V` stubbed | 832 |
| waters passed by `thread const&` instead of by value | 576 (unchanged) |
| `lk_bridge_fraction` removed | 576 |
| `lk_fraction` removed | 576 |
| `lk_isotropic_pair` removed | 576 |

So the energy function is the whole of the gap (1024 -> 576): the two 4x3
water matrices plus geometry cost 192, `V` costs 256. But **no single callee
inside V accounts for any of it** -- removing any one leaves enough pressure to
stay at 576. The compiler already elides the by-value matrix copies. Reducing
it means restructuring the whole evaluation, in a header CUDA shares, where
register pressure also binds. Not attempted.

Tried and reverted: an early `has_water(wmat_polar)` exit in
`lk_ball_atom_energy_full` before loading the occluder's waters -- bit-identical
(V's own first act is that test) and it skips MAX_N_WATER shared reads on the
discard path. maxthr unmoved at 576; kernel **3642 -> 3659 ms**, no gain.
Reverted under the keep/revert gate.

Kept from that edit: `Mat<Real, 4, 3>` -> `Mat<Real, MAX_N_WATER, 3>` in
`lk_ball_atom_energy_full`, matching `lk_ball_atom_derivs_full` just below it.
The literal was correct only because MAX_N_WATER is 4.

hbond's item -- collapsing `HBondSingleResData`'s seven shared pointers -- was
gated in the plan on a CUDA node confirming no regression. No such node was
available, so it was not attempted.

## Step 5 -- KEPT. The interaction-graph build was half serial.

Found while assessing step 2. `InteractionGraphBuilder::f` accumulates the
sparse rotamer-pair energies twice: once inside `if (bump_check)` through
`forall_independent` (parallel), and once on the compacted numbering through
`forall` (**serial**). Both walk all 17.0M entries at 100 poses. That is the
whole of the 1.75x-on-6-threads figure: solving
`P + S = 11.5s` CPU against `P/6 + S = 6.5s` wall gives S ~ 5.5s.

The serial one is not an oversight. CPU accumulation is deliberately
non-atomic (accumulate.hh: "parallel callers must write to disjoint output
elements"), and three of that lambda's seven cases fold into a shared cell --
`bg_bg_energies[pose]`, and `energy1b` of the surviving partner when a block
is relegated to the background, which every neighbouring background block
contributes to. Flipping it to `forall_independent` would have raced.

**Split it in two instead.** The bulk case -- both blocks molten, both
rotamers kept -- writes cells no other entry touches: the diagonal carries
self-pairs only, so one `energy1b` cell per rotamer, and the off-diagonal
writes the two chunk cells of its own rotamer pair. That is the same disjoint
scatter the bump-check branch already runs in parallel. It now runs through
`forall_independent`; the shared-cell cases run through a second `forall` that
still walks every entry but exits after a handful of loads for the ones the
parallel pass handled.

| | before | after |
|---|---|---|
| build_interaction_graph, wall (4 calls, 100 poses) | 6545 ms | **3076** |
| parallelism on 6 threads | 1.75x | **4.08x** |
| 100-pose CYCLE | 31662 ms | **28189** (-11.0%) |

CUDA is unaffected: `DeviceOperations<CUDA>::forall_independent` is defined as
`forall`, so both passes compile to the same launches it made before. All 408
pack tests pass.

## Step 6 -- KEPT. Opt-in relaxed transcendentals on Metal.

Metal compiles with `MTLMathModeSafe` and the shader API exposes no
`MTLCompileOptions`, so `::metal::exp` is correctly rounded. `lk_fraction`
alone evaluates MAX_WATER exponentials and a logarithm per atom pair.

Measured globally first, at 30 poses:

| term | precise | fast | precise rel | fast rel |
|---|---|---|---|---|
| lk_ball | 1150.3 ms | **902.7** | 5.534e-07 | **5.031e-07** |
| ljlk | 1073.0 | **953.2** | 4.174e-07 | 4.174e-07 |
| hbond | 632.4 | 630.5 | 3.333e-07 | 3.333e-07 |
| elec | 781.8 | **805.7** | 9.248e-07 | **1.302e-06** |

elec goes the wrong way on both counts -- it reaches `exp` through the
sigmoidal dielectric in `elec/potentials/potentials.hh` (not `elec.hh`, which
has no transcendentals at all; that cost a wrong turn). So this is **not** the
default. `tmol::portable::math::fast_exp` / `fast_log` are separate names, and
only `lk_ball.hh` and `lk_isotropic.hh` opt in. Host and CUDA resolve them to
`::std` and stay bit-identical.

Result: lk_ball 1.28x with its error slightly *better*, ljlk 1.12x bit-for-bit
unchanged, elec and hbond untouched at their old numbers.
100-pose CYCLE 28189 -> **27322** (-3.1%).

One side effect worth knowing: mixed minimization now converges in 45
line-search evaluations against the host's 43 (it was 43/43). ljlk feeds
minimization, so the changed rounding moves the trajectory slightly. The two
extra evaluations cost ~165 ms against the ~480 ms the rotamer kernels save,
so it stays net positive -- but it is the same sensitivity recorded earlier
for gradient determinism, and worth re-checking if more call sites opt in.

## Also tried, measured, reverted

- **`fast_divide`** (`metal::fast::divide`) on elec's one-per-atom-pair
  Coulombic divide: 781.3 -> 780.9 ms, error identical. A single divide is not
  the bottleneck -- consistent with these kernels being occupancy-bound rather
  than ALU-bound. The helper was removed rather than left unused.
- **Merging and coalescing the two halves on the GPU** instead of the host.
  The halves are lopsided (device 16995425 entries, host 1274175, and
  coalescing removes exactly 1274175 -- every host entry duplicates a device
  one and the device half is already unique), and an MPS sparse coalesce
  measured 363 ms against roughly 600 on the host. But the device half is not
  sorted (29325 descents), so nothing cheaper than a full sort is available,
  and the round trip ate the difference: CYCLE 27404 against 27322. Reverted.
- **lk_ball early `has_water` exit** -- see step 4.

## What is left, in the order the measurements now rank it

1. **The device pole in calcRPEs, 11.8 s.** lk_ball 3642 and ljlk 3422 are the
   two largest. lk_ball's gap is understood (above) but needs a restructure.
2. **build IG, 6.6 s, host.** Cheapest real option is the serial-`forall`
   audit, not a Metal port.
3. **SA, 3.8 s, host.** A port is tractable in principle -- CUDA's annealer is
   already parallel, so backend trajectory divergence is already accepted -- but
   it is a session of its own.
