# Development

For build requirements and platform support, see {doc}`installation
</installation>`.

## Local setup

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
python -m pip install "torch>=2.14" "scikit-build-core>=0.10" "cmake>=3.24,<4" "pybind11>=2.12" ninja packaging
python -m pip install --no-build-isolation -e ".[dev]"
```

Build with PyTorch 2.14+ headers. Production extensions target the stable C ABI
in PyTorch 2.11+. Test-only and JIT extensions use the local PyTorch C++ API and
must be rebuilt when changing that runtime.

TMol locates Torch libraries/headers and pybind11 from this Python interpreter.
It does not use TorchConfig.cmake or the unstable Torch C++ ABI for production.

Requirements:

- Python 3.11 or newer.
- PyTorch 2.14+ build headers; PyTorch 2.11+ runtime.
- A C++17 compiler (C++20 for test/JIT extensions with recent PyTorch).
- CMake 3.24 or newer.
- CUDA toolkit with `nvcc` for CUDA builds.

Without CUDA, use a CPU-only build:

```bash
pip install --no-build-isolation -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

## Build extensions

TMol builds extensions with CMake through `scikit-build-core`.

```bash
# Production extensions
pip install --no-build-isolation -e .

# Include test-only C++/CUDA extensions
pip install --no-build-isolation -e ".[dev]" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON

# CUDA development build, including GPU test helpers
pip install --no-build-isolation -e ".[dev]" -Ccmake.define.TMOL_ENABLE_CUDA=ON \
  -Ccmake.define.TMOL_BUILD_TESTS=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="80;90"

# Separate distribution containing only CUDA operators
pip wheel --no-build-isolation --no-deps ./packages/cu13

# Control parallelism
MAX_JOBS=4 pip install --no-build-isolation -e . -Ccmake.define.TMOL_NVCC_THREADS=2
```

Build settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | `native` compiles only for GPUs visible at configure time. `all`, used for release wheels, emits SASS for every sm_75+ target reported by `nvcc --list-gpu-code`, plus PTX for the newest target. |
| `TMOL_BUILD_TESTS` | `OFF` | Build test-only extensions. |
| `TMOL_NVCC_THREADS` | `4` | Threads per `nvcc` invocation. |
| `TMOL_ENABLE_CUDA` | `OFF` | Enable CUDA in a development build; CUDA operator packages enable it automatically. |
| `TMOL_STABLE_HEADERS` | build-time Torch include directory | Optional PyTorch 2.14+ headers when testing production modules against an older runtime; test helpers retain that runtime's headers. |
| `MAX_JOBS` | auto | Maximum parallel build jobs. |

## Extension loading

TMol can load kernels two ways:

- AOT: pre-built shared libraries bundled in the installed wheel.
- JIT: source files compiled on first use through `torch.utils.cpp_extension`.

Environment variables:

| Variable | Effect |
| --- | --- |
| `TMOL_USE_JIT=1` | Force JIT mode. |
| `TMOL_JIT_FALLBACK=1` | Try AOT first, then JIT if AOT is unavailable. |

Use JIT mode while editing kernels. Use AOT mode for normal installed packages.
On Linux, the JIT loader detects PyTorch's OpenMP CPU backend and propagates
the corresponding compiler and linker flags, so `at::parallel_for` has the
same thread behavior as the CMake-built extension. `OMP_NUM_THREADS` and
`torch.set_num_threads()` therefore apply consistently in both modes.

## Tests

```bash
# All tests
pytest tmol/tests/ -v

# Specific file
pytest tmol/tests/score/test_score_function.py -v

# Skip CUDA-parametrized cases
pytest tmol/tests/ -v -k "not cuda"

# Coverage
pytest tmol/tests/ --cov=./tmol --junitxml=results.xml

# Benchmarks
pytest --benchmark-enable --benchmark-only --benchmark-max-time=.1
```

Ligand charge generation is intentionally strict. Partial charges come from the
SMILES to OpenBabel MMFF94 mol2 step and are applied by atom index. There is no
RDKit/Gasteiger fallback and no charge-mode switch.

## Containers

Docker:

```bash
docker build -t tmol-dev -f containers/docker/tmol-dev.Dockerfile .
docker run --gpus all -it -v "$(pwd):/tmol_host" -w /tmol_host tmol-dev bash
pip install --no-build-isolation -e .
```

Apptainer:

```bash
apptainer build tmol-dev.sif containers/apptainer/tmol-dev.def
apptainer run --nv --bind "$(pwd):/tmol_host" tmol-dev.sif
```

## CI

GitHub Actions runs linting, the Linux x86-64, Linux aarch64, and Apple Silicon
CPU suites, and CPU documentation builds on hosted runners. CUDA tests,
benchmarks, and the CUDA-only tutorial smoke tests use the self-hosted DIGS
runner. PR docs builds upload rendered HTML artifacts; same-repository PRs also
publish previews under the Pages site.

## Releasing

Versioned wheel and sdist publication happens from `v*` tags. The tag version
must match `[project].version` in `pyproject.toml`.

`scripts/release_matrix.py` defines 28 wheels: 12 CPU/API wheels and 16 CUDA
operator wheels across Python 3.11–3.14 and supported platforms. CUDA builds
use a toolkit floor of 12.8 or 13.0, independent of the build-time Torch minor.
All three project versions and the extras/dependencies must agree.

CI audits each wheel for unstable PyTorch C++ imports and tests the **same
binary** against PyTorch 2.11 and 2.14 where CUDA availability permits. It also
checks native scoring/gradients, actual pip resolution through staging indexes,
source metadata, wheel contents, and RECORD hashes before publication. All
three distributions go to PyPI without renaming wheels or adding local versions.
The existing `PYPI_API_TOKEN` repository secret must cover `tmol-ops-cu12` and
`tmol-ops-cu13` as well as `tmol`. For the first upload, an account-scoped token
can create both new projects; a token scoped only to `tmol` cannot. The workflow
uploads both CUDA distributions before the base release, so missing project
access stops publication before new base extras become visible. No separate
package index or user-side build is needed. Publication remains gated on the
complete wheel matrix and CPU/GPU installation checks.

Operators register directly with PyTorch's stable dispatcher. Python autograd
owns differentiation and saved-variable version checks; saved tensors cross the
boundary in batches. Native allocations, dispatch, streams, and generator access
use the stable C API. PyTorch 2.14+ provides direct Python tensor interop; older
runtimes exchange real tensor handles through a private stable operator. Neither
path copies tensor storage. Whole-pose scoring groups eligible precomputed
coordinate derivatives into one autograd context and CUDA reduction. This path
checks that static tensors are frozen before skipping per-term gradient queries;
Python wrappers, hooks, trainable parameters, and weighted/parallel execution
retain their ordinary differentiation paths. The engine schedules remaining
term graphs directly, without a nested backward call. Saved derivatives stay
under PyTorch's saved-variable ownership; CUDA reduction receives the entire
tensor list through one native dispatcher call on every supported runtime.
New native operations must preserve this boundary.
`scripts/check_native_abi.py` catches accidental C++ ABI imports and unversioned
Linux C++ runtime imports. Production uses C++17 to retain the manylinux floor.
CUDA packing draws phase seeds through Torch's public dispatcher rather than
accessing `CUDAGeneratorImpl`; fixed seeds remain reproducible, but trajectories
can differ from releases before this refactor.

Before using a versioned wheel URL, check the GitHub Releases page. The version
in a checkout is not proof that a release has been published.

## Code style

TMol uses Black for Python formatting, Flake8 for linting, and clang-format for
C++/CUDA formatting.

```bash
black --check .
black .
flake8
```

Run pre-commit before opening a PR:

```bash
pre-commit run --all-files
```
