# Development

This page summarizes local development, extension loading, testing, CI, and
release workflows. It consolidates the root `DEVELOPMENT.md` and `dev/README.md`
guidance into the docs.

## Local Setup

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
pip install -e ".[dev]"
```

Requirements:

- Python 3.11 or newer.
- PyTorch 2.5 or newer.
- A C++17 compiler.
- CMake 3.18 or newer.
- CUDA toolkit with `nvcc` for CUDA builds.

Without CUDA, use a CPU-only build:

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

## Building Extensions

tmol builds extensions with CMake through `scikit-build-core`.

```bash
# Production extensions
pip install -e .

# Include test-only C++/CUDA extensions
pip install -e . -Ccmake.define.TMOL_BUILD_TESTS=ON

# Select GPU architectures
pip install -e . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="80;90"

# Control parallelism
MAX_JOBS=4 pip install -e . -Ccmake.define.TMOL_NVCC_THREADS=2
```

Important CMake variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `80;86;89;90` | GPU architectures to compile. |
| `TMOL_BUILD_TESTS` | `OFF` | Build test-only extensions. |
| `TMOL_NVCC_THREADS` | `4` | Threads per `nvcc` invocation. |
| `TMOL_ENABLE_CUDA` | `ON` | Turn off for CPU-only builds. |
| `MAX_JOBS` | auto | Maximum parallel build jobs. |

## AOT vs JIT Extension Loading

tmol can load kernels two ways:

- AOT: pre-built shared libraries bundled in the installed wheel.
- JIT: source files compiled on first use through `torch.utils.cpp_extension`.

Environment variables:

| Variable | Effect |
| --- | --- |
| `TMOL_USE_JIT=1` | Force JIT mode. |
| `TMOL_JIT_FALLBACK=1` | Try AOT first, then JIT if AOT is unavailable. |

Use JIT mode while editing kernels. Use AOT mode for normal installed packages.

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
pip install -e .
```

Apptainer:

```bash
apptainer build tmol-dev.sif containers/apptainer/tmol-dev.def
apptainer run --nv --bind "$(pwd):/tmol_host" tmol-dev.sif
```

## CI

GitHub Actions runs linting, CPU tests, CUDA tests, and benchmarks on the
self-hosted GPU runner. The docs workflow builds Sphinx docs on PRs and pushes
to `master`; PR builds upload rendered HTML artifacts and same-repository PRs
also publish preview docs under the Pages site.

## Releasing

Versioned wheel and sdist publication happens from `v*` tags. The tag version
must match `[project].version` in `pyproject.toml`.

Before using a versioned wheel URL, check the GitHub Releases page. The version
in a checkout is not proof that a release has been published.

## Code Style

tmol uses Black for Python formatting, Flake8 for linting, and clang-format for
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
