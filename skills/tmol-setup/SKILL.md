---
name: tmol-setup
description: >
  Install and verify TMol on CPU or CUDA. Use for wheel selection, source or
  editable builds, AOT versus JIT extension loading, PyTorch/CUDA compatibility,
  and import or compiler failures. Do not use for scoring or modeling once an
  installation already works.
allowed-tools: Bash, Read, Write, AskUserQuestion
license: Apache-2.0
---

# TMol setup

Produce a verified TMol installation using the least expensive path compatible
with the user's Python, PyTorch, platform, and device. Run commands from the TMol
repository root only when building a checkout.

This workflow requires Python 3.11+ and PyTorch 2.5+. Source or JIT builds need
a C++ compiler; only CUDA source or JIT builds need `nvcc`.

## Select the installation path

1. Inspect `python`, `torch.__version__`, `torch.version.cuda`, the platform,
   and GPU availability before choosing a wheel.
2. Prefer a matching wheel from the current GitHub release. Wheel tags must
   match the Python ABI and PyTorch minor; CUDA wheels must also match the CUDA
   lane.
3. Use `pip install tmol` when source-distribution fallback behavior is
   acceptable.
4. Use an editable source build for development or an unsupported wheel lane.
   Request CPU-only explicitly when CUDA is not wanted.

Read `docs/installation.md` for current supported lanes and runtime notes before
constructing a versioned wheel URL.

## Source builds

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
TMOL_DISABLE_WHEEL_FETCH=1 python -m pip install -e ".[dev]"
```

CPU-only:

```bash
TMOL_DISABLE_WHEEL_FETCH=1 python -m pip install -e . \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

For an existing PyTorch environment, install the small build dependencies and
use `--no-build-isolation`; see `docs/user_guide/development.md`. Control build
parallelism with `MAX_JOBS` and CUDA compiler parallelism with
`TMOL_NVCC_THREADS` instead of oversubscribing a scheduler allocation.

## AOT and JIT loading

Normal wheel installs use ahead-of-time compiled extensions. During kernel
development, force source compilation with:

```bash
TMOL_USE_JIT=1 python -c "import tmol; print(tmol.__version__)"
```

Use `TMOL_JIT_FALLBACK=1` only when the desired behavior is AOT first and JIT
after an AOT load failure. CPU JIT needs a C++ compiler and Ninja; CUDA JIT also
needs a compatible `nvcc`.

## Verify

```bash
python - <<'PY'
import torch
import tmol

print("tmol", tmol.__version__)
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("CPU threads", torch.get_num_threads())
PY
```

For a source checkout, finish with a narrow CPU test before attempting the full
suite:

```bash
pytest -q tmol/tests/score/test_score_function.py -k "not cuda"
```

## Failure routing

- ABI, PyTorch-minor, or CUDA-lane mismatch: install a matching wheel rather
  than debugging symbols in an incompatible binary.
- `GLIBCXX_* not found`: use a newer runtime/compiler or build against the host;
  do not copy arbitrary C++ runtime libraries into the package.
- no CUDA compiler: choose CPU-only or install a toolkit whose `nvcc` is
  compatible with the active PyTorch build.
- extension changes not appearing: use a fresh `TORCH_EXTENSIONS_DIR` or rebuild
  the editable install, then record which mode was tested.

Do not modify shell startup files, install system packages, or download large
toolchains unless the user authorized those environment changes.
