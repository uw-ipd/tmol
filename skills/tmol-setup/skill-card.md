# Skill Card

## Description

`tmol-setup` selects and verifies a compatible TMol CPU or CUDA installation,
including release wheels, editable source builds, and JIT extension loading.

## Owner and license

TMol maintainers. Apache-2.0; see the repository `LICENSE`.

## Requirements

Python 3.11+, PyTorch 2.5+, and a matching supported platform. Source/JIT builds
need CMake, Ninja, and a C++ compiler; CUDA source/JIT builds additionally need
`nvcc`. No credentials are required for public releases.

## Risks and mitigations

- Binary tags are coupled to the Python ABI, PyTorch minor, and CUDA lane. The
  skill inspects these before selecting a wheel.
- Source builds can consume substantial CPU, memory, and time. The skill exposes
  `MAX_JOBS` and does not install system toolchains without authorization.
- JIT caches can hide stale builds. The skill records the loading mode and uses
  a fresh cache when validating changed native sources.

## References and output

References: `docs/installation.md` and `docs/user_guide/development.md`.

Output: a working import, reported TMol/PyTorch/CUDA versions, device
availability, and a bounded test result. Installation may modify only the
selected Python environment and explicitly chosen build/cache directories.

## Skill version

0.1.0 (2026-09)
