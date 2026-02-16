"""Auto-configure CUDA environment for pip-installed CUDA toolkit.

When ``nvidia-cuda-nvcc`` is installed via pip (the ``[cuda]`` extra), the
nvcc binary, runtime libraries, and CCCL headers end up scattered across
``site-packages/nvidia/cu*/``, ``nvidia/cuda_cccl/``, etc.  PyTorch's
``cpp_extension`` module expects a single ``CUDA_HOME`` directory with
``bin/nvcc``, ``lib/libcudart.so``, and ``include/``.

This module bridges the gap by:

1. Discovering the pip-installed nvcc (``nvidia/cu*/bin/nvcc``).
2. Setting ``CUDA_HOME``, ``PATH``, and ``LD_LIBRARY_PATH`` so that PyTorch
   finds the correct compiler and libraries.
3. Creating compatibility symlinks that PyTorch cu12 wheels expect
   (``nvidia/cu12 -> nvidia/cu13``, ``libcudart.so -> libcudart.so.N``).
4. Locating the CCCL include directory for headers like ``nv/target``.

**No-op behaviour**: If ``CUDA_HOME`` (or ``CUDA_PATH``) is already set in the
environment (e.g. inside an NGC container where ``/usr/local/cuda`` exists),
*nothing* is modified.

Call :func:`setup` **before** importing ``torch.utils.cpp_extension``.
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import site

log = logging.getLogger(__name__)

_setup_done = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup() -> None:
    """Discover pip-installed CUDA toolkit and configure the environment.

    Safe to call multiple times (idempotent after the first call).
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    # Respect explicit user / container settings.
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        log.debug(
            "CUDA_HOME already set (%s); skipping pip CUDA auto-setup.",
            os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        )
        return

    cuda_home = _find_pip_cuda_home()
    if cuda_home is None:
        log.debug("No pip-installed nvcc found; skipping CUDA auto-setup.")
        return

    log.info("tmol: auto-detected pip CUDA toolkit at %s", cuda_home)

    # 1. CUDA_HOME -----------------------------------------------------------
    os.environ["CUDA_HOME"] = cuda_home

    # 2. PATH ----------------------------------------------------------------
    bin_dir = os.path.join(cuda_home, "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path

    # 3. LD_LIBRARY_PATH -----------------------------------------------------
    lib_dir = os.path.join(cuda_home, "lib")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + ld if ld else "")

    # 4. libcudart.so symlink ------------------------------------------------
    _ensure_libcudart_symlink(lib_dir)

    # 5. cu12 compat symlink -------------------------------------------------
    _ensure_cu12_compat_symlink(cuda_home)


def get_cccl_include() -> str | None:
    """Return the CCCL include directory, or *None* if not found.

    The ``nvidia-cuda-cccl`` pip package installs headers (``nv/target``,
    ``cub/``, ``thrust/``) into ``site-packages/nvidia/cuda_cccl/include/``.
    """
    for sp in _site_packages():
        cccl_inc = os.path.join(sp, "nvidia", "cuda_cccl", "include")
        if os.path.isdir(cccl_inc):
            return cccl_inc
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _site_packages() -> list[str]:
    """Return candidate site-packages directories."""
    paths: list[str] = []
    with contextlib.suppress(AttributeError):
        paths.extend(site.getsitepackages())
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        paths.append(user_sp)
    return paths


def _find_pip_cuda_home() -> str | None:
    """Find the pip-installed CUDA home (``nvidia/cu*/``) that contains nvcc."""
    for sp in _site_packages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        # Sort descending so we prefer the newest cu version (e.g. cu13 > cu12)
        for cuda_dir in sorted(glob.glob(os.path.join(nvidia_dir, "cu*")), reverse=True):
            nvcc = os.path.join(cuda_dir, "bin", "nvcc")
            if os.path.isfile(nvcc) and os.access(nvcc, os.X_OK):
                return cuda_dir
    return None


def _ensure_libcudart_symlink(lib_dir: str) -> None:
    """Create ``libcudart.so`` symlink if only a versioned ``.so.N`` exists.

    The linker flag ``-lcudart`` needs the unversioned name.
    """
    unversioned = os.path.join(lib_dir, "libcudart.so")
    if os.path.exists(unversioned):
        return

    candidates = sorted(glob.glob(os.path.join(lib_dir, "libcudart.so.*")))
    if not candidates:
        return

    target = os.path.basename(candidates[-1])  # highest version
    try:
        os.symlink(target, unversioned)
        log.info("tmol: created symlink %s -> %s", unversioned, target)
    except OSError as exc:
        log.warning(
            "tmol: could not create libcudart.so symlink in %s: %s. "
            "JIT compilation may fail at link time. "
            "Fix: ln -s %s %s",
            lib_dir,
            exc,
            target,
            unversioned,
        )


def _ensure_cu12_compat_symlink(cuda_home: str) -> None:
    """Create ``nvidia/cu12`` symlink for PyTorch cu12 compatibility.

    PyTorch cu12 wheels hardcode library/include paths under
    ``site-packages/nvidia/cu12/``.  If the actual toolkit is cu13 (or
    later), we create a symlink so those paths resolve correctly.
    """
    nvidia_dir = os.path.dirname(cuda_home)  # .../nvidia/
    cuda_dirname = os.path.basename(cuda_home)  # e.g. "cu13"

    if cuda_dirname == "cu12":
        return  # nothing to do

    cu12_path = os.path.join(nvidia_dir, "cu12")
    if os.path.exists(cu12_path):
        return  # already exists (real dir or symlink)

    try:
        os.symlink(cuda_dirname, cu12_path)
        log.info("tmol: created compat symlink %s -> %s", cu12_path, cuda_dirname)
    except OSError as exc:
        log.warning(
            "tmol: could not create cu12 compat symlink: %s. "
            "PyTorch JIT compilation may fail to locate nvcc. "
            "Fix: ln -s %s %s",
            exc,
            cuda_dirname,
            cu12_path,
        )

