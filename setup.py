"""
tmol setup.py — Ahead-of-time compilation of C++/CUDA extensions.

Builds pre-compiled shared libraries (.so) for all tmol extension modules,
eliminating the need for nvcc at install time or runtime.

Patterns adapted from:
  - xformers  (single _C.so, TORCH_LIBRARY registration, graceful loader)
  - flash-attention  (NinjaBuildExtension, CachedWheelsCommand, wheel caching)

Environment variables:
  TMOL_SKIP_CUDA_BUILD=TRUE   Skip C++/CUDA compilation (for sdist creation)
  TMOL_SKIP_TEST_EXTS=TRUE    Skip test extensions (for wheel builds)
  TMOL_FORCE_CXX11_ABI=TRUE   Force C++11 ABI (for nvcr container compat)
  TORCH_CUDA_ARCH_LIST         GPU architectures (default: "8.0 8.6 8.9 9.0+PTX")
  MAX_JOBS                     Max parallel compilation jobs
  NVCC_THREADS                 Threads per nvcc invocation (default: 4)
"""

import os
import platform
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path

from setuptools import setup

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SKIP_CUDA_BUILD = os.getenv("TMOL_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
SKIP_TEST_EXTS = os.getenv("TMOL_SKIP_TEST_EXTS", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("TMOL_FORCE_CXX11_ABI", "FALSE") == "TRUE"
FORCE_BUILD = os.getenv("TMOL_FORCE_BUILD", "FALSE") == "TRUE"
NVCC_THREADS = os.getenv("NVCC_THREADS", "4")

THIS_DIR = Path(__file__).parent.resolve()
# Include dirs can be absolute (setuptools allows it for -I flags)
INCLUDE_DIRS = [str(THIS_DIR), str(THIS_DIR / "tmol" / "extern")]
# Source paths must be relative to setup.py directory


PACKAGE_NAME = "tmol"

# ---------------------------------------------------------------------------
# Auto-discover CUDA before importing torch (torch reads CUDA_HOME at import)
# ---------------------------------------------------------------------------


def _auto_discover_cuda():
    """Set CUDA_HOME if not already set, checking common locations."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return  # Already set by user

    import shutil

    # 1. Check /usr/local/cuda (standard on GPU nodes / containers)
    if os.path.isfile("/usr/local/cuda/bin/nvcc"):
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        return
    # 2. Check if nvcc is on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is at <cuda_home>/bin/nvcc
        os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)
        return
    # 3. Check pip-installed nvidia-cuda-nvcc in site-packages
    try:
        import site

        for sp in site.getsitepackages():
            for sub in ["nvidia/cuda_nvcc", "nvidia/cu13", "nvidia/cu12"]:
                candidate = os.path.join(sp, sub, "bin", "nvcc")
                if os.path.isfile(candidate):
                    os.environ["CUDA_HOME"] = os.path.join(sp, sub)
                    return
    except Exception:
        pass


if not SKIP_CUDA_BUILD:
    _auto_discover_cuda()

# ---------------------------------------------------------------------------
# Import torch (required at build time for CUDAExtension)
# ---------------------------------------------------------------------------

import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# Force C++11 ABI for nvcr container compatibility
if FORCE_CXX11_ABI:
    torch._C._GLIBCXX_USE_CXX11_ABI = True

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

CXX_FLAGS = ["-O3", "-std=c++17", "-w"]
NVCC_FLAGS = [
    "-std=c++17",
    "--expt-extended-lambda",
    "-O3",
    "-w",
    "-DWITH_NVTX",
    "--threads",
    NVCC_THREADS,
]

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
NVCC_FLAGS += [
    f"-DTORCH_VERSION_MAJOR={TORCH_MAJOR}",
    f"-DTORCH_VERSION_MINOR={TORCH_MINOR}",
]

# If TORCH_CUDA_ARCH_LIST is not set, default to modern GPUs
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.6 8.9 9.0+PTX"


# ---------------------------------------------------------------------------
# Helper: find nvtx include path
# ---------------------------------------------------------------------------


def _nvtx_include_dirs():
    """Find NVTX include path for -DWITH_NVTX support."""
    try:
        import nvidia.nvtx as _nvtx

        p = os.path.join(_nvtx.__path__[0], "include")
        if os.path.isdir(p):
            return [p]
    except ImportError:
        pass
    # Fallback: relative to nvcc
    if CUDA_HOME:
        p = os.path.join(CUDA_HOME, "include")
        if os.path.isdir(p):
            return [p]
    return []


# ---------------------------------------------------------------------------
# Extension builders
# ---------------------------------------------------------------------------


def _make_cuda_ext(name, sources, define_macros=None, extra_include_dirs=None):
    """Create a CUDAExtension with tmol defaults."""
    macros = list(define_macros or [])
    inc = list(INCLUDE_DIRS) + _nvtx_include_dirs() + list(extra_include_dirs or [])
    return CUDAExtension(
        name=name,
        sources=list(sources),
        include_dirs=inc,
        define_macros=macros,
        extra_compile_args={
            "cxx": [*CXX_FLAGS, "-DWITH_CUDA", "-DWITH_NVTX"],
            "nvcc": [*NVCC_FLAGS, "-DWITH_CUDA"],
        },
    )


def _make_cpp_ext(name, sources, define_macros=None, extra_include_dirs=None):
    """Create a CppExtension (CPU-only) with tmol defaults."""
    macros = list(define_macros or [])
    inc = list(INCLUDE_DIRS) + list(extra_include_dirs or [])
    return CppExtension(
        name=name,
        sources=list(sources),
        include_dirs=inc,
        define_macros=macros,
        extra_compile_args={
            "cxx": [*CXX_FLAGS, "-DWITH_NVTX"],
        },
    )


def _make_pybind_ext(name, sources, cuda=False, extra_include_dirs=None):
    """Create a pybind11 extension with TORCH_EXTENSION_NAME set to the last component."""
    last_component = name.rsplit(".", 1)[-1]
    macros = [("TORCH_EXTENSION_NAME", last_component)]
    if cuda:
        return _make_cuda_ext(name, sources, define_macros=macros, extra_include_dirs=extra_include_dirs)
    else:
        return _make_cpp_ext(name, sources, define_macros=macros, extra_include_dirs=extra_include_dirs)


# ---------------------------------------------------------------------------
# Production extensions
# ---------------------------------------------------------------------------


def _production_extensions():
    """All production C++/CUDA extension modules."""
    exts = []

    # -----------------------------------------------------------------------
    # 1. tmol._C — all 14 TORCH_LIBRARY modules in one shared library
    # -----------------------------------------------------------------------
    _C_sources = [
        # io/details/compiled
        "tmol/io/details/compiled/compiled.ops.cpp",
        "tmol/io/details/compiled/gen_pose_leaf_atoms.cpu.cpp",
        "tmol/io/details/compiled/gen_pose_leaf_atoms.cuda.cu",
        "tmol/io/details/compiled/resolve_his_taut.cpu.cpp",
        "tmol/io/details/compiled/resolve_his_taut.cuda.cu",
        # kinematics/compiled (compiled_ops — TORCH_LIBRARY only)
        "tmol/kinematics/compiled/compiled_ops.cpp",
        "tmol/kinematics/compiled/compiled.cpu.cpp",
        "tmol/kinematics/compiled/compiled.cuda.cu",
        # pack/compiled
        "tmol/pack/compiled/compiled.ops.cpp",
        "tmol/pack/compiled/compiled.cpu.cpp",
        "tmol/pack/compiled/compiled.cuda.cu",
        # pack/rotamer/dunbrack
        "tmol/pack/rotamer/dunbrack/compiled.ops.cpp",
        "tmol/pack/rotamer/dunbrack/compiled.cpu.cpp",
        "tmol/pack/rotamer/dunbrack/compiled.cuda.cu",
        # pose/compiled/apsp
        "tmol/pose/compiled/apsp_vestibule.ops.cpp",
        "tmol/pose/compiled/apsp.cpu.cpp",
        "tmol/pose/compiled/apsp.cuda.cu",
        # score/backbone_torsion
        "tmol/score/backbone_torsion/potentials/compiled.ops.cpp",
        "tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.cpu.cpp",
        "tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.cuda.cu",
        # score/cartbonded
        "tmol/score/cartbonded/potentials/compiled.ops.cpp",
        "tmol/score/cartbonded/potentials/cartbonded_pose_score.cpu.cpp",
        "tmol/score/cartbonded/potentials/cartbonded_pose_score.cuda.cu",
        # score/constraint
        "tmol/score/constraint/potentials/compiled.ops.cpp",
        "tmol/score/constraint/potentials/constraint_score.cpu.cpp",
        "tmol/score/constraint/potentials/constraint_score.cuda.cu",
        # score/disulfide
        "tmol/score/disulfide/potentials/compiled.ops.cpp",
        "tmol/score/disulfide/potentials/disulfide_pose_score.cpu.cpp",
        "tmol/score/disulfide/potentials/disulfide_pose_score.cuda.cu",
        # score/dunbrack
        "tmol/score/dunbrack/potentials/compiled.ops.cpp",
        "tmol/score/dunbrack/potentials/dunbrack_pose_score.cpu.cpp",
        "tmol/score/dunbrack/potentials/dunbrack_pose_score.cuda.cu",
        # score/elec
        "tmol/score/elec/potentials/compiled.ops.cpp",
        "tmol/score/elec/potentials/elec_pose_score.cpu.cpp",
        "tmol/score/elec/potentials/elec_pose_score.cuda.cu",
        # score/hbond
        "tmol/score/hbond/potentials/compiled.ops.cpp",
        "tmol/score/hbond/potentials/hbond_pose_score.cpu.cpp",
        "tmol/score/hbond/potentials/hbond_pose_score.cuda.cu",
        # score/ljlk (note: rotamer_pair_energy sources commented out in original)
        "tmol/score/ljlk/potentials/compiled.ops.cpp",
        "tmol/score/ljlk/potentials/ljlk_pose_score.cpu.cpp",
        "tmol/score/ljlk/potentials/ljlk_pose_score.cuda.cu",
        # score/lk_ball
        "tmol/score/lk_ball/potentials/compiled.ops.cpp",
        "tmol/score/lk_ball/potentials/lk_ball_pose_score.cpu.cpp",
        "tmol/score/lk_ball/potentials/lk_ball_pose_score.cuda.cu",
        "tmol/score/lk_ball/potentials/gen_pose_waters.cpu.cpp",
        "tmol/score/lk_ball/potentials/gen_pose_waters.cuda.cu",
    ]
    exts.append(_make_cuda_ext("tmol._C", _C_sources))

    # -----------------------------------------------------------------------
    # 2. Pybind11 production modules (separate .so each)
    # -----------------------------------------------------------------------

    # bspline (CPU-only pybind)
    exts.append(
        _make_pybind_ext(
            "tmol.numeric.bspline_compiled._compiled",
            ["tmol/numeric/bspline_compiled/bspline.pybind.cpp"],
            cuda=False,
        )
    )

    # inverse kinematics (CPU+CUDA pybind)
    exts.append(
        _make_pybind_ext(
            "tmol.kinematics.compiled._compiled_inverse_kin",
            [
                "tmol/kinematics/compiled/compiled_inverse_kin.cpp",
                "tmol/kinematics/compiled/compiled.cpu.cpp",
                "tmol/kinematics/compiled/compiled.cuda.cu",
            ],
            cuda=True,
        )
    )

    # cubic hermite polynomial (CPU-only pybind)
    exts.append(
        _make_pybind_ext(
            "tmol.score.common._cubic_hermite_polynomial",
            ["tmol/score/common/_cubic_hermite_polynomial.cpp"],
            cuda=False,
        )
    )

    return exts


# ---------------------------------------------------------------------------
# Test extensions
# ---------------------------------------------------------------------------


def _test_extensions():
    """Test-only C++/CUDA extension modules (compiled AOT, no JIT needed)."""
    exts = []

    # Test pybind modules with dedicated compiled.py wrappers
    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.hbond.potentials._ext",
            ["tmol/tests/score/hbond/potentials/compiled.pybind.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.ljlk.potentials._ext",
            ["tmol/tests/score/ljlk/potentials/compiled.pybind.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.lk_ball.potentials._ext",
            ["tmol/tests/score/lk_ball/potentials/compiled.pybind.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common.polynomial._ext",
            ["tmol/tests/score/common/polynomial/polynomial.pybind.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common.geom._ext",
            ["tmol/tests/score/common/geom/geom.pybind.cpp"],
            cuda=False,
        )
    )

    # geom CUDA test extension (separate because loaded independently)
    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common.geom._ext_cuda",
            ["tmol/tests/score/common/geom/geom.cu"],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common._uaid_util",
            ["tmol/tests/score/common/uaid_util.pybind.cc"],
            cuda=False,
        )
    )

    # Test modules loaded inline in test functions
    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.bonded_atom._ext",
            [
                "tmol/tests/score/bonded_atom/test.pybind.cpp",
                "tmol/tests/score/bonded_atom/test_cpu.cpp",
                "tmol/tests/score/bonded_atom/test_cuda.cu",
            ],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common.dispatch._ext",
            [
                "tmol/tests/score/common/dispatch/test.pybind.cpp",
                "tmol/tests/score/common/dispatch/test_cpu.cpp",
                "tmol/tests/score/common/dispatch/test_cuda.cu",
            ],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common._warp_segreduce",
            [
                "tmol/tests/score/common/warp_segreduce.cpp",
                "tmol/tests/score/common/warp_segreduce.cuda.cu",
            ],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.common._warp_stride_reduce",
            [
                "tmol/tests/score/common/warp_stride_reduce.cpp",
                "tmol/tests/score/common/warp_stride_reduce.cuda.cu",
            ],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.score.ljlk.potentials._sphere_overlap",
            ["tmol/tests/score/ljlk/potentials/sphere_overlap.cu"],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.kinematics.segscan._ext",
            ["tmol/tests/kinematics/segscan/segscan.cu"],
            cuda=True,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.pack.rotamer.dunbrack._ext",
            [
                "tmol/tests/pack/rotamer/dunbrack/compiled.pybind.cpp",
                "tmol/tests/pack/rotamer/dunbrack/test_cpu.cpp",
                "tmol/tests/pack/rotamer/dunbrack/test_cuda.cu",
            ],
            cuda=True,
        )
    )

    # Utility test extensions (tensor struct, accessor, collection, heap)
    exts.append(
        _make_pybind_ext(
            "tmol.tests.utility.tensor._tensor_struct",
            ["tmol/tests/utility/tensor/tensor_struct.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.utility.tensor._tensor_accessor",
            ["tmol/tests/utility/tensor/tensor_accessor.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.utility.tensor._tensor_collection",
            ["tmol/tests/utility/tensor/tensor_collection.cpp"],
            cuda=False,
        )
    )

    exts.append(
        _make_pybind_ext(
            "tmol.tests.utility.datastructures._in_place_heap",
            ["tmol/tests/utility/datastructures/in_place_heap.cpp"],
            cuda=False,
        )
    )

    # TORCH_LIBRARY test module (custom_op)
    exts.append(
        _make_cuda_ext(
            "tmol.tests.utility.torchscript._custom_op",
            ["tmol/tests/utility/torchscript/custom_op.cpp"],
            define_macros=[("TORCH_EXTENSION_NAME", "_custom_op")],
        )
    )

    return exts


# ---------------------------------------------------------------------------
# NinjaBuildExtension — memory-aware build (from flash-attention)
# ---------------------------------------------------------------------------


class NinjaBuildExtension(BuildExtension):
    """BuildExtension that auto-limits MAX_JOBS based on available memory."""

    def __init__(self, *args, **kwargs):
        if not os.environ.get("MAX_JOBS"):
            import psutil

            nvcc_threads = max(1, int(NVCC_THREADS))
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024**3)
            # ~5GB peak memory per nvcc thread
            max_num_jobs_memory = max(1, int(free_memory_gb / (5 * nvcc_threads)))
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            print(
                f"[tmol] Auto-set MAX_JOBS={max_jobs}, NVCC_THREADS={nvcc_threads}. "
                "Set MAX_JOBS=N or NVCC_THREADS=N to override."
            )
            os.environ["MAX_JOBS"] = str(max_jobs)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# CachedWheelsCommand — download pre-built wheel before building
# (from flash-attention)
# ---------------------------------------------------------------------------

BASE_WHEEL_URL = "https://github.com/uw-ipd/tmol/releases/download/{tag_name}/{wheel_name}"


def _get_package_version():
    """Read version from pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # Python < 3.11
    try:
        with open(Path(__file__).parent / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "0.0.0"


def _get_platform():
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    else:
        return "unknown"


def _get_wheel_url():
    from packaging.version import parse

    version = _get_package_version()
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = _get_platform()
    torch_cuda_version = parse(torch.version.cuda) if torch.version.cuda else None
    cuda_version = f"{torch_cuda_version.major}" if torch_cuda_version else "cpu"
    torch_version = f"{TORCH_MAJOR}.{TORCH_MINOR}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    wheel_name = (
        f"{PACKAGE_NAME}-{version}+cu{cuda_version}torch{torch_version}"
        f"cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    )
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{version}", wheel_name=wheel_name)
    return wheel_url, wheel_name


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class CachedWheelsCommand(_bdist_wheel):
        """Try downloading a pre-built wheel before building from source."""

        def run(self):
            if FORCE_BUILD:
                return super().run()
            try:
                wheel_url, wheel_filename = _get_wheel_url()
                print(f"[tmol] Checking for pre-built wheel: {wheel_url}")
                urllib.request.urlretrieve(wheel_url, wheel_filename)
                if not os.path.exists(self.dist_dir):
                    os.makedirs(self.dist_dir)
                impl_tag, abi_tag, plat_tag = self.get_tag()
                archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
                wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
                os.rename(wheel_filename, wheel_path)
                print(f"[tmol] Using pre-built wheel: {wheel_path}")
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
                print(f"[tmol] Pre-built wheel not found ({e}). Building from source...")
                super().run()
except ImportError:
    CachedWheelsCommand = None


# ---------------------------------------------------------------------------
# Assemble extensions
# ---------------------------------------------------------------------------

ext_modules = []
if not SKIP_CUDA_BUILD:
    if CUDA_HOME is None:
        warnings.warn(
            "[tmol] CUDA_HOME not found. C++/CUDA extensions will not be built. "
            "Install a CUDA toolkit or set CUDA_HOME to enable compilation.",
            stacklevel=2,
        )
    else:
        ext_modules = _production_extensions()
        if not SKIP_TEST_EXTS:
            ext_modules += _test_extensions()

cmdclass = {"build_ext": NinjaBuildExtension}
if CachedWheelsCommand is not None:
    cmdclass["bdist_wheel"] = CachedWheelsCommand


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name=PACKAGE_NAME,
    ext_modules=ext_modules,
    cmdclass=cmdclass if ext_modules else {},
    # NOTE: build-time deps (packaging, psutil, ninja, torch) are declared in
    # pyproject.toml [build-system].requires — do NOT use setup_requires here.
)
