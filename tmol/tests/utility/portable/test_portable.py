"""Guards for the portable math layer shared with the (future) Metal backend.

These compile C++ in a subprocess rather than going through a pybind module:
what they assert is a property of the *headers* -- that tmol's kernel-evaluated
math neither reaches Eigen nor the C++ standard library's containers -- which a
compiled extension could not express.

Two tiers, because the math is in two states:

* Tier 1 (``portable_cone.cc``) compiles against nothing but the portable
  layer. No Eigen, no torch, no std containers: the state a Metal translation
  unit requires.
* Tier 2 (``portable_cone_tensor.cc``) covers math that still reads through
  TView/TensorAccessor. It cannot be freestanding until TView gains a Metal
  accessor, but it is already Eigen-free and must stay that way.
"""

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
POISON = HERE / "poison"

requires_cxx = pytest.mark.skipif(
    shutil.which("c++") is None, reason="no host C++ compiler available"
)


def _torch_flags():
    """Tier 2 reaches the tensor layer, which needs torch's headers."""
    import torch

    inc = Path(torch.__file__).parent / "include"
    return [
        f"-I{inc}",
        f"-I{inc / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{sysconfig.get_paths()['include']}",
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
    ]


def _compile(source, out, includes=(), defines=(), run=False):
    cmd = [
        shutil.which("c++"),
        "-std=c++17",
        "-O1",
        *defines,
        *(f"-I{inc}" for inc in includes),
        f"-I{ROOT}",
        f"-I{ROOT / 'tmol' / 'extern'}",
        str(source),
    ]
    cmd += ["-o", str(out)] if run else ["-fsyntax-only"]
    return subprocess.run(cmd, capture_output=True, text=True)


@requires_cxx
def test_freestanding_math_needs_only_the_portable_layer(tmp_path):
    """Tier 1: compiles with no Eigen, no torch, nothing but the portable layer."""
    proc = _compile(
        HERE / "portable_cone.cc",
        tmp_path / "cone",
        includes=[POISON],
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_tensor_dependent_math_does_not_reach_eigen(tmp_path):
    """Tier 2: Eigen must not reach math that still uses the tensor layer."""
    proc = _compile(
        HERE / "portable_cone_tensor.cc",
        tmp_path / "cone_tensor",
        includes=[POISON],
        defines=_torch_flags(),
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_guard_trips_when_eigen_is_reintroduced(tmp_path):
    """The guard must actually fail -- a guard that cannot fail proves nothing."""
    src = tmp_path / "regressed.cc"
    src.write_text(
        "#include <Eigen/Core>\n" + (HERE / "portable_cone.cc").read_text()
    )
    proc = _compile(
        src, tmp_path / "regressed", includes=[POISON]
    )
    assert proc.returncode != 0
    assert "must stay portable" in proc.stderr


@requires_cxx
def test_portable_headers_use_no_std_containers():
    """Metal has no standard library; the portable headers may not need one."""
    banned = (
        "#include <tuple>",
        "#include <vector>",
        "#include <algorithm>",
        "#include <type_traits>",
        "std::tuple",
        "std::vector",
    )
    for name in ("qualifiers.hh", "math.hh", "matrix.hh", "tuple.hh"):
        text = (ROOT / "tmol" / "utility" / "portable" / name).read_text()
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("//")
        )
        for token in banned:
            assert token not in code, f"{name} uses {token}"


@requires_cxx
def test_portable_matrix_matches_eigen(tmp_path):
    """Byte layout identical to Eigen; arithmetic equal to within rounding."""
    binary = tmp_path / "equivalence"
    proc = _compile(
        HERE / "portable_equivalence.cc",
        binary,
        includes=[sysconfig.get_paths()["include"]],
        run=True,
    )
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
