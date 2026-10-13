import os
from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


REPOSITORY = Path(__file__).parents[2]


def _project_metadata(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _requirement_names(requirements):
    return {
        re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0].lower()
        for requirement in requirements
    }


def test_preparation_import_needs_neither_tensor_framework():
    script = r"""
import builtins
import importlib.util
import sys

real_find_spec = importlib.util.find_spec
real_import = builtins.__import__


def without_tensor_frameworks(name, *args, **kwargs):
    if name in {"torch", "jax"}:
        return None
    return real_find_spec(name, *args, **kwargs)


def block_tensor_frameworks(name, *args, **kwargs):
    root = name.partition(".")[0]
    if root in {"torch", "jax"}:
        raise ModuleNotFoundError(
            f"No module named {root!r}", name=root
        )
    return real_import(name, *args, **kwargs)


importlib.util.find_spec = without_tensor_frameworks
builtins.__import__ = block_tensor_frameworks
import tmol
from tmol.preparation import prepare_disulfide_topology

assert callable(prepare_disulfide_topology)
assert "torch" not in sys.modules
assert "jax" not in sys.modules

try:
    tmol.has_cuda
except ModuleNotFoundError as error:
    assert error.name == "torch"
    assert "requires the Torch frontend" in str(error)
else:
    raise AssertionError("tmol.has_cuda unexpectedly loaded without Torch")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
    )


def test_current_frontend_distributions_have_independent_dependencies():
    torch_metadata = _project_metadata(REPOSITORY / "pyproject.toml")
    jax_metadata = _project_metadata(REPOSITORY / "packages/jax/pyproject.toml")

    torch_project = torch_metadata["project"]
    jax_project = jax_metadata["project"]
    torch_requirements = _requirement_names(torch_project["dependencies"])
    jax_requirements = _requirement_names(jax_project["dependencies"])

    assert torch_project["name"] == "tmol"
    assert "torch" in torch_requirements
    assert "jax" not in torch_requirements

    assert jax_project["name"] == "tmol-jax"
    assert "jax" in jax_requirements
    assert "torch" not in jax_requirements
    assert "tmol" not in jax_requirements

    assert torch_metadata["tool"]["scikit-build"]["wheel"]["packages"] == ["tmol"]
    assert jax_metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "tmol_jax*"
    ]


def test_optional_packages_preserve_disjoint_framework_ownership():
    """Static ownership contract; this is not a built-wheel install proof."""
    jax_metadata = _project_metadata(REPOSITORY / "packages/jax/pyproject.toml")
    jax_cuda_metadata = _project_metadata(
        REPOSITORY / "packages/jax-cuda/pyproject.toml"
    )
    assert jax_metadata["project"]["optional-dependencies"]["native-cuda"] == [
        "tmol-jax-cuda"
    ]
    assert jax_cuda_metadata["project"]["name"] == "tmol-jax-cuda"
    assert jax_cuda_metadata["project"]["dynamic"] == ["version", "dependencies"]
    assert jax_cuda_metadata["tool"]["setuptools"]["packages"]["find"][
        "include"
    ] == ["tmol_jax_cuda"]

    jax_cuda_setup = (REPOSITORY / "packages/jax-cuda/setup.py").read_text(
        encoding="utf-8"
    )
    assert '"jax>=0.11.2"' in jax_cuda_setup
    assert 'f"tmol-jax==' in jax_cuda_setup
    assert '"torch' not in jax_cuda_setup

    for major in (12, 13):
        metadata = _project_metadata(
            REPOSITORY / f"packages/cu{major}/pyproject.toml"
        )
        requirements = _requirement_names(metadata["project"]["dependencies"])
        assert requirements == {"tmol", "torch"}
        assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == []


def test_shared_library_install_waits_for_single_neutral_owner():
    """Keep the common ABI out of both adapters until base tmol is neutral."""
    shared_cmake = (
        REPOSITORY / "tmol/score/disulfide/potentials/shared/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "add_library(tmol_disulfide_shared_cpu SHARED" in shared_cmake
    assert "install(TARGETS tmol_disulfide_shared_cpu" not in shared_cmake
    assert "install(TARGETS tmol_disulfide_shared_cuda" not in shared_cmake
