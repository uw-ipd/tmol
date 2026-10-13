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
    assert "tmol[torch]" in str(error)
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
    build_requirements = _requirement_names(torch_metadata["build-system"]["requires"])
    torch_extra = _requirement_names(torch_project["optional-dependencies"]["torch"])
    jax_extra = _requirement_names(torch_project["optional-dependencies"]["jax"])
    jax_requirements = _requirement_names(jax_project["dependencies"])

    assert torch_project["name"] == "tmol"
    assert torch_project["version"] == "0.1.64"
    assert torch_project["optional-dependencies"]["torch"] == ["torch>=2.11"]
    assert torch_project["optional-dependencies"]["jax"] == ["jax>=0.11.2"]
    assert torch_requirements.isdisjoint({"torch", "jax", "tmol-jax"})
    assert torch_extra == {"torch"}
    assert jax_extra == {"jax"}
    assert torch_extra | jax_extra == {"torch", "jax"}
    assert {"torch", "jax"} <= build_requirements

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
    assert jax_cuda_metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "tmol_jax_cuda"
    ]

    jax_cuda_setup = (REPOSITORY / "packages/jax-cuda/setup.py").read_text(
        encoding="utf-8"
    )
    assert '"jax>=0.11.2"' in jax_cuda_setup
    assert 'f"tmol-jax==' in jax_cuda_setup
    assert '"torch' not in jax_cuda_setup

    for major in (12, 13):
        metadata = _project_metadata(REPOSITORY / f"packages/cu{major}/pyproject.toml")
        requirements = _requirement_names(metadata["project"]["dependencies"])
        assert requirements == {"tmol", "torch"}
        assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == []


def test_base_package_is_single_shared_library_owner():
    """The base wheel owns one common ABI and its two dormant adapters."""
    shared_cmake = (
        REPOSITORY / "tmol/score/disulfide/potentials/shared/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    root_cmake = (REPOSITORY / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(tmol_disulfide_shared_cpu SHARED" in shared_cmake
    assert (
        "install(TARGETS tmol_disulfide_shared_cpu LIBRARY DESTINATION tmol)"
        in shared_cmake
    )
    assert "install(TARGETS tmol_disulfide_shared_cuda" in shared_cmake
    assert 'DESTINATION "${_TMOL_NATIVE_PACKAGE}"' in shared_cmake
    assert (
        "install(TARGETS tmol_disulfide_shared_torch LIBRARY DESTINATION tmol)"
        in root_cmake
    )
    assert (
        "install(TARGETS tmol_disulfide_shared_jax LIBRARY DESTINATION tmol)"
        in root_cmake
    )
    for package in ("jax", "jax-cuda", "cu12", "cu13"):
        package_root = REPOSITORY / "packages" / package
        for cmake in package_root.rglob("CMakeLists.txt"):
            text = cmake.read_text(encoding="utf-8")
            assert "tmol_disulfide_shared_cpu" not in text


def test_future_install_migration_is_explicit_and_unreleased():
    documentation = (REPOSITORY / "docs/installation.md").read_text(encoding="utf-8")
    for command in (
        'pip install "tmol[torch]"',
        'pip install "tmol[jax]"',
        'pip install "tmol[torch,jax]"',
    ):
        assert command in documentation
    assert "will not be republished with different dependencies" in documentation
    assert "not yet a working JAX install" in documentation
    assert "namespace and adapter migration" in documentation
    assert "not implemented yet" in documentation
