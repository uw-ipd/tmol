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
