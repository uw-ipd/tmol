"""Static checks for the Colab bootstrap and release wheel matrices."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from packaging.version import Version

ROOT = Path(__file__).parents[2]


def _load_colab_setup():
    path = ROOT / "docs/tutorial/notebook_setup.py"
    spec = importlib.util.spec_from_file_location("_test_notebook_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_colab_uses_a_released_version_and_current_tutorials():
    module = _load_colab_setup()
    source = (ROOT / "docs/tutorial/notebook_setup.py").read_text(encoding="utf-8")

    assert module.TUTORIAL_REF == "master"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    # Colab needs published wheels; the source version can be unreleased.
    release = module.TMOL_RELEASE
    assert Version(release) <= Version(project["project"]["version"])
    assert (
        project["tool"]["scikit-build"]["cmake"]["define"]["CMAKE_CUDA_ARCHITECTURES"][
            "default"
        ]
        == "native"
    )
    assert "install_tutorial_source" not in source
    assert "git+https://github.com/uw-ipd/tmol.git" not in source
    assert "CMAKE_CUDA_ARCHITECTURES" not in source
    assert "_tmol_tutorial_visualize.py" not in source
    for notebook_path in sorted((ROOT / "docs/tutorial").glob("[0-9][0-9]_*.ipynb")):
        notebook = notebook_path.read_text(encoding="utf-8")
        assert "install_tutorial_source" not in notebook
        assert (
            f"/uw-ipd/tmol/blob/master/docs/tutorial/{notebook_path.name}" in notebook
        )
        assert "kdidi/sphinx-docs-refactor" not in notebook


def test_colab_pip_install_constrains_active_torch(monkeypatch):
    module = _load_colab_setup()

    def check_call(command):
        constraint = Path(command[command.index("--constraint") + 1])
        assert constraint.read_text(encoding="utf-8") == "torch==2.11.0+cu128\n"
        assert command[-1] == "numpy>=1.24"
        assert "--only-binary=:all:" in command

    monkeypatch.setattr(module.subprocess, "check_call", check_call)
    module._pip_install(["numpy>=1.24"], "2.11.0+cu128")


@pytest.mark.parametrize(
    "torch_version,cuda,extra",
    [
        ("2.11.0+cu128", "12.8", "cu12"),
        ("2.13.0+cu129", "12.9", "cu12"),
        ("2.14.1+cu132", "13.2", "cu13"),
    ],
)
def test_colab_selects_cuda_major_and_preserves_torch(
    monkeypatch, torch_version, cuda, extra
):
    module = _load_colab_setup()
    torch = SimpleNamespace(
        __version__=torch_version,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda _: "test GPU"
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []
    monkeypatch.setattr(
        module, "_pip_install", lambda reqs, version: calls.append((reqs, version))
    )
    module.setup_colab([])
    assert len(calls) == 1
    assert calls[0][0][0] == f"tmol[{extra}]=={module.TMOL_RELEASE}"
    assert calls[0][1] == torch_version


@pytest.mark.parametrize(
    "version,cuda,available",
    [("2.10.0", "12.8", True), ("2.14.1", "11.8", True), ("2.14.1", None, False)],
)
def test_colab_rejects_unsupported_runtime_before_install(
    monkeypatch, version, cuda, available
):
    module = _load_colab_setup()
    torch = SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: available),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        module, "_pip_install", lambda *args: pytest.fail("must not install")
    )
    with pytest.raises(RuntimeError, match="PyTorch"):
        module.setup_colab([])


def _tutorial_notebooks() -> list[Path]:
    notebooks = sorted((ROOT / "docs/tutorial").glob("[0-9][0-9]_*.ipynb"))
    assert notebooks, "no tutorial notebooks found in docs/tutorial"
    return notebooks


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_colab_declares_the_repository_files_used_by_each_tutorial():
    module = _load_colab_setup()
    declared = set(module.FIXTURES)
    assert all((ROOT / "tmol/tests/data" / fixture).is_file() for fixture in declared)

    for path in _tutorial_notebooks():
        referenced = set()
        for node in ast.walk(ast.parse(_notebook_code(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "data_path"
            ):
                parts = [ast.literal_eval(arg) for arg in node.args]
                referenced.add("/".join(parts))
        assert referenced <= declared, (path.name, referenced - declared)


def test_notebook_setup_exports_resolve_to_real_tmol_names():
    module = _load_colab_setup()
    for name in module.__all__:
        if name in module._SOURCES:
            package = module._SOURCES[name].replace(".", "/")
            init = (ROOT / package / "__init__.py").read_text(encoding="utf-8")
            assert re.search(rf"\b{name}\b", init), (name, module._SOURCES[name])
        else:
            assert name in module._ALIASES or hasattr(module, name), name


def test_tutorials_use_the_shared_setup_and_do_not_import_test_data():
    for path in _tutorial_notebooks():
        code = _notebook_code(path)
        assert "from notebook_setup import *" in code, path.name
        assert "device, param_db = setup()" in code, path.name
        assert "tmol.tests" not in code, path.name
        ast.parse(code)


def test_docs_workflow_uses_hosted_cpu_and_gpu_ci_executes_gpu_cells():
    docs = _workflow(".github/workflows/docs.yml")
    job = docs["jobs"]["docs"]
    assert job["runs-on"] == "ubuntu-latest"
    execution_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "execute tutorials and build docs"
    )
    assert execution_step["run"] == ".github/ci/run_docs_cpu_work.sh"

    gpu_script = (ROOT / ".github/ci/run_ci_gpu_work.sh").read_text(encoding="utf-8")
    assert "--gpu-smoke" in gpu_script
    assert "--execution-device cuda" in gpu_script
    flagged = [
        path.name
        for path in _tutorial_notebooks()
        if json.loads(path.read_text(encoding="utf-8"))["metadata"]
        .get("tmol", {})
        .get("gpu_smoke")
    ]
    # Every tutorial must run on CUDA: CPU-only runs hide device-mismatch bugs.
    assert flagged == [path.name for path in _tutorial_notebooks()]
