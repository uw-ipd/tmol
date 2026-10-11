"""Static checks for the Colab bootstrap and release wheel matrices."""

from __future__ import annotations

import ast
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

ROOT = Path(__file__).parents[2]


def _load_colab_setup():
    path = ROOT / "docs/tutorial/colab_setup.py"
    spec = importlib.util.spec_from_file_location("_test_colab_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_colab_selects_the_published_wheel_for_each_supported_python():
    module = _load_colab_setup()
    source = (ROOT / "docs/tutorial/colab_setup.py").read_text(encoding="utf-8")

    assert module.TUTORIAL_REF == "master"
    assert module.RELEASE_WHEEL_TORCH_MINOR == "2.11"
    assert module.RELEASE_WHEEL_CUDA == "12.8"
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
    assert module.RELEASE_WHEEL_PYTHONS == {(3, 12), (3, 13)}
    assert module._wheel_url((3, 12)).endswith(
        f"/v{release}/tmol-{release}+cu128torch2.11-"
        "cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    assert module._wheel_url((3, 13)).endswith(
        f"/v{release}/tmol-{release}+cu128torch2.11-"
        "cp313-cp313-manylinux_2_28_x86_64.whl"
    )
    with pytest.raises(ValueError, match="Unsupported Colab Python version"):
        module._wheel_url((3, 14))
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

    monkeypatch.setattr(module.subprocess, "check_call", check_call)
    module._pip_install(["numpy>=1.24"], "2.11.0+cu128")


def test_colab_without_nvidia_smi_explains_how_to_select_a_gpu(monkeypatch):
    module = _load_colab_setup()

    def missing_command(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(module.subprocess, "run", missing_command)
    with pytest.raises(RuntimeError, match="Change runtime type > T4 GPU"):
        module.setup_colab([])


def test_colab_declares_the_repository_files_used_by_each_tutorial():
    for path in sorted((ROOT / "docs/tutorial").glob("[0-9][0-9]_*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        nodes = list(ast.walk(ast.parse(code)))
        declared = set()
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setup_colab"
            ):
                declared.update(ast.literal_eval(node.args[0]))
        referenced = {
            node.value
            for node in nodes
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("tmol/tests/data/")
            and (ROOT / node.value).is_file()
        }
        assert referenced <= declared, (path.name, referenced - declared)
        assert all((ROOT / fixture).is_file() for fixture in declared)


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
    assert "02_gpu_batching.ipynb" in gpu_script
    assert "06_fast_relax.ipynb" in gpu_script
    assert "--execution-device cuda" in gpu_script
