"""Static checks for the Colab bootstrap and release wheel matrices."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

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


def test_colab_uses_published_v0147_wheel_without_source_fallback():
    module = _load_colab_setup()
    source = (ROOT / "docs/tutorial/colab_setup.py").read_text(encoding="utf-8")

    assert module.TUTORIAL_REF == "master"
    assert module.RELEASE_WHEEL_TORCH_MINOR == "2.11"
    assert module.RELEASE_WHEEL_CUDA == "12.8"
    assert module.RELEASE_WHEEL_PYTHON == (3, 12)
    assert module.TMOL_WHEEL.endswith(
        "/v0.1.47/" "tmol-0.1.47+cu128torch2.11-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    assert "install_tutorial_source" not in source
    assert "git+https://github.com/uw-ipd/tmol.git" not in source
    assert "CMAKE_CUDA_ARCHITECTURES" not in source
    assert "_tmol_tutorial_visualize.py" not in source
    for notebook_path in sorted((ROOT / "docs/tutorial").glob("[0-9][0-9]_*.ipynb")):
        notebook = notebook_path.read_text(encoding="utf-8")
        assert "install_tutorial_source" not in notebook
        assert "/uw-ipd/tmol/blob/master/docs/tutorial/" in notebook
        assert "kdidi/sphinx-docs-refactor" not in notebook


def test_colab_pip_install_constrains_active_torch(monkeypatch):
    module = _load_colab_setup()

    def check_call(command):
        constraint = Path(command[command.index("--constraint") + 1])
        assert constraint.read_text(encoding="utf-8") == "torch==2.11.0+cu128\n"
        assert command[-1] == "numpy>=1.24"

    monkeypatch.setattr(module.subprocess, "check_call", check_call)
    module._pip_install(["numpy>=1.24"], "2.11.0+cu128")


def test_colab_release_lanes_match_publish_and_smoke_matrices():
    build_template = _workflow(".github/workflows/_build_wheel.yml")
    cuda_archs = build_template[True]["workflow_call"]["inputs"]["cuda-archs"]
    assert "75;80;86;89;90" in cuda_archs["description"]
    assert "80;86;89;90;100" in cuda_archs["description"]

    publish = _workflow(".github/workflows/publish.yml")
    publish_rows = publish["jobs"]["build_wheels"]["strategy"]["matrix"]["include"]
    assert len(publish_rows) == 25

    colab_rows = [row for row in publish_rows if row.get("cuda-archs") == "75;80;89"]
    assert colab_rows == [
        {
            "python-version": "3.12",
            "container-image": "nvcr.io/nvidia/pytorch:25.02-py3",
            "torch-version": "2.11",
            "torch-package-version": "2.11.0",
            "pip-torch-cuda-url": "https://download.pytorch.org/whl/cu128",
            "cuda-archs": "75;80;89",
            "runs-on": "ubuntu-22.04",
            "label": "py3.12 pt2.11 x86_64 Colab cu128",
        }
    ]
    assert any(
        row["torch-version"] == "2.8"
        and row["pip-torch-cuda-url"].endswith("/cu129")
        and row["runs-on"] == "ubuntu-22.04"
        for row in publish_rows
    )

    manifest_step = next(
        step
        for step in publish["jobs"]["upload"]["steps"]
        if step.get("name") == "Validate release wheel manifest"
    )
    assert "--gpu-count 25" in manifest_step["run"]
    assert "--cpu-count 8" in manifest_step["run"]
    assert "--require cu128torch2.11:cp312:x86_64" in manifest_step["run"]
    assert "--require cu128torch2.8:cp312:x86_64" not in manifest_step["run"]

    smoke = _workflow(".github/workflows/wheel-smoke.yml")
    build_rows = smoke["jobs"]["build"]["strategy"]["matrix"]["include"]
    test_rows = smoke["jobs"]["test"]["strategy"]["matrix"]["include"]
    assert len(build_rows) == len(test_rows) == 32
    assert any(
        row.get("local-tag") == "cu128torch2.11" and row.get("cuda-archs") == "75;80;89"
        for row in build_rows
    )
    assert any(row.get("local-tag") == "cu128torch2.11" for row in test_rows)
    assert any(row.get("local-tag") == "cu129torch2.8" for row in build_rows)
    assert any(row.get("local-tag") == "cu129torch2.8" for row in test_rows)


def test_docs_workflow_executes_gpu_only_notebook_cells():
    docs = _workflow(".github/workflows/docs.yml")
    job = docs["jobs"]["docs"]
    assert job["runs-on"] == ["self-hosted", "gpu"]
    execution_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute tutorials and build docs"
    )
    assert ".github/ci/srun_gpu_retry.sh" in execution_step["run"]

    script = (ROOT / ".github/ci/run_docs_gpu_work.sh").read_text(encoding="utf-8")
    assert "02_gpu_batching.ipynb" in script
    assert "06_fast_relax.ipynb" in script
    assert "--execution-device cuda" in script
