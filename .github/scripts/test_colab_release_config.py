"""Static checks for the Colab bootstrap and release wheel matrices."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def _load_colab_setup():
    path = ROOT / "docs/tutorial/colab_setup.py"
    spec = importlib.util.spec_from_file_location("_test_colab_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_release_matrix():
    path = ROOT / "scripts/release_matrix.py"
    spec = importlib.util.spec_from_file_location("_test_release_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_colab_selects_the_published_wheel_for_each_supported_python():
    module = _load_colab_setup()
    source = (ROOT / "docs/tutorial/colab_setup.py").read_text(encoding="utf-8")

    assert module.TUTORIAL_REF == "master"
    assert module.TMOL_RELEASE == "0.1.54"
    assert module.RELEASE_WHEEL_TORCH_MINOR == "2.11"
    assert module.RELEASE_WHEEL_CUDA == "12.8"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert module.TMOL_RELEASE == project["project"]["version"]
    assert (
        project["tool"]["scikit-build"]["cmake"]["define"]["CMAKE_CUDA_ARCHITECTURES"][
            "default"
        ]
        == "native"
    )
    assert module.RELEASE_WHEEL_PYTHONS == {(3, 12), (3, 13)}
    assert module._wheel_url((3, 12)).endswith(
        "/v0.1.54/" "tmol-0.1.54+cu128torch2.11-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    assert module._wheel_url((3, 13)).endswith(
        "/v0.1.54/" "tmol-0.1.54+cu128torch2.11-cp313-cp313-manylinux_2_28_x86_64.whl"
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


def test_release_matrix_drives_publish_smoke_and_manifest():
    matrix = _load_release_matrix()
    publish_rows = matrix.gpu_wheel_rows()
    smoke_rows = matrix.linux_wheel_rows()
    expected_keys = matrix.expected_wheel_keys()

    assert len(publish_rows) == 34
    assert len(smoke_rows) == 50
    assert len(expected_keys) == 58
    assert len(
        {(row["python-tag"], row["local-tag"], row["arch"]) for row in smoke_rows}
    ) == len(smoke_rows)

    colab_rows = [row for row in publish_rows if row.get("cuda-archs")]
    assert {
        (row["python-version"], row["local-tag"], row["cuda-archs"])
        for row in colab_rows
    } == {
        ("3.12", "cu128torch2.11", "75;80;89"),
        ("3.13", "cu128torch2.11", "75;80;89"),
    }
    assert {
        (row["python-version"], row["arch"])
        for row in publish_rows
        if row["local-tag"] == "cu132torch2.14"
    } == {
        (python_version, arch)
        for python_version in matrix.PYTHON_VERSIONS
        for arch in matrix.LINUX_ARCHES
    }
    # This release lane was previously absent from the complete smoke matrix.
    assert any(
        row["python-version"] == "3.12"
        and row["local-tag"] == "cu130torch2.12"
        and row["arch"] == "x86_64"
        for row in smoke_rows
    )
    assert {
        row["runs-on"]
        for row in publish_rows
        if row["local-tag"] == "cu132torch2.12" and row["arch"] == "x86_64"
    } == {"self-hosted"}
    assert {
        row["runs-on"]
        for row in smoke_rows
        if row["local-tag"] == "cu132torch2.12" and row["arch"] == "x86_64"
    } == {"ubuntu-22.04"}

    build_template = _workflow(".github/workflows/_build_wheel.yml")
    cuda_archs = build_template[True]["workflow_call"]["inputs"]["cuda-archs"]
    assert "75;80;86;89;90" in cuda_archs["description"]
    assert "every sm_75+ target reported by nvcc" in cuda_archs["description"]

    publish = _workflow(".github/workflows/publish.yml")
    verify = publish["jobs"]["verify_release_version"]
    assert verify["outputs"]["gpu_matrix"] == "${{ steps.matrix.outputs.gpu }}"
    assert verify["outputs"]["linux_cpu_matrix"] == (
        "${{ steps.matrix.outputs.linux_cpu }}"
    )
    assert verify["outputs"]["macos_matrix"] == "${{ steps.matrix.outputs.macos }}"
    assert publish["jobs"]["build_wheels"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.verify_release_version.outputs.gpu_matrix) }}"
    )
    manifest_step = next(
        step
        for step in publish["jobs"]["upload"]["steps"]
        if step.get("name") == "Validate release wheel manifest"
    )
    assert manifest_step["run"] == "python scripts/validate_release_manifest.py wheels"
    assert publish["jobs"]["build_cpu_wheel"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.verify_release_version.outputs.linux_cpu_matrix) }}"
    )
    assert publish["jobs"]["build_macos_cpu_wheel"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.verify_release_version.outputs.macos_matrix) }}"
    )

    version_step = next(
        step
        for step in publish["jobs"]["upload"]["steps"]
        if step.get("name") == "Determine version from sdist filename"
    )
    assert "Version(sys.argv[1]).is_prerelease" in version_step["run"]

    smoke = _workflow(".github/workflows/wheel-smoke.yml")
    assert smoke["jobs"]["build"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.release_matrix.outputs.linux) }}"
    )
    assert smoke["jobs"]["test"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.release_matrix.outputs.linux) }}"
    )
    expected_cpu_smoke = {("2.13", "x86_64"), ("2.14", "aarch64")}
    for job in ("build-linux-cpu", "test-linux-cpu"):
        assert {
            (row["torch-version"], row["arch"])
            for row in smoke["jobs"][job]["strategy"]["matrix"]["include"]
        } == expected_cpu_smoke
    assert {
        row["torch-version"]
        for row in smoke["jobs"]["build-linux-cuda"]["strategy"]["matrix"]["include"]
    } == {"2.14"}


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
