"""Binary wheels must constrain the ABI that their filenames advertise."""

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import set_release_wheel_version as script
from release_matrix import linux_wheel_rows, macos_wheel_rows


@pytest.mark.parametrize(
    "local_tag",
    sorted({row["local-tag"] for row in linux_wheel_rows() + macos_wheel_rows()}),
)
def test_wheel_requires_its_compiled_torch_minor(tmp_path, monkeypatch, local_tag):
    project = tmp_path / "pyproject.toml"
    project.write_bytes((Path(__file__).parents[1] / "pyproject.toml").read_bytes())
    monkeypatch.setattr(script, "PYPROJECT", project)
    monkeypatch.setattr(sys, "argv", ["set_release_wheel_version.py", local_tag])
    script.main()
    data = tomllib.loads(project.read_text())
    major, minor = map(int, local_tag.split("torch")[1].split("."))
    assert data["project"]["version"].endswith(f"+{local_tag}")
    for dependencies in [
        data["project"]["dependencies"],
        data["build-system"]["requires"],
    ]:
        requirement = next(
            Requirement(dep) for dep in dependencies if Requirement(dep).name == "torch"
        )
        assert f"{major}.{minor}.0" in requirement.specifier
        assert f"{major}.{minor}.1" in requirement.specifier
        assert f"{major}.{minor - 1}.0" not in requirement.specifier
        assert f"{major}.{minor + 1}.0" not in requirement.specifier


def test_invalid_tag_leaves_project_unchanged(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    original = (Path(__file__).parents[1] / "pyproject.toml").read_bytes()
    project.write_bytes(original)
    monkeypatch.setattr(script, "PYPROJECT", project)
    monkeypatch.setattr(sys, "argv", ["set_release_wheel_version.py", "cpu"])
    with pytest.raises(SystemExit, match="PyTorch minor"):
        script.main()
    assert project.read_bytes() == original
