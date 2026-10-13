"""Native packages preserve versioned, non-overlapping backend ownership."""

import tomllib
from pathlib import Path
from packaging.requirements import Requirement


def requirement(project, name):
    return next(
        Requirement(value)
        for value in project["dependencies"]
        if Requirement(value).name == name
    )


def test_native_package_metadata():
    root = Path(__file__).resolve().parents[1]
    base = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = base["version"]
    base_names = {Requirement(value).name for value in base["dependencies"]}

    assert base_names.isdisjoint({"torch", "jax", "tmol-jax"})
    assert base["optional-dependencies"]["torch"] == ["torch>=2.11"]
    assert base["optional-dependencies"]["jax"] == ["jax>=0.11.2"]

    for major in (12, 13):
        project = tomllib.loads(
            (root / f"packages/cu{major}/pyproject.toml").read_text()
        )["project"]
        assert project["name"] == f"tmol-ops-cu{major}"
        assert project["version"] == version
        assert str(requirement(project, "tmol").specifier) == f"=={version}"
        assert str(requirement(project, "torch").specifier) == ">=2.11"
        assert base["optional-dependencies"][f"cu{major}"] == [
            f"{project['name']}=={version}"
        ]
