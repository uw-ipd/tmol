"""Check the renamed candidate's pip identity without a native CUDA build."""

from importlib.metadata import distributions
import json
import subprocess
import sys

from pypi_cuda_wheels import pypi_gpu_wheel
from release_artifacts import write_index
from staging_index import serve
from test_release_artifacts import wheel


def test_pip_installs_renamed_cuda_candidate(tmp_path):
    source = wheel(
        tmp_path / "source",
        "9.9.9+cu130torch2.13",
        "==2.13.*",
        tag="cp312-cp312-manylinux_2_28_x86_64",
    )
    candidate = pypi_gpu_wheel(source, tmp_path / "files")
    installed = tmp_path / "installed"
    report = tmp_path / "report.json"
    with serve(tmp_path) as url:
        write_index(
            [candidate],
            tmp_path / "simple/tmol-cu130-torch213/index.html",
            f"{url}/files",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-deps",
                "--no-cache-dir",
                "--ignore-installed",
                "--only-binary=:all:",
                "--platform=manylinux_2_28_x86_64",
                "--python-version=3.12",
                "--target",
                str(installed),
                "--report",
                str(report),
                "--index-url",
                f"{url}/simple",
                "tmol-cu130-torch213==9.9.9",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        selected = json.loads(report.read_text())["install"][0]
        assert selected["download_info"]["url"] == f"{url}/files/{candidate.name}"
    metadata = next(distributions(path=[str(installed)])).metadata
    assert metadata["Name"] == "tmol-cu130-torch213"
    assert metadata["Version"] == "9.9.9"
    assert metadata.get_all("Requires-Dist") == ["torch==2.13.*"]
    assert (installed / "tmol/_C.so").read_bytes() == bytes(range(256))
    assert (
        installed / "tmol/database/default/params.bin"
    ).read_bytes() == b"runtime data"
    assert not (installed / "tmol/tests").exists()
