"""Prepared metadata must describe the wheel the frontend actually installs."""

import shutil
import zipfile
from pathlib import Path

from pyproject_hooks import BuildBackendHookCaller


def test_frontend_metadata_matches_fetched_abi_wheel(tmp_path):
    dist_info = "tmol-0.1.56+cputorch2.13.dist-info"
    wheel = "tmol-0.1.56+cputorch2.13-py3-none-any.whl"
    expected = (
        "Metadata-Version: 2.2\nName: tmol\nVersion: 0.1.56+cputorch2.13\n"
        "Requires-Dist: torch==2.13.*\n"
    ).encode()
    shutil.copy(Path(__file__).parents[1] / "tmol_build_backend.py", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="tmol"\nversion="0.1.56"\ndependencies=["torch>=2.8"]\n'
        '[build-system]\nrequires=["scikit-build-core"]\n'
        'build-backend="fixture_backend"\nbackend-path=["."]\n'
    )
    with zipfile.ZipFile(tmp_path / wheel, "w") as archive:
        archive.writestr(dist_info + "/METADATA", expected)
        archive.writestr(
            dist_info + "/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(dist_info + "/RECORD", "")
    # Only replace the download transport; execute the real backend hooks and
    # standard frontend fallback in subprocesses, without network/native builds.
    (tmp_path / "fixture_backend.py").write_text(
        "from tmol_build_backend import *\n"
        "import tmol_build_backend as backend\nfrom pathlib import Path\nimport shutil\n"
        "backend._is_repo_checkout = lambda: False\n"
        f"backend._candidate_wheel_filenames = lambda: [{wheel!r}]\n"
        "def download(url, path):\n"
        f"    shutil.copy(Path(__file__).parent / {wheel!r}, path)\n"
        "    return True\n"
        "backend._download_to_path = download\n"
    )
    metadata, wheels = tmp_path / "metadata", tmp_path / "wheels"
    metadata.mkdir()
    wheels.mkdir()
    frontend = BuildBackendHookCaller(tmp_path, "fixture_backend", backend_path=["."])
    prepared = metadata / frontend.prepare_metadata_for_build_wheel(str(metadata))
    filename = frontend.build_wheel(str(wheels), metadata_directory=str(prepared))
    with zipfile.ZipFile(wheels / filename) as archive:
        assert archive.read(dist_info + "/METADATA") == expected
    assert (prepared / "METADATA").read_bytes() == expected
