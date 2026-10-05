"""Check the source metadata before making a release visible on PyPI."""

import io
import tarfile

import pytest

from validate_pypi_sdist import validate_sdist


def make_sdist(tmp_path, *, version="1.2.3", metadata_version=None, requirements=()):
    path = tmp_path / f"tmol-{version}.tar.gz"
    metadata = (
        "Metadata-Version: 2.4\nName: tmol\n"
        f"Version: {metadata_version or version}\n"
        + "".join(f"Requires-Dist: {req}\n" for req in requirements)
        + "\n"
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(f"tmol-{version}/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return path


def test_accepts_index_dependencies_and_optional_extras(tmp_path):
    validate_sdist(
        make_sdist(
            tmp_path,
            requirements=("atomworks>=3,<4", 'pytest>=8; extra == "dev"'),
        )
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "atomworks @ git+https://github.com/RosettaCommons/atomworks.git@main",
        'testdata @ https://example.org/data.whl ; extra == "dev"',
    ],
)
def test_rejects_direct_dependencies_including_extras(tmp_path, requirement):
    with pytest.raises(ValueError, match="direct URL"):
        validate_sdist(make_sdist(tmp_path, requirements=(requirement,)))


def test_rejects_local_version(tmp_path):
    with pytest.raises(ValueError, match="public tmol source version"):
        validate_sdist(make_sdist(tmp_path, version="1.2.3+cputorch2.14"))


def test_rejects_mismatched_version(tmp_path):
    with pytest.raises(ValueError, match="versions differ"):
        validate_sdist(make_sdist(tmp_path, metadata_version="1.2.4"))
