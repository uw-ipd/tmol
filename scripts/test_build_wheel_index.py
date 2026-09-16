"""Tests for the PEP 503 wheel index generated over the GitHub Releases."""

from __future__ import annotations

from pathlib import Path

import pytest

from build_wheel_index import NO_VARIANT, build, variant_of


def asset(name: str, digest: str | None = None) -> dict:
    return {
        "name": name,
        "browser_download_url": f"https://example.invalid/{name}",
        **({"digest": digest} if digest else {}),
    }


@pytest.mark.parametrize(
    "filename,variant",
    [
        ("tmol-0.1.55+cu130torch2.13-cp312-cp312-manylinux_2_28_x86_64.whl", "cu130torch2.13"),
        ("tmol-0.1.55+cputorch2.14-cp314-cp314-macosx_14_0_arm64.whl", "cputorch2.14"),
        ("tmol-0.1.55-cp312-cp312-manylinux_2_28_x86_64.whl", NO_VARIANT),
        ("tmol-0.1.55.tar.gz", None),
        ("release-notes.md", None),
    ],
)
def test_variant_comes_from_the_local_version_segment(filename, variant):
    assert variant_of(filename) == variant


def test_every_release_stays_resolvable_under_its_own_variant(tmp_path: Path):
    """A new release must not drop older versions out of the index."""
    counts = build(
        [
            asset("tmol-0.1.56+cu130torch2.13-cp312-cp312-manylinux_2_28_x86_64.whl"),
            asset("tmol-0.1.55+cu130torch2.13-cp312-cp312-manylinux_2_28_x86_64.whl"),
            asset("tmol-0.1.55+cputorch2.14-cp312-cp312-manylinux_2_28_x86_64.whl"),
            asset("tmol-0.1.55.tar.gz"),
        ],
        tmp_path,
        "tmol",
    )
    assert counts == {"cu130torch2.13": 2, "cputorch2.14": 1}

    page = (tmp_path / "cu130torch2.13" / "tmol" / "index.html").read_text()
    assert "tmol-0.1.55+cu130torch2.13" in page and "tmol-0.1.56+cu130torch2.13" in page
    # The variant sub-index must not leak wheels built for another torch.
    assert "cputorch2.14" not in page
    assert 'href="tmol/"' in (tmp_path / "cu130torch2.13" / "index.html").read_text()

    root = (tmp_path / "index.html").read_text()
    assert 'href="cu130torch2.13/"' in root and 'href="cputorch2.14/"' in root


def test_links_carry_the_hash_pip_checks(tmp_path: Path):
    build(
        [asset("tmol-0.1.55+cputorch2.14-cp312-cp312-manylinux_2_28_x86_64.whl", "sha256:" + "a" * 64)],
        tmp_path,
        "tmol",
    )
    page = (tmp_path / "cputorch2.14" / "tmol" / "index.html").read_text()
    assert page.count("#sha256=" + "a" * 64) == 1
