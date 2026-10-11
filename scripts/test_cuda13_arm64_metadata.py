"""The NVIDIA workaround must preserve binaries and refuse unfamiliar packages."""

import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path

import pytest

import repair_cuda13_arm64_metadata as repair


@pytest.fixture
def distribution(tmp_path, monkeypatch):
    info = tmp_path / "nvidia_cusparselt_cu13-0.8.0.dist-info"
    info.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: nvidia-cusparselt-cu13\nVersion: 0.8.0\n"
    wheel = (
        b"Wheel-Version: 1.0\nGenerator: setuptools (80.9.0)\n"
        b"Root-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n\n"
    )
    assert hashlib.sha256(wheel).hexdigest() == repair.WHEEL_SHA256
    library = b"\x7fELF\x02\x01" + bytes(12) + b"\xb7\x00" + bytes(range(256))
    contents = {
        f"{info.name}/METADATA": metadata,
        f"{info.name}/WHEEL": wheel,
        repair.LIBRARY: library,
        f"{info.name}/INSTALLER": b"pip\n",
    }
    for name, data in contents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(
        [
            [name, repair.record_digest(data), len(data)]
            for name, data in contents.items()
        ]
        + [[f"{info.name}/RECORD", "", ""]]
    )
    (info / "RECORD").write_text(buffer.getvalue())
    monkeypatch.setitem(
        repair.KNOWN,
        "0.8.0",
        (
            "fixture",
            hashlib.sha256(metadata).hexdigest(),
            repair.record_digest(library)[7:],
        ),
    )
    return importlib.metadata.Distribution.at(info)


def snapshot(dist):
    return {str(p): Path(dist.locate_file(p)).read_bytes() for p in dist.files}


def test_repairs_only_wheel_tag_and_record_and_is_idempotent(distribution):
    before = snapshot(distribution)
    assert repair.repair_distribution(distribution)
    after = snapshot(distribution)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {
        "nvidia_cusparselt_cu13-0.8.0.dist-info/WHEEL",
        "nvidia_cusparselt_cu13-0.8.0.dist-info/RECORD",
    }
    for item in distribution.files:
        data = Path(distribution.locate_file(item)).read_bytes()
        if item.hash:
            assert f"{item.hash.mode}={item.hash.value}" == repair.record_digest(data)
            assert item.size == len(data)
    assert repair.GOOD_TAG in distribution.read_text("WHEEL").encode()
    assert not repair.repair_distribution(distribution)
    assert snapshot(distribution) == after


@pytest.mark.parametrize("target", ["METADATA", "WHEEL", "RECORD", "library", "arch"])
def test_unknown_or_corrupt_inputs_fail_without_writes(distribution, target):
    if target in {"library", "arch"}:
        path = Path(distribution.locate_file(repair.LIBRARY))
        data = path.read_bytes()
        path.write_bytes(data + b"corrupt" if target == "library" else b"x" + data[1:])
    else:
        path = Path(
            distribution.locate_file(f"nvidia_cusparselt_cu13-0.8.0.dist-info/{target}")
        )
        data = path.read_bytes()
        path.write_bytes(
            data + b"changed"
            if target != "RECORD"
            else data.replace(b"sha256=", b"sha512=")
        )
    before = snapshot(distribution)
    with pytest.raises(ValueError):
        repair.repair_distribution(distribution)
    assert snapshot(distribution) == before


def test_unknown_version_is_not_repaired(distribution, monkeypatch):
    monkeypatch.setattr(repair, "KNOWN", {})
    before = snapshot(distribution)
    with pytest.raises(ValueError, match="unrecognized"):
        repair.repair_distribution(distribution)
    assert snapshot(distribution) == before


def test_record_write_failure_rolls_back_wheel(distribution, monkeypatch):
    original_write = repair.atomic_write
    before = snapshot(distribution)

    def fail_record(path, data):
        if path.name == "RECORD":
            raise OSError("disk full")
        original_write(path, data)

    monkeypatch.setattr(repair, "atomic_write", fail_record)
    with pytest.raises(OSError, match="disk full"):
        repair.repair_distribution(distribution)
    assert snapshot(distribution) == before


@pytest.mark.parametrize("system,machine", [("Darwin", "arm64"), ("Linux", "x86_64")])
def test_other_platforms_are_untouched(monkeypatch, system, machine):
    monkeypatch.setattr(repair.platform, "system", lambda: system)
    monkeypatch.setattr(repair.platform, "machine", lambda: machine)

    def forbidden(*args):
        pytest.fail("must not inspect distributions on an unaffected platform")

    monkeypatch.setattr(repair.importlib.metadata, "distribution", forbidden)
    repair.main()
