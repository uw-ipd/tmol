"""Verify archived input/evidence hashes and optionally compare a fresh run."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path)
args = parser.parse_args()
manifest = json.loads((ROOT / "metadata/checksums.json").read_text())
for relative, expected in manifest.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert actual == expected, f"Checksum mismatch: {relative}"
print(f"PASS: {len(manifest)} input and reference files match SHA-256 checksums")


def close(actual, expected, context):
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=.01), (context, actual, expected)


if args.results:
    count = 0
    for file in sorted((args.results / "controls").glob("*.json")):
        reference = ROOT / "reference/controls" / (file.name + ".gz")
        if not reference.exists():
            continue  # Extra ligand inputs belong to the optional topology audit.
        new = json.loads(file.read_text())
        old = json.loads(gzip.decompress(reference.read_bytes()))
        assert new["status"] == "ok"
        for control in old["controls"]:
            for term, expected in old["controls"][control]["weighted_terms"].items():
                close(new["controls"][control]["weighted_terms"][term], expected, (file.stem, control, term))
        native = json.loads((args.results / "dunbrack" / file.name).read_text())
        frozen = json.loads(gzip.decompress((ROOT / "reference/dunbrack" / (file.name + ".gz")).read_bytes()))
        for control in frozen["controls"]:
            for i, expected in enumerate(frozen["controls"][control]["rotdev_per_residue"]):
                close(native["controls"][control]["rotdev_per_residue"][i], expected, (file.stem, control, i+1))
        pyro = json.loads((args.results / f"exports/pyrosetta-{file.stem}.json").read_text())
        frozen_pyro = json.loads(gzip.decompress((ROOT / f"reference/exports/pyrosetta-{file.stem}.json.gz").read_bytes()))
        for term, expected in frozen_pyro["unweighted_terms"].items():
            close(pyro["unweighted_terms"][term], expected, (file.stem, "PyRosetta", term))
        count += 1
    assert count, "No completed diagnostic datasets found"
    print(f"PASS: {count} freshly rescored datasets reproduce the archived native scores and controls")
