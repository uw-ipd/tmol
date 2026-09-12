"""Verify frozen chemistry/coordinates and every unweighted replay score term."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(reference, candidate):
    hashes = json.loads((reference / "sha256.json").read_text())
    reference_data = json.loads((reference / "scores.json").read_text())
    candidate_data = json.loads((candidate / "scores.json").read_text())
    if not candidate_data.get("replay"):
        raise ValueError("Candidate must be a fixed-input replay, not fresh generation")
    expected = {}
    for row in reference_data["runs"]:
        key = (row["backbone"], row["opt_h"])
        if row["repeat"] != 0 or key in expected:
            raise ValueError("Reference must contain one unique repeat of every case")
        expected[key] = row
    if not expected:
        raise ValueError("Empty reference cases")
    required = {"scores.json"}
    for backbone, opt_h in expected:
        required.add(f"{backbone}-0.tmol")
        required.add(f"{backbone}-0-{'opth' if opt_h else 'raw'}.npz")
    if set(hashes) != required:
        raise ValueError(
            "Reference checksum inventory does not cover exactly its inputs"
        )
    for name, wanted in hashes.items():
        if digest(reference / name) != wanted:
            raise ValueError(f"Reference checksum mismatch: {name}")

    repeats = {}
    for row in candidate_data["runs"]:
        cases = repeats.setdefault(row["repeat"], {})
        key = (row["backbone"], row["opt_h"])
        if key in cases:
            raise ValueError(f"Duplicate candidate case: {key}")
        cases[key] = row
    if not repeats or any(set(cases) != set(expected) for cases in repeats.values()):
        raise ValueError("Candidate must cover every reference case in every repeat")

    comparisons = []
    candidate_hashes = {"scores.json": digest(candidate / "scores.json")}
    for repeat, cases in sorted(repeats.items()):
        for (backbone, opt_h), row in cases.items():
            wanted = expected[backbone, opt_h]
            suffix = "opth" if opt_h else "raw"
            input_name = f"{backbone}-0-{suffix}.npz"
            output_name = f"{backbone}-{repeat}-{suffix}.npz"
            with np.load(reference / input_name, allow_pickle=False) as before:
                with np.load(candidate / output_name, allow_pickle=False) as after:
                    for field in ("atom_keys", "atom_types", "block_types", "coords"):
                        np.testing.assert_array_equal(
                            before[field],
                            after[field],
                            err_msg=f"{output_name}: {field}",
                        )
            candidate_hashes[output_name] = digest(candidate / output_name)
            if row["block_types"] != wanted["block_types"]:
                raise ValueError(f"Block identity mismatch for {output_name}")
            if set(row["scores"]) != set(wanted["scores"]):
                raise ValueError(f"Score term inventory mismatch for {output_name}")
            for term, target in wanted["scores"].items():
                value = row["scores"][term]
                difference = abs(value - target)
                passed = (
                    math.isfinite(value)
                    and math.isfinite(target)
                    and difference <= max(1e-2, 1e-4 * abs(target))
                )
                comparisons.append(
                    {
                        "backbone": backbone,
                        "opt_h": opt_h,
                        "repeat": repeat,
                        "term": term,
                        "reference": target,
                        "candidate": value,
                        "absolute_difference": difference,
                        "passed": passed,
                    }
                )
    return {
        "reference_commit": reference_data["git_head"],
        "candidate_commit": candidate_data["git_head"],
        "reference_environment": {
            k: v for k, v in reference_data.items() if k != "runs"
        },
        "candidate_environment": {
            k: v for k, v in candidate_data.items() if k != "runs"
        },
        "reference_sha256": hashes,
        "candidate_sha256": candidate_hashes,
        "cases": len(expected) * len(repeats),
        "terms": len(comparisons),
        "passed": all(row["passed"] for row in comparisons),
        "exact_terms": sum(row["absolute_difference"] == 0 for row in comparisons),
        "max_absolute_difference": max(
            row["absolute_difference"] for row in comparisons
        ),
        "atol": 1e-2,
        "rtol": 1e-4,
        "tolerance_rule": "max(atol, rtol * abs(reference))",
        "comparisons": comparisons,
        "limits": "Numerical regression for frozen parameter bundles and exact coordinates. Does not validate freshly generated parameters, reconstruct historical YAML provenance, establish independent force-field accuracy or replace the original failing generation/reference tests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "cases",
                    "terms",
                    "passed",
                    "exact_terms",
                    "max_absolute_difference",
                ]
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
