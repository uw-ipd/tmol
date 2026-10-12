#!/usr/bin/env python3
"""Generate the browser playground from actual TMol scoring calls.

Run from a checkout with TMol installed. Only Phe45 chi1/chi2 rotate; all
other atoms stay fixed. The browser selects scored samples, never interpolates
energies. Hydrogens rotate with their bonded heavy atoms and enter every score.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

import numpy as np
import torch
import tmol
from tmol.io import pose_stack_from_pdb, pose_stack_to_pdb_string
from tmol.score import beta2016_score_function

ROOT = Path(__file__).resolve().parents[1]


def downstream(bonds, first, second):
    neighbors = {}
    for a, b, *_ in bonds:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    seen = {first}
    todo = [second]
    moved = []
    while todo:
        atom = todo.pop()
        if atom in seen:
            continue
        seen.add(atom)
        moved.append(atom)
        todo.extend(neighbors[atom])
    return moved


def rotate(coords, atom_indices, first, second, angle):
    origin = coords[first].clone()
    axis = coords[second] - origin
    axis /= torch.linalg.vector_norm(axis)
    offsets = coords[atom_indices] - origin
    theta = np.deg2rad(angle)
    coords[atom_indices] = (
        offsets * np.cos(theta)
        + torch.linalg.cross(axis.expand_as(offsets), offsets) * np.sin(theta)
        + (offsets @ axis)[:, None] * axis * (1 - np.cos(theta))
        + origin
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/_static/playground/phenylalanine.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare scores with the existing dataset without changing it",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    source = ROOT / "tmol/tests/data/pdb/1ubq.pdb"
    pose = pose_stack_from_pdb(str(source), torch.device("cpu"))
    residue_index = 44
    block = pose.packed_block_types.active_block_types[
        pose.block_type_ind[0, residue_index].item()
    ]
    assert block.name == "PHE"
    offset = pose.block_coord_offset[0, residue_index].item()
    indices = {a.name: offset + i for i, a in enumerate(block.atoms)}
    bonds = [(indices[a], indices[b]) for a, b, *_ in block.bonds]
    axes = [(indices["CA"], indices["CB"]), (indices["CB"], indices["CG"])]
    masks = [downstream(bonds, *axis) for axis in axes]
    moving = set(masks[0])
    fixed = [i for i in range(pose.coords.shape[1]) if i not in moving]

    pdb_lines = [
        line
        for line in pose_stack_to_pdb_string(pose).splitlines()
        if line.startswith("ATOM")
        and not line[12:16].strip().lstrip("0123456789").startswith("H")
    ]
    selected = [line for line in pdb_lines if int(line[22:26]) == 45]
    assert selected and all(line[17:20] == "PHE" for line in selected)
    display_indices = [indices[line[12:16].strip()] for line in selected]
    for line, index in zip(selected, display_indices):
        np.testing.assert_allclose(
            [float(line[k : k + 8]) for k in (30, 38, 46)],
            pose.coords[0, index].numpy(),
            atol=0.001,
        )

    sfxn = beta2016_score_function(pose.device)
    scorer = sfxn.render_whole_pose_scoring_module(pose)
    pair_scorer = sfxn.render_block_pair_scoring_module(pose)
    terms = [st.name for st in sfxn.all_score_types()]
    angles = list(range(-180, 180, 15))
    frames = []
    for a in angles:
        for b in angles:
            coords = pose.coords.clone()
            for axis, mask, angle in zip(axes, masks, (a, b)):
                rotate(coords[0], mask, *axis, angle)
            assert torch.equal(coords[0, fixed], pose.coords[0, fixed])
            # Rotation must preserve every covalent bond length in this residue.
            for first, second in bonds:
                torch.testing.assert_close(
                    torch.linalg.vector_norm(coords[0, first] - coords[0, second]),
                    torch.linalg.vector_norm(
                        pose.coords[0, first] - pose.coords[0, second]
                    ),
                    atol=2e-5,
                    rtol=2e-5,
                )
            with torch.no_grad():
                values = scorer(coords, sum_terms=False)[:, 0]
                total = scorer(coords)[0]
                pairs = pair_scorer(coords)[0]
                torch.testing.assert_close(pairs.sum(), total, atol=0.005, rtol=1e-4)
                partners = pairs[residue_index, :] + pairs[:, residue_index]
                partners[residue_index] = 0  # Exclude the residue's self term.
            assert torch.isfinite(values).all()
            torch.testing.assert_close(values.sum(), total, atol=0.005, rtol=1e-5)
            frames.append(
                {
                    "angles": [a, b],
                    "coordinates": coords[0, display_indices].numpy().round(5).tolist(),
                    "terms": values.numpy().round(6).tolist(),
                    "total": round(float(values.sum()), 6),
                    "partners": partners.numpy().round(6).tolist(),
                }
            )
        print(f"Scored {len(frames)}/{len(angles) ** 2} conformations", flush=True)

    best = min(range(len(frames)), key=lambda i: frames[i]["total"])
    # Start at a recoverable clash, rather than the most extreme overlap.
    target = frames[best]["total"] + 150
    start = min(range(len(frames)), key=lambda i: abs(frames[i]["total"] - target))
    for index in (start, best, angles.index(0) * len(angles) + angles.index(0)):
        coords = pose.coords.clone()
        for axis, mask, angle in zip(axes, masks, frames[index]["angles"]):
            rotate(coords[0], mask, *axis, angle)
        coords.requires_grad_(True)
        scorer(coords).sum().backward()
        assert torch.isfinite(coords.grad).all()

    source_root = Path(tmol.__file__).resolve().parents[1]
    if not (source_root / "pyproject.toml").is_file():
        # Docs CI installs a wheel built from this checkout.
        source_root = ROOT
    code_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    data = {
        "schema_version": 1,
        "title": "Rotate Phe45 in ubiquitin",
        "residue": 45,
        "angles": angles,
        "start": start,
        "best": best,
        "reference": angles.index(0) * len(angles) + angles.index(0),
        "pdb": "\n".join(pdb_lines) + "\nEND\n",
        "residue_pdb": selected,
        "terms": terms,
        "residue_labels": [
            f"{pose.packed_block_types.active_block_types[i].name.split(':')[0]}{j + 1}"
            for j, i in enumerate(pose.block_type_ind[0].tolist())
        ],
        "weights": sfxn.weights_tensor().detach().numpy().tolist(),
        "frames": frames,
        "provenance": {
            "tmol_version": tomllib.loads((source_root / "pyproject.toml").read_text())[
                "project"
            ]["version"],
            "source_commit": code_commit,
            "torch_version": torch.__version__,
            "device": "cpu",
            "score_function": "beta2016_score_function",
            "input": "tmol/tests/data/pdb/1ubq.pdb",
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "step_degrees": 15,
            "seed": 0,
            "hydrogens_scored": True,
            "other_atoms_fixed": True,
            "interpolated": False,
        },
    }
    if args.verify:
        expected = json.loads(args.output.read_text())
        for key in (
            "schema_version",
            "residue",
            "angles",
            "terms",
            "start",
            "best",
            "reference",
            "residue_labels",
        ):
            assert data[key] == expected[key], key
        assert (
            data["provenance"]["input_sha256"] == expected["provenance"]["input_sha256"]
        )
        for key in ("pdb", "residue_pdb"):
            actual_lines = data[key].splitlines() if key == "pdb" else data[key]
            saved_lines = expected[key].splitlines() if key == "pdb" else expected[key]
            assert len(actual_lines) == len(saved_lines), key
            for actual, saved in zip(actual_lines, saved_lines):
                assert actual[:30] == saved[:30] and actual[54:] == saved[54:], key
                if actual.startswith("ATOM"):
                    np.testing.assert_allclose(
                        [float(actual[k : k + 8]) for k in (30, 38, 46)],
                        [float(saved[k : k + 8]) for k in (30, 38, 46)],
                        atol=0.002,
                        rtol=0,
                    )
        np.testing.assert_allclose(data["weights"], expected["weights"])
        assert len(data["frames"]) == len(expected["frames"])
        for actual, saved in zip(data["frames"], expected["frames"]):
            assert actual["angles"] == saved["angles"]
            np.testing.assert_allclose(
                actual["coordinates"], saved["coordinates"], atol=0.002, rtol=0
            )
            np.testing.assert_allclose(
                actual["terms"], saved["terms"], atol=0.002, rtol=5e-5
            )
            np.testing.assert_allclose(
                actual["total"], saved["total"], atol=0.002, rtol=5e-5
            )
            np.testing.assert_allclose(
                actual["partners"], saved["partners"], atol=0.002, rtol=5e-5
            )
        print(f"Verified all {len(frames)} playground samples against TMol")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(
        f"Saved {args.output}: {len(frames)} scored conformations; best {frames[best]['total']:.3f}"
    )


if __name__ == "__main__":
    main()
