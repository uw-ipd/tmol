"""Run the same group geometry/packing assertions through an explicit input route."""

import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from tmol.io import pose_stack_from_biotite
from tmol.tests.data import data_path
from tmol.tests.pack import test_conjugated_group_packing as packing
from tmol.tests.pack.rotamer import test_group_chemical_geometry as geometry
from review.pr503.profile_workloads import read_structure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=["tmol", "atomworks"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    def pose_from_route(stem, device):
        array = read_structure(
            data_path("covalent_fixtures", stem + ".cif"), args.reader
        )
        return pose_stack_from_biotite(
            array,
            device,
            prepare_ligands=True,
            ligand_seed=20250828,
            no_optH=True,
            return_context=True,
        )

    saved = geometry._pose, packing._pose
    geometry._pose = packing._pose = pose_from_route
    cases = [geometry.test_group_conformers_preserve_all_bonds_angles_and_chirality]
    if args.pack:
        cases += [
            packing.test_the_bond_survives_packing,
            packing.test_the_packers_energy_matches_what_the_pose_scores,
        ]
    rows = []
    try:
        for fixture in sorted(packing.FIXTURES):
            for case in cases:
                row = dict(fixture=fixture, case=case.__name__)
                start = time.perf_counter()
                try:
                    case(fixture, device)
                    row["status"] = "pass"
                except Exception:
                    row["status"] = "fail"
                    row["error"] = traceback.format_exc()
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        geometry._pose, packing._pose = saved
    args.output.write_text(
        json.dumps(
            dict(
                reader=args.reader,
                device=str(device),
                torch=torch.__version__,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    if any(r["status"] != "pass" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
