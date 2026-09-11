"""Paired linkage-axis correction: preparation/build latency and bond geometry."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from tmol.io import pose_stack_from_cif
from tmol.ligand import _conjugation_patches as patches
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
from tmol.pose._conjugated_groups import find_conjugated_groups
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    baseline = "fa41a727e"

    def previous(path):
        namespace = {}
        exec(
            compile(
                subprocess.check_output(
                    ["git", "show", baseline + ":" + path], text=True
                ),
                path,
                "exec",
            ),
            namespace,
        )
        return namespace

    old_patches = previous("tmol/ligand/_conjugation_patches.py")
    old_groups = previous("tmol/pack/rotamer/_conjugated_groups.py")
    old_sampler = previous("tmol/pack/rotamer/_conjugated_chi_sampler.py")
    old_sampler["group_sampled_chi"] = old_groups["group_sampled_chi"]
    old = old_sampler["ConjugatedChiSampler"]

    @contextmanager
    def version(before):
        saved = (
            patches.conjugation_patch,
            ConjugatedChiSampler.group_conformers,
            ConjugatedChiSampler.group_coords,
        )
        if before:
            patches.conjugation_patch = old_patches["conjugation_patch"]
            ConjugatedChiSampler.group_conformers = old.group_conformers
            ConjugatedChiSampler.group_coords = old.group_coords
        try:
            yield
        finally:
            (
                patches.conjugation_patch,
                ConjugatedChiSampler.group_conformers,
                ConjugatedChiSampler.group_coords,
            ) = saved

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def bond_error(pose, rotamers):
        worst = 0.0
        for group in find_conjugated_groups(pose):
            for block in group.blocks:
                bt = pose.packed_block_types.active_block_types[
                    int(pose.block_type_ind[group.pose, block])
                ]
                first = int(rotamers.rot_offset_for_block[group.pose, block])
                count = int(rotamers.n_rots_for_block[group.pose, block])
                offset = rotamers.coord_offset_for_rot[first : first + count].long()
                xyz = rotamers.coords[
                    offset[:, None] + torch.arange(bt.n_atoms, device=device)
                ]
                start = int(pose.block_coord_offset[group.pose, block])
                source = pose.coords[group.pose, start : start + bt.n_atoms]
                bonds = torch.tensor(bt.bond_indices, dtype=torch.int64, device=device)
                expected = (source[bonds[:, 0]] - source[bonds[:, 1]]).norm(dim=-1)
                actual = (xyz[:, bonds[:, 0]] - xyz[:, bonds[:, 1]]).norm(dim=-1)
                worst = max(worst, float((actual - expected).abs().max()))
        return worst

    rows = []
    for fixture in FIXTURES.values():
        row = {"fixture": fixture}
        for variant in ("before", "after"):
            row[variant] = {
                "prepare_seconds": [],
                "build_seconds": [],
                "max_bond_error": 0.0,
            }
        for repetition in range(8):
            for variant in (
                ("before", "after") if repetition % 2 == 0 else ("after", "before")
            ):
                with version(variant == "before"):
                    sync()
                    start = time.perf_counter()
                    pose, context = pose_stack_from_cif(
                        data_path("covalent_fixtures", fixture + ".cif"),
                        device,
                        prepare_ligands=True,
                        ligand_seed=20250828,
                        no_optH=True,
                        return_context=True,
                    )
                    sync()
                    prepared = time.perf_counter() - start
                    task, _ = _task(pose, context.parameter_database, device)
                    task = SetPackerTask.from_packer_task(task)
                    sync()
                    start = time.perf_counter()
                    pose, rotamers = build_rotamers(
                        pose, task, context.parameter_database.chemical
                    )
                    sync()
                    built = time.perf_counter() - start
                measurement = row[variant]
                if repetition:
                    measurement["prepare_seconds"].append(prepared)
                    measurement["build_seconds"].append(built)
                measurement["rotamers"] = rotamers.n_rotamers_total
                measurement["rotamer_coordinate_bytes"] = (
                    rotamers.coords.numel() * rotamers.coords.element_size()
                )
                measurement["max_bond_error"] = max(
                    measurement["max_bond_error"], bond_error(pose, rotamers)
                )
                if variant == "after":
                    assert measurement["max_bond_error"] < 1e-4
        for variant in ("before", "after"):
            for stage in ("prepare", "build"):
                row[variant][stage + "_median"] = statistics.median(
                    row[variant][stage + "_seconds"]
                )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=baseline,
                device=str(device),
                torch=torch.__version__,
                comparison="Prior patch generation, group enumeration and group-coordinate methods; shared native kernels and current input reader",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
