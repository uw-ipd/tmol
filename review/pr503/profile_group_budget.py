"""Compare group enumeration with the pre-budget implementation, same anchor rows."""

import argparse
import json
import subprocess
import time
import statistics
from pathlib import Path
import torch
from tmol.io import pose_stack_from_cif
from tmol.pack import PackerTask, PackerPalette, SetPackerTask
from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
from tmol.pack.rotamer._conjugated_groups import (
    add_conjugated_group_sampler,
    find_conjugated_groups,
)
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--output", required=True)
args = p.parse_args()
device = torch.device("cuda:0" if args.device == "cuda" else "cpu")


def previous(path):
    ns = {}
    exec(
        compile(
            subprocess.check_output(["git", "show", "fccd9ad5c:" + path], text=True),
            path,
            "exec",
        ),
        ns,
    )
    return ns


old = previous("tmol/pack/rotamer/_conjugated_chi_sampler.py")
old_groups = previous("tmol/pack/rotamer/_conjugated_groups.py")
old_groups["_budgeted_chi_samples"] = previous("tmol/pack/rotamer/_chi_budget.py")[
    "_budgeted_chi_samples"
]
old["group_sampled_chi"] = old_groups["group_sampled_chi"]
methods = {
    "before": old["ConjugatedChiSampler"].group_conformers,
    "after": ConjugatedChiSampler.group_conformers,
}
rows = []
for fixture in ("lys_biotin_1bdo", "oglycan_sia_1g1s", "nglycan_tree_1ax2"):
    pose, ctx = pose_stack_from_cif(
        Path("tmol/tests/data/covalent_fixtures") / f"{fixture}.cif",
        device,
        prepare_ligands=True,
        ligand_seed=20250828,
        no_optH=True,
        return_context=True,
    )
    task = PackerTask(pose, PackerPalette())
    lib = create_dunbrack_sampler_from_database(ctx.parameter_database, device)
    task.add_conformer_sampler(lib)
    sampler = add_conjugated_group_sampler(task, pose)
    task.restrict_to_repacking()
    task = SetPackerTask.from_packer_task(task)
    for bt in pose.packed_block_types.active_block_types:
        lib.annotate_residue_type(bt)
    lib.annotate_packed_block_types(pose.packed_block_types)
    anchor = sampler.anchor_library_chi(pose, task, find_conjugated_groups(pose))
    row = {
        "fixture": fixture,
        "library_counts": {str(k): len(v[1]) for k, v in anchor.items()},
    }
    for name, method in methods.items():
        durations = []
        for rep in range(8):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            groups = method(sampler, pose, anchor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if rep:
                durations.append(time.perf_counter() - start)
        row[name] = {
            "seconds": durations,
            "median": statistics.median(durations),
            "conformers": [len(a) for g, c, a in groups],
            "member_rotamers": [len(g.blocks) * len(a) for g, c, a in groups],
            "chi_array_bytes": sum(a.nbytes for g, c, a in groups),
        }
        if name == "after":
            assert all(n <= 1000 for n in row[name]["member_rotamers"])
    rows.append(row)
    print(json.dumps(row), flush=True)
Path(args.output).write_text(
    json.dumps(
        {
            "baseline": "fccd9ad5c",
            "device": str(device),
            "torch": torch.__version__,
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
