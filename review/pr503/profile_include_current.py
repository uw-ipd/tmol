"""Compare include-current copy plans, DOF fills and real rotamer construction."""

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
from tmol.pose import PoseStackBuilder

BASELINE = "455ca2805"
SOURCE = "tmol/pack/rotamer/_include_current_sampler.py"
NAME = "create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler"


class CurrentPalette(PackerPalette):
    def default_conformer_samplers(self):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    module = importlib.import_module("tmol.pack.rotamer._include_current_sampler")
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    namespace = dict(vars(module))
    exec(compile(source, "baseline_include_current", "exec"), namespace)
    planners = {"before": namespace[NAME], "after": getattr(module, NAME)}
    fillers = {
        "before": namespace["IncludeCurrentSampler"].fill_dofs_for_samples,
        "after": IncludeCurrentSampler.fill_dofs_for_samples,
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(functions, calls):
        samples = {side: [] for side in functions}
        for trial in range(7):
            for side in list(functions)[:: -1 if trial % 2 else 1]:
                sync()
                start = time.perf_counter()
                for _ in range(calls):
                    output = functions[side]()
                    del output
                sync()
                samples[side].append((time.perf_counter() - start) * 1000 / calls)
        peaks = {}
        if device.type == "cuda":
            for side, function in functions.items():
                gc.collect()
                sync()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                output = function()  # noqa: F841 -- retain through peak measurement
                sync()
                peaks[side] = torch.cuda.max_memory_allocated() - baseline
                del output
        medians = {side: statistics.median(v) for side, v in samples.items()}
        return {
            "milliseconds": samples,
            "median_milliseconds": medians,
            "speedup": medians["before"] / medians["after"],
            "extra_cuda_peak_bytes": peaks,
            "calls_per_round": calls,
        }

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    synthetic = []
    for width in (40, 1024):
        for n in (12, 12000):
            pbt = SimpleNamespace(
                n_atoms=tensor([15, 21, 32, width]), max_n_atoms=width
            )
            poses = SimpleNamespace(
                block_type_ind=tensor([[0, 1, -1], [2, 0, 1]], torch.int32),
                packed_block_types=pbt,
                device=device,
            )
            mapping = tensor([0, 1, 3, 4, 5])
            task = SimpleNamespace(global_block_ind_for_considered_block_types=mapping)
            gbt = torch.arange(n, device=device) % 5
            types = poses.block_type_ind.flatten()[mapping[gbt]].long()
            selected = torch.arange(n - 1, -1, -1, device=device)
            sizes = pbt.n_atoms[types]
            starts = torch.cumsum(sizes, 0) - sizes
            inputs = (
                poses,
                task,
                gbt,
                types,
                selected,
                torch.bincount(gbt, minlength=5).int(),
                gbt[selected].int(),
                starts,
            )
            before, after = [function(*inputs) for function in planners.values()]
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            pairs = before[0].numel()
            del before, after, a, b
            functions = {
                side: (lambda function=function: function(*inputs))
                for side, function in planners.items()
            }
            synthetic.append(
                {
                    "max_atoms": width,
                    "conformers": n,
                    "copy_pairs": pairs,
                    "indices_exact": True,
                    **measure(functions, 10),
                }
            )

    pdb = Path("tmol/tests/data/pdb/1ubq.pdb").read_text()
    pose = pose_stack_from_pdb(pdb, device)
    real = []
    for n_poses in (1, 16):
        poses = PoseStackBuilder.from_poses([pose] * n_poses, device)
        task = PackerTask(poses, CurrentPalette())
        task.restrict_to_repacking()
        sampler = IncludeCurrentSampler()
        task.add_conformer_sampler(sampler)
        finalized = SetPackerTask.from_packer_task(task)
        captured = []

        def capture(self, *values):
            captured.append((*values[:-1], values[-1].clone()))
            fillers["after"](self, *values)

        def build(function):
            IncludeCurrentSampler.fill_dofs_for_samples = function
            try:
                return build_rotamers(
                    poses, finalized, poses.packed_block_types.chem_db
                )[1]
            finally:
                IncludeCurrentSampler.fill_dofs_for_samples = fillers["after"]

        warm = build(capture)
        del warm
        before, after = [build(function) for function in fillers.values()]
        torch.testing.assert_close(before.coords, after.coords, rtol=0, atol=0)
        n_rotamers = len(after.block_ind_for_rot)
        n_atoms = after.coords.shape[0]
        del before, after
        values = captured[0]
        expected = values[-1].clone()
        actual = values[-1].clone()
        fillers["before"](sampler, *values[:-1], expected)
        fillers["after"](sampler, *values[:-1], actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del expected, actual
        real.append(
            {
                "poses": n_poses,
                "rotamers": n_rotamers,
                "atoms": n_atoms,
                "dofs_and_coordinates_exact": True,
                "fill": measure(
                    {
                        side: (lambda function=function: function(sampler, *values))
                        for side, function in fillers.items()
                    },
                    10,
                ),
                "build": measure(
                    {
                        side: (lambda function=function: build(function))
                        for side, function in fillers.items()
                    },
                    3,
                ),
            }
        )
    report = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", BASELINE], text=True
        ).strip(),
        "source_sha256": {
            "before": hashlib.sha256(source.encode()).hexdigest(),
            "after": hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        },
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "synthetic": synthetic,
        "real": real,
        "limits": "Seven alternating warm rounds; explicit synchronization around timings. Synthetic planner cases use15/21/32atomtypes and unrelated PBT maximum40/1024. Real cases use1ubq, IncludeCurrent only,1/16poses. Construction excludes parsing and initial annotation/compilation, includes subsequent build_rotamers work. This is not a full sampling/packing/scoring benchmark. Extra CUDA allocated peaks exclude live inputs and retained caches, not total native/process memory.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
