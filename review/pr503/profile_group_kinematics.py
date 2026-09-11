"""Paired scalar/batched group kinematics with identical full rotamer outputs."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.io import pose_stack_from_cif
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    baseline = "09258fbad"
    namespace = {}
    path = "tmol/pack/rotamer/_conjugated_chi_sampler.py"
    exec(
        compile(
            subprocess.check_output(["git", "show", baseline + ":" + path], text=True),
            path,
            "exec",
        ),
        namespace,
    )
    methods = {
        "before": namespace["ConjugatedChiSampler"].fill_dofs_for_samples,
        "after": ConjugatedChiSampler.fill_dofs_for_samples,
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def use_method(method, times):
        def measured(*args, **kwargs):
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                initial_memory = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            result = method(*args, **kwargs)
            sync()
            elapsed = time.perf_counter() - start
            peak_extra = (
                torch.cuda.max_memory_allocated(device) - initial_memory
                if device.type == "cuda"
                else None
            )
            times.append((elapsed, peak_extra))
            return result

        saved = ConjugatedChiSampler.fill_dofs_for_samples
        ConjugatedChiSampler.fill_dofs_for_samples = measured
        try:
            yield
        finally:
            ConjugatedChiSampler.fill_dofs_for_samples = saved

    rows = []
    for fixture in FIXTURES.values():
        pose, context = pose_stack_from_cif(
            data_path("covalent_fixtures", fixture + ".cif"),
            device,
            prepare_ligands=True,
            ligand_seed=20250828,
            no_optH=True,
            return_context=True,
        )
        task, _ = _task(pose, context.parameter_database, device)
        task = SetPackerTask.from_packer_task(task)
        observations = {
            name: {"fill_seconds": [], "build_seconds": [], "fill_peak_extra_bytes": []}
            for name in methods
        }
        reference = None
        max_error = 0.0
        # Alternate order, discard the first pair (JIT / annotation setup).
        for rep in range(8):
            for name in (("before", "after") if rep % 2 == 0 else ("after", "before")):
                fill_times = []
                sync()
                start = time.perf_counter()
                with use_method(methods[name], fill_times):
                    _, rotamers = build_rotamers(
                        pose, task, context.parameter_database.chemical
                    )
                sync()
                elapsed = time.perf_counter() - start
                if rep:
                    observations[name]["fill_seconds"].append(
                        sum(t[0] for t in fill_times)
                    )
                    observations[name]["build_seconds"].append(elapsed)
                    if device.type == "cuda":
                        observations[name]["fill_peak_extra_bytes"].append(
                            max(t[1] for t in fill_times)
                        )
                coordinates = rotamers.coords.cpu()
                if reference is None:
                    reference = (coordinates, rotamers.n_rots_for_block.cpu())
                else:
                    torch.testing.assert_close(
                        coordinates, reference[0], atol=2e-5, rtol=2e-5
                    )
                    torch.testing.assert_close(
                        rotamers.n_rots_for_block.cpu(), reference[1], rtol=0, atol=0
                    )
                    max_error = max(
                        max_error, float((coordinates - reference[0]).abs().max())
                    )
                del rotamers
        row = dict(
            fixture=fixture,
            rotamer_coordinate_shape=list(reference[0].shape),
            max_coordinate_difference=max_error,
        )
        for name, measurements in observations.items():
            measurements["fill_median"] = statistics.median(
                measurements["fill_seconds"]
            )
            measurements["build_median"] = statistics.median(
                measurements["build_seconds"]
            )
            row[name] = measurements
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=baseline,
                candidate=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                diff=subprocess.check_output(["git", "diff", "--", path], text=True),
                device=str(device),
                torch=torch.__version__,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
