"""Scoring workload benchmark for the spline index change (run in each checkout)."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
import torch
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.dunbrack import DunbrackEnergyTerm
from tmol.score.backbone_torsion import BackboneTorsionEnergyTerm
from tmol.tests.data import pdb
from tmol.tests.score.dunbrack.test_parameter_identity import setup, render

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()
device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
database = ParameterDatabase.get_default()
input_pdb = pdb.data["1ubq"]
base_pose = pose_stack_from_pdb(input_pdb, device)
results = []
values = []
for count in (1, 64):
    pose = PoseStackBuilder.from_poses([base_pose] * count, device)
    coords = pose.coords.detach().clone().requires_grad_(True)
    for name, kind in [
        ("dunbrack", DunbrackEnergyTerm),
        ("backbone", BackboneTorsionEnergyTerm),
    ]:
        term = kind(database, device)
        setup(term, pose)
        for pairs in (False, True):
            module = render(term, pose, pairs)
            for gradient in (False, True):

                def call():
                    if gradient:
                        scores = module(coords)
                        deriv = torch.autograd.grad(scores.sum(), coords)[0]
                        return scores.detach(), deriv
                    with torch.no_grad():
                        return (module(coords),)

                for _ in range(5):
                    call()
                expected = call()
                assert all(torch.isfinite(t).all() for t in expected)
                values.append(tuple(t.cpu() for t in expected))
                del expected
                samples = []
                repetitions = 20 if device.type == "cuda" else 5
                for _ in range(7):
                    if device.type == "cuda":
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(repetitions):
                            call()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / repetitions)
                    else:
                        start = time.perf_counter()
                        for _ in range(repetitions):
                            call()
                        samples.append(
                            (time.perf_counter() - start) * 1000 / repetitions
                        )
                peak = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    initial = torch.cuda.memory_allocated()
                    output = call()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - initial
                    del output
                results.append(
                    {
                        "term": name,
                        "poses": count,
                        "residues_per_pose": pose.max_n_blocks,
                        "block_pairs": pairs,
                        "gradient": gradient,
                        "milliseconds": samples,
                        "median_milliseconds": statistics.median(samples),
                        "peak_allocated_bytes_above_inputs": peak,
                    }
                )
                print(json.dumps(results[-1]), flush=True)
source = Path("tmol/numeric/bspline_compiled/bspline.hh")
value_path = args.output.with_suffix(".pt")
torch.save(values, value_path)
result = {
    "checkout_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "device": str(device),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
    "input_pdb_sha256": hashlib.sha256(input_pdb.encode()).hexdigest(),
    "values_path": str(value_path),
    "results": results,
    "limits": "Default 1ubq chemistry replicated across 1 or 64 poses; Dunbrack and backbone terms separately, whole-pose/block-pair outputs, forward and forward-plus-gradient. Construction/compilation excluded. Seven warm rounds per process; a separate driver alternates checkouts. CUDA-event times include dispatch/launch sequence. Managed allocator peaks exclude parameters/inputs already live, compiler resources and kernel local memory.",
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
