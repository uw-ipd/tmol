"""Warm canonical cartbonded timing with fixed inputs and source provenance."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import tmol

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.tests.data import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    root = Path(tmol.__file__).resolve().parent.parent
    source = root / "tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh"
    db = ParameterDatabase.get_default()
    # Load text rather than treating the filename as a PDB record.
    single = pose_stack_from_pdb(Path(data_path("pdb", "1ubq.pdb")).read_text(), device)
    # Compile only the old cartbonded kernel under a distinct C++/torch
    # namespace. Both kernels consume identical current pose/parameter tensors.
    # The benchmark isolates native scoring, not whole-application setup.
    from tmol._load_ext import load_ops
    from tmol.score.common._scoring_module import TermWholePoseScoringModule

    baseline_sources = args.baseline / "tmol/score/cartbonded/potentials"
    build = args.output.parent / "cartbonded-baseline-native"
    build.mkdir(parents=True, exist_ok=True)
    for source_file in baseline_sources.iterdir():
        if source_file.suffix not in (".hh", ".cpp", ".cu"):
            continue
        text = source_file.read_text()
        import re

        text = re.sub(r"<tmol/score/cartbonded/potentials/([^>]+)>", r'"\1"', text)
        text = text.replace(
            "namespace cartbonded {", "namespace cartbonded_review_baseline {"
        )
        text = text.replace("cartbonded::", "cartbonded_review_baseline::")
        text = text.replace(
            "TORCH_LIBRARY(tmol_cartbonded,",
            "TORCH_LIBRARY(tmol_cartbonded_review_baseline,",
        )
        target = build / source_file.name
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
    baseline_ops = load_ops(
        "review_cartbonded_baseline",
        str(build / "loader.py"),
        [
            "compiled.ops.cpp",
            "cartbonded_pose_score.cpu.cpp",
            "cartbonded_pose_score.cuda.cu",
        ],
        "tmol_cartbonded_review_baseline",
    )
    rows = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for batch in args.batches:
        pose = PoseStackBuilder.from_poses([single] * batch, device)
        term = CartBondedEnergyTerm(db, device)
        for bt in pose.packed_block_types.active_block_types:
            term.setup_block_type(bt)
        term.setup_packed_block_types(pose.packed_block_types)
        term.setup_poses(pose)
        attributes = term.get_score_term_attributes(pose)
        scorers = {
            "candidate": term.render_whole_pose_scoring_module(pose),
            "baseline": TermWholePoseScoringModule(
                "CartBondedBaseline",
                pose,
                attributes[:7] + attributes[10:],
                baseline_ops.cartbonded_pose_scores,
            ),
        }
        coords = pose.coords.clone().requires_grad_(True)

        def forward(label):
            with torch.no_grad():
                return scorers[label](coords)

        def backward(label):
            return torch.autograd.grad(scorers[label](coords).sum(), coords)[0]

        for _ in range(5):
            for label in scorers:
                forward(label)
                backward(label)
        timings = {label: {"forward": [], "forward_backward": []} for label in scorers}
        for sample in range(7):
            for name, func in (("forward", forward), ("forward_backward", backward)):
                for label in (
                    ("baseline", "candidate")
                    if sample % 2 == 0
                    else ("candidate", "baseline")
                ):
                    sync()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        func(label)
                    sync()
                    timings[label][name].append(
                        (time.perf_counter() - start) / args.iterations
                    )
        values = {label: forward(label).detach().cpu() for label in scorers}
        gradients = {label: backward(label).detach().cpu() for label in scorers}
        torch.testing.assert_close(
            values["candidate"], values["baseline"], rtol=1e-6, atol=1e-5
        )
        torch.testing.assert_close(
            gradients["candidate"], gradients["baseline"], rtol=1e-6, atol=1e-5
        )
        rows.append(
            dict(
                batch=batch,
                max_abs_score_difference=float(
                    (values["candidate"] - values["baseline"]).abs().max()
                ),
                max_abs_gradient_difference=float(
                    (gradients["candidate"] - gradients["baseline"]).abs().max()
                ),
                score_sha256={
                    label: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    for label, value in values.items()
                },
                gradient_sha256={
                    label: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    for label, value in gradients.items()
                },
                timings=timings,
            )
        )
    result = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        baseline_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.baseline, text=True
        ).strip(),
        baseline_native_sha256=hashlib.sha256(
            (baseline_sources / "cartbonded_pose_score.impl.hh").read_bytes()
        ).hexdigest(),
        native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        threads=torch.get_num_threads(),
        iterations=args.iterations,
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
