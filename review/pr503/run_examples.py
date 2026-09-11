"""Run every new chemistry CIF through preparation, scoring and backward.

Run from the tmol checkout with TMOL_USE_JIT=1 if extensions are not built:
    python review/pr503/run_examples.py --device cuda --output examples.json
"""

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from tmol.io import pose_stack_from_cif
from tmol.score import beta2016_score_function


def run(device, output):
    root = Path(__file__).resolve().parents[2]
    data = root / "tmol/tests/data"
    paths = sorted((data / "ncaa_fixtures").glob("*.cif"))
    paths += sorted((data / "covalent_fixtures").glob("*.cif"))
    paths.append(data / "cif/cyclic_peptide_1jbl.cif")
    results = []
    for path in paths:
        begin = time.perf_counter()
        row = dict(fixture=str(path.relative_to(data)), device=device)
        try:
            pose, context = pose_stack_from_cif(
                path,
                torch.device(device),
                prepare_ligands=True,
                ligand_seed=20260909,
                no_optH=True,
                return_context=True,
            )
            score = beta2016_score_function(
                torch.device(device), param_db=context.parameter_database
            )
            module = score.render_whole_pose_scoring_module(pose)
            coords = pose.coords.detach().clone().requires_grad_(True)
            energy = module(coords)
            energy.sum().backward()
            assert torch.isfinite(energy).all(), "non-finite energy"
            assert torch.isfinite(coords.grad).all(), "non-finite gradient"
            row.update(
                status="passed",
                atoms=pose.coords.shape[1],
                blocks=pose.max_n_blocks,
                score=float(energy.sum().detach()),
                max_abs_gradient=float(coords.grad.abs().max()),
            )
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        row["seconds"] = time.perf_counter() - begin
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(output).write_text(json.dumps(results, indent=2) + "\n")
    return all(row["status"] == "passed" for row in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default="examples.json")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.device, args.output) else 1)
