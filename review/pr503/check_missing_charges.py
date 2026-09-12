"""Measure the effect of a missing residue charge table on native electrostatics."""

import argparse
import json
from pathlib import Path
import subprocess

import attr
import torch

from tmol.database import ParameterDatabase
from tmol.io import extended_pose_stack_from_sequences
import tmol.score.elec._elec_energy_term as elec_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revision:
        source = subprocess.check_output(
            ["git", "show", f"{args.revision}:tmol/score/elec/_params.py"], text=True
        )
        namespace = {
            "__name__": "tmol.score.elec._params",
            "__package__": "tmol.score.elec",
        }
        exec(compile(source, "previous_elec_params.py", "exec"), namespace)
        elec_module.ElecParamResolver = namespace["ElecParamResolver"]
    database = ParameterDatabase.get_default()
    missing = attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            elec=attr.evolve(
                database.scoring.elec,
                atom_charge_parameters=tuple(
                    r
                    for r in database.scoring.elec.atom_charge_parameters
                    if r.res.partition(":")[0] != "ALA"
                ),
            ),
        ),
    )
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    output = dict(
        revision=args.revision or "working-tree", device=str(device), cases={}
    )
    for label, db in [("complete", database), ("missing_ALA", missing)]:
        pose = extended_pose_stack_from_sequences(["AA"], device=device)
        term = elec_module.ElecEnergyTerm(db, device)
        try:
            term.setup_packed_block_types(pose.packed_block_types)
        except KeyError as error:
            output["cases"][label] = dict(outcome="rejected_at_setup", error=str(error))
            continue
        term.setup_poses(pose)
        scorer = term.render_whole_pose_scoring_module(pose)
        coords = pose.coords.detach().requires_grad_(True)
        score = scorer(coords)
        score.sum().backward()
        output["cases"][label] = dict(
            outcome="scored",
            energy=score.detach().cpu().tolist(),
            gradient=coords.grad.cpu().tolist(),
            energy_finite=bool(torch.isfinite(score).all()),
            gradient_finite=bool(torch.isfinite(coords.grad).all()),
        )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                label: {k: v for k, v in result.items() if k != "gradient"}
                for label, result in output["cases"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
