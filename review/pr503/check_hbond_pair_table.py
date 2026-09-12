"""Native hydrogen-bond behavior when a declared backbone pair is omitted."""

import argparse
import json
from pathlib import Path
import subprocess

import attr
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_cif
import tmol.score.hbond._hbond_dependent_term as dependent_module
import tmol.score.hbond._hbond_energy_term as energy_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revision:
        source = subprocess.check_output(
            ["git", "show", f"{args.revision}:tmol/score/hbond/_params.py"], text=True
        )
        namespace = {
            "__name__": "tmol.score.hbond._params",
            "__package__": "tmol.score.hbond",
        }
        exec(compile(source, "previous_hbond_params.py", "exec"), namespace)
        dependent_module.HBondParamResolver = namespace["HBondParamResolver"]
        energy_module.CompactedHBondDatabase = namespace["CompactedHBondDatabase"]
    database = ParameterDatabase.get_default()
    hb = database.scoring.hbond
    donor = next(g.donor_type for g in hb.donor_atom_types if g.d == "Nbb")
    acceptor = next(g.acceptor_type for g in hb.acceptor_atom_types if g.a == "OCbb")
    missing = attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            hbond=attr.evolve(
                hb,
                pair_parameters=tuple(
                    p
                    for p in hb.pair_parameters
                    if (p.donor_type, p.acceptor_type) != (donor, acceptor)
                ),
            ),
        ),
    )
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    output = dict(
        revision=args.revision or "working-tree",
        device=str(device),
        missing_pair=[donor, acceptor],
        cases={},
    )
    for label, db in [("complete", database), ("missing_backbone_pair", missing)]:
        pose = pose_stack_from_cif(Path("tmol/tests/data/cif/1UBQ.cif"), device)
        try:
            term = energy_module.HBondEnergyTerm(db, device)
            term.setup_packed_block_types(pose.packed_block_types)
        except ValueError as error:
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
            energy_finite=bool(torch.isfinite(score).all()),
            gradient_finite=bool(torch.isfinite(coords.grad).all()),
            gradient=coords.grad.cpu().tolist(),
        )

    # Diagnostic NaNs are represented as strings, keeping the artifact valid JSON.
    def clean(value):
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            return str(value)
        if isinstance(value, list):
            return [clean(x) for x in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    output = clean(output)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                label: {k: v for k, v in row.items() if k != "gradient"}
                for label, row in output["cases"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
