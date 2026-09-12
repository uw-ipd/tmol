"""Reproduce missing-type scoring with optional pinned Python import overlays."""

import argparse
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

MODULES = {
    "tmol.score._chemical_database",
    "tmol.score._atom_type_dependent_term",
    "tmol.score.hbond._hbond_dependent_term",
    "tmol.score.ljlk._params",
    "tmol.score.ljlk._ljlk_energy_term",
    "tmol.score.lk_ball._lk_ball_energy_term",
}
ROOT = Path(__file__).resolve().parents[2]


class PreviousImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, revision):
        self.revision = revision

    def find_spec(self, fullname, path, target=None):
        if fullname in MODULES:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__.replace(".", "/") + ".py"
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{self.revision}:{relative}"], text=True
        )
        module.__file__ = str(ROOT / relative)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revision:
        sys.meta_path.insert(0, PreviousImports(args.revision))
    import attr
    import torch
    from tmol.database import ParameterDatabase
    from tmol.io import extended_pose_stack_from_sequences
    from tmol.score.ljlk import LJLKEnergyTerm
    from tmol.score.lk_ball import LKBallEnergyTerm

    database = ParameterDatabase.get_default()
    missing = attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            ljlk=attr.evolve(
                database.scoring.ljlk,
                atom_type_parameters=tuple(
                    row
                    for row in database.scoring.ljlk.atom_type_parameters
                    if row.name != "CH3"
                ),
            ),
        ),
    )
    result = {"revision": args.revision or "working-tree", "cases": {}}
    for cls in (LJLKEnergyTerm, LKBallEnergyTerm):
        for label, db in [("complete", database), ("missing_CH3", missing)]:
            pose = extended_pose_stack_from_sequences(
                ["AA"], device=torch.device("cpu")
            )
            term = cls(db, torch.device("cpu"))
            key = f"{cls.__name__}:{label}"
            try:
                term.setup_packed_block_types(pose.packed_block_types)
            except ValueError as error:
                result["cases"][key] = dict(
                    outcome="rejected_at_setup", error=str(error)
                )
                continue
            term.setup_poses(pose)
            scorer = term.render_whole_pose_scoring_module(pose)
            coords = pose.coords.detach().requires_grad_(True)
            energy = scorer(coords)
            energy.sum().backward()
            result["cases"][key] = dict(
                outcome="scored",
                energy=[
                    float(v) if torch.isfinite(v) else str(float(v))
                    for v in energy.detach().flatten()
                ],
                energy_finite=bool(torch.isfinite(energy).all()),
                gradient_finite=bool(torch.isfinite(coords.grad).all()),
            )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
