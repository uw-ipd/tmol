"""Save generated chemistry and pose coordinates to localize reference drift."""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import cattr
import numpy as np
import torch
from rdkit import rdBase
from openbabel import openbabel

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand import prepare_ligands
from tmol.score import beta2016_score_function
from tmol.tests.score.test_noncanonical_scoring import FIXTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--classes", nargs="+", default=list(FIXTURES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--block-pairs", action="store_true")
    parser.add_argument(
        "--replay",
        type=Path,
        help="Use this run's saved .tmol parameters and coordinates",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    import tmol

    root = Path(tmol.__file__).resolve().parent
    metadata = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "openbabel": openbabel.OBReleaseVersion(),
        "device": str(device),
        "seed": 20250828,
        "replay": str(args.replay) if args.replay else None,
        "runs": [],
    }
    base = ParameterDatabase.get_default()
    original_names = {r.name for r in base.chemical.residues}
    for repeat in range(args.repeats):
        for backbone in args.classes:
            stem = FIXTURES[backbone]
            array = atom_array_from_cif(
                root / "tests/data/ncaa_fixtures" / (stem + ".cif")
            )
            prefix = f"{backbone}-{repeat}"
            if args.replay:
                db, _ = prepare_ligands(
                    array,
                    param_db=base,
                    params_files=[str(args.replay / f"{backbone}-0.tmol")],
                )
            else:
                db, _ = prepare_ligands(
                    array,
                    param_db=base,
                    seed=metadata["seed"],
                    params_output=str(args.output / f"{prefix}.tmol"),
                )
            added = [r for r in db.chemical.residues if r.name not in original_names]
            chemical = {r.name: cattr.unstructure(r) for r in added}
            cart = {
                name: cattr.unstructure(value)
                for name, value in db.scoring.cartbonded.residue_params.items()
                if name not in base.scoring.cartbonded.residue_params
            }
            (args.output / f"{prefix}-parameters.json").write_text(
                json.dumps(dict(chemical=chemical, cartbonded=cart), indent=2) + "\n"
            )
            for opt_h in (False, True):
                pose = pose_stack_from_biotite(
                    array, device, param_db=db, no_optH=bool(args.replay) or not opt_h
                )
                suffix = "opth" if opt_h else "raw"
                if args.replay:
                    saved = np.load(args.replay / f"{backbone}-0-{suffix}.npz")
                    replay_keys = [
                        f"{block}:{bt.name}:{atom.name}"
                        for block, type_index in enumerate(
                            pose.block_type_ind[0].tolist()
                        )
                        for bt in [
                            pose.packed_block_types.active_block_types[type_index]
                        ]
                        for atom in bt.atoms
                    ]
                    assert replay_keys == saved["atom_keys"].tolist()
                    pose.coords.copy_(torch.tensor(saved["coords"], device=device))
                score = beta2016_score_function(device, param_db=db)
                values = score.render_whole_pose_scoring_module(pose)(
                    pose.coords, sum_terms=False, apply_weights=False
                )
                scores = {
                    st.name: float(values[i].sum().detach())
                    for i, st in enumerate(score.all_score_types())
                }
                if args.block_pairs:
                    pairs = score.render_block_pair_scoring_module(pose)(
                        pose.coords, sum_terms=False, apply_weights=False
                    )
                    np.savez_compressed(
                        args.output / f"{prefix}-{suffix}-pairs.npz",
                        energies=pairs.detach().cpu().numpy(),
                        score_types=np.array(
                            [st.name for st in score.all_score_types()]
                        ),
                    )
                atom_keys, elements, types = [], [], []
                for block, type_index in enumerate(pose.block_type_ind[0].tolist()):
                    bt = pose.packed_block_types.active_block_types[type_index]
                    types.append(bt.name)
                    for atom in bt.atoms:
                        atom_keys.append(f"{block}:{bt.name}:{atom.name}")
                        elements.append(atom.atom_type)
                np.savez_compressed(
                    args.output / f"{prefix}-{suffix}.npz",
                    coords=pose.coords.detach().cpu().numpy(),
                    atom_keys=np.array(atom_keys),
                    atom_types=np.array(elements),
                    block_types=np.array(types),
                )
                metadata["runs"].append(
                    dict(
                        backbone=backbone,
                        repeat=repeat,
                        opt_h=opt_h,
                        scores=scores,
                        block_types=types,
                    )
                )
                (args.output / "scores.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                print(
                    prefix,
                    suffix,
                    "lengths",
                    scores["cart_lengths"],
                    "rep",
                    scores["fa_ljrep"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
