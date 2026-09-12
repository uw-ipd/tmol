"""Parse once with AtomWorks; prepare and score with tmol's matching database.

Install tmol[atomworks]; see ATOMWORKS.md for release limitations. Example:
  ALLOW_BIOTITE_CCD=1 python review/pr503/atomworks_example.py \
      tmol/tests/data/ncaa_fixtures/capped_peptide_ace_nh2.cif --device cpu
"""

import argparse

import torch

from tmol.io import pose_stack_from_cif
from tmol.score import beta2016_score_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    pose, context = pose_stack_from_cif(
        args.structure,
        device,
        reader="atomworks",
        prepare_ligands=True,
        ligand_seed=args.seed,
        no_optH=True,
        return_context=True,
    )
    score = beta2016_score_function(device, param_db=context.parameter_database)
    module = score.render_whole_pose_scoring_module(pose)
    coords = pose.coords.detach().clone().requires_grad_(True)
    energy = module(coords)
    energy.sum().backward()
    assert torch.isfinite(energy).all() and torch.isfinite(coords.grad).all()
    print(
        {
            "poses": pose.n_poses,
            "blocks": pose.max_n_blocks,
            "energy": energy.detach().cpu().tolist(),
        }
    )


if __name__ == "__main__":
    main()
