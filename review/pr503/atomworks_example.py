"""Parse once with AtomWorks; prepare and score with tmol's matching database.

Requires the reviewed AtomWorks branch; see ATOMWORKS.md. Example:
  ALLOW_BIOTITE_CCD=1 python review/pr503/atomworks_example.py \
      tmol/tests/data/ncaa_fixtures/capped_peptide_ace_nh2.cif --device cpu
"""

import argparse

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
import torch

from tmol.io import pose_stack_from_biotite
from tmol.score import beta2016_score_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    config = ParseConfig(
        model=1,
        build_assembly=None,
        remove_ccds=(),
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        long_bond_policy="keep",
        struct_conn_distance_policy="keep",
        add_bond_types_from_struct_conn=("covale", "disulf"),
        hydrogen_policy="remove",
        ccd_mirror_path=None,
        add_id_and_entity_annotations=False,
    )
    array = parse(args.structure, config=config)["asym_unit"]
    array = array[0] if array.coord.ndim == 3 else array
    pose, context = pose_stack_from_biotite(
        array,
        device,
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
