import torch

from tmol.relax.fast_relax import fast_relax
import time

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

# from tmol.pack.compiled.compiled import build_interaction_graph
from tmol.pack.packer_task import PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.kinematics.move_map import MoveMap
from tmol.kinematics.fold_forest import FoldForest

# from tmol.pack.rotamer.build_rotamers import build_rotamers
# from tmol.pack.rotamer.fixed_aa_chi_sampler import (
#     FixedAAChiSampler,
# )
# from tmol.pack.datatypes import PackerEnergyTables
# from tmol.pack.simulated_annealing import run_simulated_annealing
# from tmol.pack.impose_rotamers import impose_top_rotamer_assignments

from tmol.io import pose_stack_from_pdb
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

from tmol.pack.pack_rotamers import pack_rotamers


def get_relax_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.ref, 1.0)
    sfxn.set_weight(ScoreType.disulfide, 1.0)

    return sfxn

def test_fast_relax_100(default_database, ubq_pdb, dun_sampler, torch_device):

    # if torch_device == torch.device("cpu"):
    #     return
    n_poses = 3
    # print("Device!", torch_device)

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    palette = PackerPalette(restype_set)
    fold_forest = FoldForest.polymeric_forest(
        torch.sum(pose_stack.block_type_ind != -1, dim=1).detach().cpu().numpy()
    )

    def task_op(task):
        task.restrict_to_repacking()
        task.set_include_current()

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)

    start_time = time.perf_counter()

    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        mm,
        fold_forest,
        [task_op]
    )

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")
    
