from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.kinematics.move_map import MoveMap
from tmol.kinematics.fold_forest import FoldForest
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.pack.packer_task import PackerPalette, PackerTask
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.score.score_types import ScoreType
from tmol.optimization.kin_min import run_kin_min


def fast_relax(
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        packer_pallete: PackerPalette,
        move_map: MoveMap,
        fold_forest: FoldForest,
        task_operations = None
):
    # Jack Maguire's tuned MonomerRelax2019.txt
    # repeat %%nrepeats%%
    # coord_cst_weight 1.0
    # scale:fa_rep 0.040
    # repack
    # scale:fa_rep 0.051
    # min 0.01
    # coord_cst_weight 0.5
    # scale:fa_rep 0.265
    # repack
    # scale:fa_rep 0.280
    # min 0.01
    # coord_cst_weight 0.0
    # scale:fa_rep 0.559
    # repack
    # scale:fa_rep 0.581
    # min 0.01
    # coord_cst_weight 0.0
    # scale:fa_rep 1
    # repack
    # min 0.00001
    # accept_to_best
    # endrepeat


    if task_operations is None:
        # Create a default task operation that
        # 1. builds rotamers using the Dunbrack rotamer library
        # 2. restricts to repacking
        # 3. includes the current rotamer

        torch_device = pose_stack.device
        from tmol.score.dunbrack.params import DunbrackParamResolver
        from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
        import tmol.database

        default_database = tmol.database.ParameterDatabase.get_default()
        param_resolver = DunbrackParamResolver.from_database(
            default_database.scoring.dun, torch_device
        )
        dun_sampler = DunbrackChiSampler.from_database(param_resolver)
        
        def default_op(task):
            task.restrict_to_repacking()
            task.set_include_current()

            fixed_sampler = FixedAAChiSampler()
            task.add_conformer_sampler(dun_sampler)
            task.add_conformer_sampler(fixed_sampler)
        task_operations = [default_op]

    fa_rep_start = sfxn._weights[ScoreType.fa_ljrep.value]

    rpms_args = [
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        packer_pallete,
        0,
        0,
        task_operations
    ]
    ps = pose_stack
    for _ in range(5):
        rpms_args[0] = ps
        rpms_args[5] = 0.040 * fa_rep_start
        rpms_args[6] = 0.051 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = 0.265 * fa_rep_start
        rpms_args[6] = 0.280 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = 0.040 * fa_rep_start
        rpms_args[6] = 0.051 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = 0.559 * fa_rep_start
        rpms_args[6] = 0.581 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = fa_rep_start
        rpms_args[6] = fa_rep_start
        ps = relax_pack_min_step(*rpms_args)
    return ps

def relax_pack_min_step(
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        packer_pallete,
        fa_rep_pack_weight,
        fa_rep_min_weight,
        task_operations
    ):
    
    task = PackerTask(pose_stack, packer_pallete)
    for op in task_operations:
        op(task)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_pack_weight)
    print("packing with fa_rep of", fa_rep_pack_weight)
    packed_pose_stack = pack_rotamers(pose_stack, sfxn, task)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_min_weight)
    print("minimizing with fa_rep of", fa_rep_min_weight)
    minimized_pose_stack = run_kin_min(packed_pose_stack, sfxn, fold_forest, move_map)
    return minimized_pose_stack




    