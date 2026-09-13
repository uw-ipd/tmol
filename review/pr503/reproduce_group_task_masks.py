"""Report group rotamers emitted after disabling its sampler or one member."""

import torch
from tmol.io import pose_stack_from_biotite
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.tests.pack.rotamer.test_group_constraints import crosslinked_lysines
from tmol.tests.pack.test_conjugated_group_packing import _task
from tmol.pose._conjugated_groups import find_conjugated_groups

pose, context = pose_stack_from_biotite(
    crosslinked_lysines("free"),
    torch.device("cpu"),
    prepare_ligands=True,
    no_optH=True,
    ligand_seed=503,
    return_context=True,
)
group = find_conjugated_groups(pose)[0]
for mode in ("all_sampler", "one_block"):
    task, sampler = _task(pose, context.parameter_database, pose.device)
    mask = torch.zeros((1, pose.max_n_blocks), dtype=torch.bool)
    if mode == "all_sampler":
        mask[0, list(group.blocks)] = True
        task.disable_sampler_by_block_mask(sampler, mask)
    else:
        mask[0, group.blocks[-1]] = True
        task.disable_packing_by_block_mask(mask)
    try:
        _, rots = build_rotamers(
            pose,
            SetPackerTask.from_packer_task(task),
            context.parameter_database.chemical,
        )
        print(
            mode,
            "mask",
            mask.tolist(),
            "member_counts",
            [int(rots.n_rots_for_block[0, b]) for b in group.blocks],
        )
    except Exception as error:
        print(mode, type(error).__name__, str(error))
