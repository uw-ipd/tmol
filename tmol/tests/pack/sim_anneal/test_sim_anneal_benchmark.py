import torch
import attr
import numpy
import pytest

from tmol.utility.tensor.common_operations import stretch
from tmol.chemical.restypes import ResidueTypeSet
from tmol.pose.pose_stack import PackedBlockTypes
from tmol.pose.pose_stack import Pose, Poses

from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer.build_rotamers import RotamerSet, build_rotamers
from tmol.pack.rotamer.bounding_spheres import create_rotamer_bounding_spheres

from tmol.pack.sim_anneal.annealer import MCAcceptRejectModule, SelectRanRotModule
from tmol.pack.sim_anneal.compiled.compiled import (
    # pick_random_rotamers,
    # metropolis_accept_reject,
    create_sim_annealer,
    delete_sim_annealer,
    register_standard_random_rotamer_picker,
    register_standard_metropolis_accept_or_rejector,
    run_sim_annealing,
)

from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.system.pose import Pose, Poses
from tmol.score.chemical_database import AtomTypeParamResolver


# class SimAEngine(torch.jit.ScriptModule):
class SimAEngine:
    def __init__(self, temperature, selector, inter_module, mc_accept_reject):

        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.temperature = _p(temperature)
        self.selector = selector
        self.inter_module = inter_module
        self.mc_accept_reject = mc_accept_reject
        self.temp_lj_energies = _p(
            torch.zeros(
                (selector.pose_id_for_context.shape[0] * 2,),
                dtype=torch.float32,
                device=selector.pose_id_for_context.device,
            )
        )
        self.temp_alt_coords = _p(
            torch.zeros(
                (
                    selector.pose_id_for_context.shape[0] * 2,
                    selector.rotamer_coords.shape[1],
                    3,
                ),
                dtype=torch.float32,
                device=selector.pose_id_for_context.device,
            )
        )
        self.temp_alt_ids = _p(
            torch.zeros(
                (selector.pose_id_for_context.shape[0] * 2, 3),
                dtype=torch.int32,
                device=selector.pose_id_for_context.device,
            )
        )

    # @torch.jit.script_method
    # def forward(self, context_coords, context_block_type):

    def go(self, context_coords, context_block_type):
        alt_coords, alt_ids, rr = self.selector.go(context_coords, context_block_type)
        lj_energies, _ = self.inter_module.go(
            context_coords, context_block_type, alt_coords, alt_ids
        )
        accept = self.mc_accept_reject.go(
            self.temperature,
            context_coords,
            context_block_type,
            self.temp_alt_coords,
            self.temp_alt_ids,
            self.temp_lj_energies.unsqueeze(0),
        )
        return accept


@pytest.mark.benchmark(group="simulated_annealing")
# @pytest.mark.parametrize("n_poses", [10, 30, 100, 300, 1000])
@pytest.mark.parametrize("n_poses", [100])  # 300
@pytest.mark.parametrize("n_components", [1])
def test_run_simA_benchmark(
    benchmark,
    n_poses,
    n_components,
    default_database,
    fresh_default_restype_set,
    rts_ubq_res,
    torch_device,
    dun_sampler,
):

    # print("torch device", torch_device)
    # def _p(t):
    #     return torch.nn.Parameter(t, requires_grad=False)

    max_n_blocks = len(rts_ubq_res)

    p = Pose.from_residues_one_chain(rts_ubq_res, torch_device)
    poses = Poses.from_poses([p] * n_poses, torch_device)
    # print("poses device", poses.device, poses.coords.device)

    palette = PackerPalette(fresh_default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)
    # print("poses device after rotamer building", poses.device, poses.coords.device)
    # print("n_rotamers", rotamer_set.coords.shape[0] // n_poses)

    bounding_spheres = create_rotamer_bounding_spheres(poses, rotamer_set)

    # come up with the intial rotamer assignment
    context_coords = poses.coords.clone()
    # print("context_coords device")
    # print(context_coords.device)
    rand_rot = torch.floor(
        torch.rand((n_poses, max_n_blocks), dtype=torch.float, device=torch_device)
        * rotamer_set.n_rots_for_block.to(torch.float32)
    ).to(torch.int64)
    rand_rot_global = rotamer_set.rot_offset_for_block + rand_rot
    packable_blocks = rotamer_set.n_rots_for_block != 0
    context_coords[packable_blocks] = rotamer_set.coords[
        rand_rot_global[packable_blocks]
    ]
    context_block_type = poses.block_type_ind.clone()
    context_block_type[packable_blocks] = rotamer_set.block_type_ind_for_rot[
        rand_rot_global[packable_blocks]
    ].to(torch.int32)

    n_poses_arange = torch.arange(n_poses, dtype=torch.int32, device=torch_device)

    annealer = torch.zeros((1,), dtype=torch.int64, device="cpu")
    create_sim_annealer(annealer)
    # print("annealer")
    # print(annealer)

    pose_id_for_context = n_poses_arange
    n_rots_for_pose = rotamer_set.n_rots_for_pose.to(torch.int32)
    rot_offset_for_pose = rotamer_set.rot_offset_for_pose.to(torch.int32)
    block_type_ind_for_rot = rotamer_set.block_type_ind_for_rot.to(torch.int32)
    block_ind_for_rot = rotamer_set.block_ind_for_rot.to(torch.int32)
    rotamer_coords = rotamer_set.coords

    alternate_coords = torch.zeros(
        (n_poses * 2, poses.packed_block_types.max_n_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    alternate_id = torch.zeros((n_poses * 2, 3), dtype=torch.int32, device=torch_device)
    random_rotamers = torch.zeros((n_poses,), dtype=torch.int32, device=torch_device)
    # print("alternate_coords initial")
    # print(alternate_coords.shape)
    # print("alternate_id initial")
    # print(alternate_id.shape)

    annealer_event = torch.zeros(1, dtype=torch.int64, device="cpu")
    score_events = torch.zeros(n_components, dtype=torch.int64, device="cpu")

    register_standard_random_rotamer_picker(
        context_coords,
        context_block_type,
        pose_id_for_context,
        n_rots_for_pose,
        rot_offset_for_pose,
        block_type_ind_for_rot,
        block_ind_for_rot,
        rotamer_coords,
        alternate_coords,
        alternate_id,
        random_rotamers,
        annealer_event,
        annealer,
    )

    temperature = torch.full((1,), 100, dtype=torch.float32, device=torch.device("cpu"))
    rotamer_component_energies = torch.zeros(
        (n_components, 2 * n_poses), dtype=torch.float32, device=torch_device
    )
    accepted = torch.zeros((n_poses,), dtype=torch.int32, device=torch_device)

    register_standard_metropolis_accept_or_rejector(
        temperature,
        context_coords,
        context_block_type,
        alternate_coords,
        alternate_id,
        rotamer_component_energies,
        accepted,
        score_events,
        annealer,
    )

    # and the score function

    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    weights = {"lj": 1.0, "lk": 1.0}

    for bt in poses.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(poses.packed_block_types)
    ljlk_energy.setup_poses(poses)

    inter_module = ljlk_energy.inter_module(
        poses.packed_block_types, poses, n_poses_arange, bounding_spheres, weights
    )

    for i in range(n_components):
        inter_module.register_with_sim_annealer(
            context_coords,
            context_block_type,
            alternate_coords,
            alternate_id,
            rotamer_component_energies[i],
            score_events[i : i + 1],
            annealer_event,
            annealer,
        )

    torch.random.manual_seed(1111)
    run_sim_annealing(annealer)

    #
    # temperature = torch.full((1,), 100, dtype=torch.float32, device=torch.device("cpu"))
    #
    # alt_coords, alt_ids, rr = selector.go(context_coords, context_block_type)
    # lj_energies, events = inter_module.go(context_coords, context_block_type, alt_coords, alt_ids)
    # accept = mc_accept_reject.go(
    #     temperature,
    #     context_coords,
    #     context_block_type,
    #     alt_coords,
    #     alt_ids,
    #     lj_energies.unsqueeze(0)
    # )
    #
    #
    # lj_energies = torch.zeros((n_poses*2,), dtype=torch.float32, device=torch_device)
    #
    # tight_loop = SimAEngine(temperature, selector, inter_module, mc_accept_reject)
    #
    # @benchmark
    # def one_step_simA():
    #     return tight_loop.go(context_coords, context_block_type)
    #
    # accept = one_step_simA

    delete_sim_annealer(annealer)
    # print("annealer")
    # print(annealer)

    # make sure that the device is still operating; that we haven't corrupted GPU memory
    torch.arange(100, device=torch_device).sum()
