import pytest
import torch

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io import pose_stack_from_pdb
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


@pytest.fixture
def ubq_40_60_pose_stack(ubq_pdb, torch_device):
    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:40], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:60], torch_device
    # )
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=40)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=60)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    return poses


@pytest.fixture
def fresh_default_packed_block_types(fresh_default_restype_set, torch_device):
    return PackedBlockTypes.from_restype_list(
        fresh_default_restype_set.chem_db,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )


@pytest.fixture
def stack_of_two_six_res_ubqs(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    # _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=6
    )

    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)
    return PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)


@pytest.fixture
def stack_of_two_six_res_ubqs_no_term(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    # _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=1, residue_end=7
    )

    res_not_connected = torch.zeros((1, 6, 2), dtype=torch.bool, device=torch_device)
    res_not_connected[0, 0, 0] = True  # simplest test case: not N-term
    res_not_connected[0, 5, 1] = True  # simplest test case: not C-term
    pose_stack = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, res_not_connected=res_not_connected
    )
    return PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)


@pytest.fixture
def jagged_stack_of_465_res_ubqs(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    # _annotate_packed_block_type_with_gen_scan_path_segs(pbt)

    def pose_stack_of_nres(nres):
        canonical_form = canonical_form_from_pdb(
            co, ubq_pdb, torch_device, residue_start=0, residue_end=nres
        )
        return pose_stack_from_canonical_form(co, pbt, **canonical_form)

    return PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x) for x in [4, 6, 5]], torch_device
    )
