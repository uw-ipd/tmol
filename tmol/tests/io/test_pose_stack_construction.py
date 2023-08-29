import torch
import numpy

from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_build_pose_stack_from_canonical_form_ubq(torch_device, ubq_pdb):
    canonical_form = canonical_form_from_pdb_lines(ubq_pdb, torch_device)
    pose_stack = pose_stack_from_canonical_form(*canonical_form)

    assert pose_stack.packed_block_types.device == torch_device
    assert pose_stack.coords.device == torch_device
    assert pose_stack.block_coord_offset.device == torch_device
    assert pose_stack.block_coord_offset64.device == torch_device
    assert pose_stack.inter_residue_connections.device == torch_device
    assert pose_stack.inter_residue_connections64.device == torch_device
    assert pose_stack.inter_block_bondsep.device == torch_device
    assert pose_stack.inter_block_bondsep64.device == torch_device
    assert pose_stack.block_type_ind.device == torch_device
    assert pose_stack.block_type_ind64.device == torch_device
    assert pose_stack.device == torch_device


def test_build_pose_stack_from_canonical_form_pert(torch_device, pertuzumab_lines):
    canonical_form = canonical_form_from_pdb_lines(pertuzumab_lines, torch_device)
    pose_stack = pose_stack_from_canonical_form(*canonical_form)

    assert pose_stack.packed_block_types.device == torch_device
    assert pose_stack.coords.device == torch_device
    assert pose_stack.block_coord_offset.device == torch_device
    assert pose_stack.block_coord_offset64.device == torch_device
    assert pose_stack.inter_residue_connections.device == torch_device
    assert pose_stack.inter_residue_connections64.device == torch_device
    assert pose_stack.inter_block_bondsep.device == torch_device
    assert pose_stack.inter_block_bondsep64.device == torch_device
    assert pose_stack.block_type_ind.device == torch_device
    assert pose_stack.block_type_ind64.device == torch_device
    assert pose_stack.device == torch_device


def test_build_pose_stack_from_canonical_form_pert_w_dslf(
    torch_device, pertuzumab_lines
):
    canonical_form = canonical_form_from_pdb_lines(pertuzumab_lines, torch_device)

    disulfides = torch.tensor(
        [[0, 22, 87], [0, 213, 435], [0, 133, 193], [0, 235, 309], [0, 359, 415]],
        dtype=torch.int64,
        device=torch_device,
    )

    pose_stack = pose_stack_from_canonical_form(*canonical_form, disulfides)

    assert pose_stack.packed_block_types.device == torch_device
    assert pose_stack.coords.device == torch_device
    assert pose_stack.block_coord_offset.device == torch_device
    assert pose_stack.block_coord_offset64.device == torch_device
    assert pose_stack.inter_residue_connections.device == torch_device
    assert pose_stack.inter_residue_connections64.device == torch_device
    assert pose_stack.inter_block_bondsep.device == torch_device
    assert pose_stack.inter_block_bondsep64.device == torch_device
    assert pose_stack.block_type_ind.device == torch_device
    assert pose_stack.block_type_ind64.device == torch_device
    assert pose_stack.device == torch_device


def test_build_pose_stack_w_disconn_segs(torch_device, pert_and_nearby_erbb2):
    pert_and_erbb2_lines, seg_lengths = pert_and_nearby_erbb2
    canonical_form = canonical_form_from_pdb_lines(pert_and_erbb2_lines, torch_device)

    seg_range_end = numpy.cumsum(numpy.array(seg_lengths, dtype=numpy.int32))
    seg_range_start = numpy.concatenate(
        (numpy.zeros((1,), dtype=numpy.int32), seg_range_end[:-1])
    )
    n_res_tot = seg_range_end[-1]
    res_not_connected = numpy.zeros((1, n_res_tot, 2), dtype=numpy.bool)
    # do not make any of the ERBB2 residues n- or c-termini,
    # and also do not connect residues that are both part of that chain
    # that span gaps
    res_not_connected[0, seg_range_start[2:], 0] = True
    res_not_connected[0, seg_range_end[2:] - 1, 1] = True
    res_not_connected = torch.tensor(res_not_connected, device=torch_device)

    disulfides = torch.tensor(
        [[0, 22, 87], [0, 213, 435], [0, 133, 193], [0, 235, 309], [0, 359, 415]],
        dtype=torch.int64,
        device=torch_device,
    )

    pose_stack = pose_stack_from_canonical_form(
        *canonical_form, disulfides, res_not_connected
    )

    assert pose_stack.packed_block_types.device == torch_device
    assert pose_stack.coords.device == torch_device
    assert pose_stack.block_coord_offset.device == torch_device
    assert pose_stack.block_coord_offset64.device == torch_device
    assert pose_stack.inter_residue_connections.device == torch_device
    assert pose_stack.inter_residue_connections64.device == torch_device
    assert pose_stack.inter_block_bondsep.device == torch_device
    assert pose_stack.inter_block_bondsep64.device == torch_device
    assert pose_stack.block_type_ind.device == torch_device
    assert pose_stack.block_type_ind64.device == torch_device
    assert pose_stack.device == torch_device
