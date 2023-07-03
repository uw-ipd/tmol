import torch
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.basic_resolution import pose_stack_from_canonical_form


def test_build_pose_stack_from_canonical_form_ubq(torch_device, ubq_pdb):
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    pose_stack = pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)

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
