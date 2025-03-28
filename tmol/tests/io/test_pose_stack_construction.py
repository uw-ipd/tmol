import torch
import numpy

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_build_pose_stack_from_canonical_form_ubq(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

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


def test_build_pose_stack_from_canonical_form_pert(torch_device, pertuzumab_pdb):
    co = default_canonical_ordering()
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)
    pbt = default_packed_block_types(torch_device)
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

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


def test_build_pose_stack_from_canonical_form_pert_w_dslf(torch_device, pertuzumab_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)

    disulfides = torch.tensor(
        [[0, 22, 87], [0, 213, 435], [0, 133, 193], [0, 235, 309], [0, 359, 415]],
        dtype=torch.int64,
        device=torch_device,
    )

    pose_stack = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, disulfides=disulfides
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


def test_build_pose_stack_w_disconn_segs(
    torch_device, pertuzumab_and_nearby_erbb2_pdb_and_segments
):
    (
        pert_and_erbb2_lines,
        res_not_connected,
    ) = pertuzumab_and_nearby_erbb2_pdb_and_segments

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pert_and_erbb2_lines, torch_device)
    res_not_connected = torch.tensor(res_not_connected, device=torch_device)

    disulfides = torch.tensor(
        [[0, 22, 87], [0, 213, 435], [0, 133, 193], [0, 235, 309], [0, 359, 415]],
        dtype=torch.int64,
        device=torch_device,
    )

    pose_stack = pose_stack_from_canonical_form(
        co,
        pbt,
        **canonical_form,
        disulfides=disulfides,
        find_additional_disulfides=True,
        res_not_connected=res_not_connected,
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


def test_build_pose_stack_w_disconn_segs_and_insertions(
    torch_device, pertuzumab_and_nearby_erbb2_pdb_and_segments
):
    (
        pert_and_erbb2_lines,
        res_not_connected,
    ) = pertuzumab_and_nearby_erbb2_pdb_and_segments
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pert_and_erbb2_lines, torch_device)

    def get_add_two_fill_shape(x):
        if len(x.shape) >= 3:
            fill_shape = (x.shape[0], 2, *x.shape[2:])
        else:
            fill_shape = (x.shape[0], 2)
        return fill_shape

    def add_two_res(x, fill_value):
        fill_shape = get_add_two_fill_shape(x)
        return torch.cat(
            [
                x[:, :50],
                torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device),
                x[:, 50:],
            ],
            dim=1,
        )

    canonical_form = {n: add_two_res(x, -1) for n, x in canonical_form.items()}

    res_not_connected = torch.tensor(res_not_connected, device=torch_device)
    res_not_connected = add_two_res(res_not_connected, False)

    disulfides_shifted = torch.tensor(
        [[0, 22, 89], [0, 215, 437], [0, 135, 195], [0, 237, 311], [0, 361, 417]],
        dtype=torch.int64,
        device=torch_device,
    )

    pose_stack, chain_ind = pose_stack_from_canonical_form(
        co,
        pbt,
        **canonical_form,
        disulfides=disulfides_shifted,
        find_additional_disulfides=True,
        res_not_connected=res_not_connected,
        return_chain_ind=True,
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
    assert chain_ind.shape == pose_stack.block_type_ind.shape
    assert chain_ind.device == pose_stack.block_type_ind.device


def test_build_pose_stack_from_canonical_form_ubq_w_atom_mapping(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    pose_stack, cf_map, ps_map = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, return_atom_mapping=True
    )
    coords = canonical_form["coords"]

    cf_atom_coords = torch.full_like(coords, numpy.nan)
    cf_atom_coords[cf_map[:, 0], cf_map[:, 1], cf_map[:, 2]] = pose_stack.coords[
        ps_map[:, 0], ps_map[:, 1]
    ]

    numpy.testing.assert_equal(coords.cpu().numpy(), cf_atom_coords.cpu().numpy())


def test_build_pose_stack_with_masked_residues(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    # now let's "mask out" some residues by setting their res_types to -1
    canonical_form["chain_id"][0, ::10] = -1
    canonical_form["res_types"][0, ::10] = -1
    canonical_form["coords"][0, ::10] = numpy.nan
    canonical_form["res_not_connected"] = torch.full(
        (1, 76, 2), False, device=torch_device
    )
    canonical_form["res_not_connected"][0, 1::10, 0] = True
    canonical_form["res_not_connected"][0, 9::10, 1] = True

    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

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
