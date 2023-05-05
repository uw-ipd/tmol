import numpy
import torch
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.details.canonical_packed_block_types import (
    default_canonical_packed_block_types
)
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
from tmol.io.details.select_from_canonical import (
    assign_block_types,
    take_block_type_atoms_from_canonical,
)
from tmol.io.details.build_missing_hydrogens import (
    _annotate_packed_block_types_atom_is_h,
    build_missing_hydrogens,
)


def test_build_missing_hydrogens(torch_device, ubq_pdb):
    pbt, atr = default_canonical_packed_block_types(torch_device)
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    # 2
    found_disulfides, res_type_variants = find_disulfides(can_rts, coords, at_is_pres)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(pbt, ch_beg, can_rts, res_type_variants, found_disulfides)

    block_coords, missing_atoms, real_atoms = take_block_type_atoms_from_canonical(
        pbt, ch_beg, block_types64, coords, at_is_pres
    )

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_h(pbt, atom_type_resolver)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_h = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.int32
    )
    block_at_is_h[real_blocks] = pbt.is_hydrogen[block_types64[real_blocks]]
    missing_atoms[block_at_is_h] = 1

    new_pose_coords, block_coord_offset = build_missing_hydrogens(
        pbt, atr, block_types64, real_atoms, block_coords, missing_atoms
    )

    # now expand the pose coords back out to n-poses x max-n-res x max-n-ats x 3
    # and then lets compare the coordinates of the newly built coordinates to what
    # was already there in the input pdb
    new_block_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    n_ats_per_block_arange_expanded = (
        torch.arange(pbt.max_n_atoms, dtype=torch.int64, device=torch_device)
        .repeat(max_n_blocks)
        .view(1, max_n_blocks, pbt.max_n_atoms)
    )
    real_expanded_pose_ats = (
        n_ats_per_block_arange_expanded < pbt.n_ats_per_block[block_types64]
    )
    expanded_coords[real_expanded_pose_ats] = new_pose_coords[:]

    built_h_pos = expanded_coords[block_at_is_h]
    orig_h_pos = block_corods[block_at_is_h]

    built_h_pos = build_h_pos.cpu().numpy()
    orig_h_pos = orig_h_pos.cpu().numpy()

    numpy.testing.assert_close(built_h_pos, orig_h_pos, atol=1e-2, rtol=1e-3)
