import numpy
import torch
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.details.canonical_packed_block_types import (
    default_canonical_packed_block_types,
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

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo


def test_build_missing_hydrogens(torch_device, ubq_pdb):
    numpy.set_printoptions(threshold=100000)

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
    # print("block coords")
    # print(block_coords.cpu().numpy())

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_h(pbt, atr)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_h = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    real_blocks = block_types64 >= 0
    block_at_is_h[real_blocks] = pbt.is_hydrogen[block_types64[real_blocks]].to(
        torch.bool
    )
    missing_atoms[block_at_is_h] = 1

    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    new_pose_coords, block_coord_offset = build_missing_hydrogens(
        pbt,
        atr,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
    )

    # print("new pose coords")
    # print(new_pose_coords.cpu().numpy())

    # now expand the pose coords back out to n-poses x max-n-res x max-n-ats x 3
    # and then lets compare the coordinates of the newly built coordinates to what
    # was already there in the input pdb
    n_ats_per_block_arange_expanded = (
        torch.arange(pbt.max_n_atoms, dtype=torch.int64, device=torch_device)
        .repeat(max_n_blocks)
        .view(1, max_n_blocks, pbt.max_n_atoms)
    )
    n_ats_per_block = torch.zeros(
        (n_poses, max_n_blocks), dtype=torch.int32, device=torch_device
    )
    n_ats_per_block[real_blocks] = pbt.n_atoms[block_types64[real_blocks]]
    real_expanded_pose_ats = (
        n_ats_per_block_arange_expanded < n_ats_per_block.unsqueeze(2)
    )
    expanded_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    expanded_coords[real_expanded_pose_ats] = new_pose_coords[:]

    built_h_pos = expanded_coords[block_at_is_h]
    orig_h_pos = block_coords[block_at_is_h]

    built_h_pos = built_h_pos.cpu().numpy()
    orig_h_pos = orig_h_pos.cpu().numpy()

    # print("built_h_pos")
    # print(built_h_pos)

    # from tmol.io.write_pose_stack_pdb import atom_records_from_coords
    # from tmol.io.pdb_parsing import to_pdb
    #
    # atom_records = atom_records_from_coords(
    #     pbt, ch_beg, block_types64, new_pose_coords, block_coord_offset
    # )
    #
    # with open("test_build_H.pdb", "w") as fid:
    #     fid.writelines(to_pdb(atom_records))

    # for i in range(built_h_pos.shape[0]):
    #     print(i, built_h_pos[i], "vs", orig_h_pos[i], "dist",  numpy.linalg.norm(built_h_pos[i] - orig_h_pos[i]))

    numpy.testing.assert_allclose(built_h_pos[1:], orig_h_pos[1:], atol=1e-2, rtol=1e-3)


def test_build_missing_hydrogens_backwards(torch_device, ubq_pdb):
    numpy.set_printoptions(threshold=100000)

    # print("ubq_pdb")
    # print(ubq_pdb)

    pbt, atr = default_canonical_packed_block_types(torch_device)
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    class FauxModule(torch.nn.Module):
        def __init__(self, coords):
            super(FauxModule, self).__init__()

            self.coords = torch.nn.Parameter(coords)

        def forward(self):
            found_disulfides, res_type_variants = find_disulfides(
                can_rts, self.coords, at_is_pres
            )
            # 3
            (
                his_taut,
                res_type_variants,
                resolved_coords,
                resolved_atom_is_present,
            ) = resolve_his_tautomerization(
                can_rts, res_type_variants, self.coords, at_is_pres
            )

            # now we'll invoke assign_block_types
            (
                block_types64,
                inter_residue_connections64,
                inter_block_bondsep64,
            ) = assign_block_types(
                pbt, ch_beg, can_rts, res_type_variants, found_disulfides
            )

            (
                block_coords,
                missing_atoms,
                real_atoms,
            ) = take_block_type_atoms_from_canonical(
                pbt, ch_beg, block_types64, self.coords, at_is_pres
            )
            # print("block coords")
            # print(block_coords.cpu().numpy())

            # now let's just say that all the hydrogen atoms are missing so we can build
            # them back
            _annotate_packed_block_types_atom_is_h(pbt, atr)
            n_poses = 1
            max_n_blocks = block_types64.shape[1]
            block_at_is_h = torch.zeros(
                (n_poses, max_n_blocks, pbt.max_n_atoms),
                dtype=torch.bool,
                device=torch_device,
            )
            real_blocks = block_types64 >= 0
            block_at_is_h[real_blocks] = pbt.is_hydrogen[block_types64[real_blocks]].to(
                torch.bool
            )
            missing_atoms[block_at_is_h] = 1

            inter_residue_connections = inter_residue_connections64.to(torch.int32)
            new_pose_coords, block_coord_offset = build_missing_hydrogens(
                pbt,
                atr,
                block_types64,
                real_atoms,
                block_coords,
                missing_atoms,
                inter_residue_connections,
            )

            return torch.sum(new_pose_coords)

    faux_module = FauxModule(coords)

    optimizer = LBFGS_Armijo(faux_module.parameters(), lr=0.1, max_iter=20)

    def closure():
        optimizer.zero_grad()
        coord_sum = faux_module()
        coord_sum.backward()
        return coord_sum

    optimizer.step(closure)
