import numpy
import torch
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
from tmol.io.details.select_from_canonical import (
    assign_block_types,
    take_block_type_atoms_from_canonical,
)
from tmol.io.details.build_missing_leaf_atoms import (
    _annotate_packed_block_types_atom_is_leaf_atom,
    build_missing_leaf_atoms,
)


from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.tests.autograd import gradcheck


def not_any_nancoord(coords):
    return torch.logical_not(torch.any(torch.isnan(coords), dim=3))


def ats_to_not_rebuild(pbt, torch_device):
    # now let's turn off building of some of these atoms in particular
    # note that we are rebuilding OXT using the (not-absent) bbO.
    ats_to_not_rebuild_at_all = ["O", "OXT"]
    ats_generally_to_leave_be = {
        "GLU": "OE2",
        "ASP": "OD2",
        "TRP": "CD2",
    }
    ats_in_bts_to_leave_alone = torch.ones(
        (pbt.n_types, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    for i, bt in enumerate(pbt.active_block_types):
        for j, at in enumerate(ats_to_not_rebuild_at_all):
            if at in bt.atom_to_idx:
                ats_in_bts_to_leave_alone[i, bt.atom_to_idx[at]] = False
        if bt.base_name in ats_generally_to_leave_be:
            ats_in_bts_to_leave_alone[
                i, bt.atom_to_idx[ats_generally_to_leave_be[bt.base_name]]
            ] = False
    return ats_in_bts_to_leave_alone


def test_build_missing_leaf_atoms(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    (
        ch_id,
        can_rts,
        coords,
    ) = (
        cf["chain_id"],
        cf["res_types"],
        cf["coords"],
    )
    at_is_pres = not_any_nancoord(coords)

    # 2
    found_disulfides, res_type_variants = find_disulfides(co, can_rts, coords)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(co, can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co,
        pbt,
        resolved_atom_is_present,
        ch_id,
        can_rts,
        res_type_variants,
        found_disulfides,
    )

    block_coords, missing_atoms, real_atoms, _ = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, at_is_pres
    )

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    real_blocks = block_types64 >= 0
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]].to(
        torch.bool
    )
    ats_in_bts_to_leave_alone = ats_to_not_rebuild(pbt, torch_device)
    block_at_allow_rebuild = torch.ones(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    block_at_allow_rebuild[real_blocks] = ats_in_bts_to_leave_alone[
        block_types64[real_blocks]
    ]

    block_at_to_rebuild = torch.logical_and(block_at_is_leaf, block_at_allow_rebuild)

    missing_atoms[block_at_to_rebuild] = True

    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    new_pose_coords, block_coord_offset = build_missing_leaf_atoms(
        pbt,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
    )

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

    built_leaf_pos = expanded_coords[block_at_is_leaf]
    orig_leaf_pos = block_coords[block_at_is_leaf]

    built_leaf_pos = built_leaf_pos.cpu().numpy()
    orig_leaf_pos = orig_leaf_pos.cpu().numpy()

    numpy.testing.assert_allclose(built_leaf_pos, orig_leaf_pos, atol=1e-1, rtol=1e-3)


def test_build_missing_leaf_atoms_error_handling(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    cf["coords"][0, :, 1, :] = numpy.nan  # turn off all the CAlphas!
    (
        ch_id,
        can_rts,
        coords,
    ) = (
        cf["chain_id"],
        cf["res_types"],
        cf["coords"],
    )

    at_is_pres = not_any_nancoord(coords)

    # 2
    found_disulfides, res_type_variants = find_disulfides(co, can_rts, coords)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(co, can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co,
        pbt,
        resolved_atom_is_present,
        ch_id,
        can_rts,
        res_type_variants,
        found_disulfides,
    )

    block_coords, missing_atoms, real_atoms, _ = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, at_is_pres
    )

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    real_blocks = block_types64 >= 0
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]].to(
        torch.bool
    )
    ats_in_bts_to_leave_alone = ats_to_not_rebuild(pbt, torch_device)
    block_at_allow_rebuild = torch.ones(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    block_at_allow_rebuild[real_blocks] = ats_in_bts_to_leave_alone[
        block_types64[real_blocks]
    ]

    block_at_to_rebuild = torch.logical_and(block_at_is_leaf, block_at_allow_rebuild)

    missing_atoms[block_at_to_rebuild] = True

    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    try:
        build_missing_leaf_atoms(
            pbt,
            block_types64,
            real_atoms,
            block_coords,
            missing_atoms,
            inter_residue_connections,
        )
        assert False
    except ValueError as err:
        gold_msg = "\n".join(
            [
                " ".join(
                    [
                        "Error: missing non-leaf atom",
                        "CA",
                        "on residue",
                        str(i),
                        pbt.active_block_types[block_types64[0, i]].name,
                        "on pose",
                        "0",
                        "real res?",
                        str(real_blocks[0, i].item()),
                    ]
                )
                for i in range(can_rts.shape[1])
            ]
        )
        assert str(err) == gold_msg


def test_build_missing_leaf_atoms_backwards(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    (
        ch_id,
        can_rts,
        coords,
    ) = (
        cf["chain_id"],
        cf["res_types"],
        cf["coords"],
    )
    at_is_pres = not_any_nancoord(coords)

    ats_in_bts_to_leave_alone = ats_to_not_rebuild(pbt, torch_device)

    class FauxModule(torch.nn.Module):
        def __init__(self, coords):
            super(FauxModule, self).__init__()

            self.coords = torch.nn.Parameter(coords)

        def forward(self):
            found_disulfides, res_type_variants = find_disulfides(
                co, can_rts, self.coords
            )
            # 3
            (
                his_taut,
                res_type_variants,
                resolved_coords,
                resolved_atom_is_present,
            ) = resolve_his_tautomerization(
                co, can_rts, res_type_variants, self.coords, at_is_pres
            )

            # now we'll invoke assign_block_types
            (
                block_types64,
                inter_residue_connections64,
                inter_block_bondsep64,
            ) = assign_block_types(
                co,
                pbt,
                resolved_atom_is_present,
                ch_id,
                can_rts,
                res_type_variants,
                found_disulfides,
            )

            (
                block_coords,
                missing_atoms,
                real_atoms,
                _,
            ) = take_block_type_atoms_from_canonical(
                pbt, block_types64, self.coords, at_is_pres
            )

            # now let's just say that all the hydrogen atoms are missing so we can build
            # them back
            _annotate_packed_block_types_atom_is_leaf_atom(pbt)
            n_poses = 1
            max_n_blocks = block_types64.shape[1]
            block_at_is_leaf = torch.zeros(
                (n_poses, max_n_blocks, pbt.max_n_atoms),
                dtype=torch.bool,
                device=torch_device,
            )
            real_blocks = block_types64 >= 0
            block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[
                block_types64[real_blocks]
            ].to(torch.bool)

            block_at_allow_rebuild = torch.ones(
                (n_poses, max_n_blocks, pbt.max_n_atoms),
                dtype=torch.bool,
                device=torch_device,
            )
            block_at_allow_rebuild[real_blocks] = ats_in_bts_to_leave_alone[
                block_types64[real_blocks]
            ]

            block_at_to_rebuild = torch.logical_and(
                block_at_is_leaf, block_at_allow_rebuild
            )

            missing_atoms[block_at_to_rebuild] = True

            inter_residue_connections = inter_residue_connections64.to(torch.int32)
            new_pose_coords, block_coord_offset = build_missing_leaf_atoms(
                pbt,
                block_types64,
                real_atoms,
                block_coords,
                missing_atoms,
                inter_residue_connections,
            )

            return torch.sum(new_pose_coords[:, :, :])

    faux_module = FauxModule(coords)

    optimizer = LBFGS_Armijo(faux_module.parameters(), lr=0.1, max_iter=1)

    def closure():
        optimizer.zero_grad()
        coord_sum = faux_module()
        coord_sum.backward()
        return coord_sum

    optimizer.step(closure)


def test_coord_sum_gradcheck(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device, residue_end=1)
    (
        ch_id,
        can_rts,
        coords,
    ) = (
        cf["chain_id"],
        cf["res_types"],
        cf["coords"],
    )
    at_is_pres = not_any_nancoord(coords)

    # currently: don't treat this as MET:nterm:cterm even though it is both the first
    # and last residue for a chain; instead, do MET:nterm
    res_not_connected = torch.tensor(
        [[[False, True]]], dtype=torch.bool, device=torch_device
    )

    # 2
    found_disulfides, res_type_variants = find_disulfides(co, can_rts, coords)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(co, can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co,
        pbt,
        resolved_atom_is_present,
        ch_id,
        can_rts,
        res_type_variants,
        found_disulfides,
        res_not_connected,
    )

    block_coords, missing_atoms, real_atoms, _ = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, at_is_pres
    )

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    real_blocks = block_types64 >= 0
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]].to(
        torch.bool
    )
    ats_in_bts_to_leave_alone = ats_to_not_rebuild(pbt, torch_device)
    block_at_allow_rebuild = torch.ones(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    block_at_allow_rebuild[real_blocks] = ats_in_bts_to_leave_alone[
        block_types64[real_blocks]
    ]
    block_at_to_rebuild = torch.logical_and(block_at_is_leaf, block_at_allow_rebuild)

    missing_atoms[block_at_to_rebuild] = True

    inter_residue_connections = inter_residue_connections64.to(torch.int32)

    def coord_score(block_coords):
        new_pose_coords, block_coord_offset = build_missing_leaf_atoms(
            pbt,
            block_types64,
            real_atoms,
            block_coords,
            missing_atoms,
            inter_residue_connections,
        )
        return torch.sum(new_pose_coords[:])

    gradcheck(
        coord_score,
        (block_coords.requires_grad_(True),),
        eps=1e-3,
        atol=5e-2,
        rtol=5e-2,
    )


def test_build_missing_hydrogens_and_oxygens_gradcheck(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    cf = canonical_form_from_pdb(co, ubq_pdb[:810], torch_device)
    (
        ch_id,
        can_rts,
        coords,
    ) = (
        cf["chain_id"],
        cf["res_types"],
        cf["coords"],
    )
    at_is_pres = not_any_nancoord(coords)
    # currently: don't treat this as MET:nterm:cterm even though it is both the first
    # and last residue for a chain; instead, do MET:nterm
    res_not_connected = torch.tensor(
        [[[False, True]]], dtype=torch.bool, device=torch_device
    )

    # 2
    found_disulfides, res_type_variants = find_disulfides(co, can_rts, coords)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(co, can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co,
        pbt,
        resolved_atom_is_present,
        ch_id,
        can_rts,
        res_type_variants,
        found_disulfides,
        res_not_connected,
    )

    block_coords, missing_atoms, real_atoms, _ = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, at_is_pres
    )

    # now let's just say that all the hydrogen atoms are missing so we can build
    # them back
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
    n_poses = 1
    max_n_blocks = block_types64.shape[1]
    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    real_blocks = block_types64 >= 0
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]].to(
        torch.bool
    )
    ats_in_bts_to_leave_alone = ats_to_not_rebuild(pbt, torch_device)
    block_at_allow_rebuild = torch.ones(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=torch_device
    )
    block_at_allow_rebuild[real_blocks] = ats_in_bts_to_leave_alone[
        block_types64[real_blocks]
    ]
    block_at_to_rebuild = torch.logical_and(block_at_is_leaf, block_at_allow_rebuild)

    missing_atoms[block_at_to_rebuild] = True

    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    new_pose_coords, block_coord_offset = build_missing_leaf_atoms(
        pbt,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
    )

    def coord_score(bc):
        # nonlocal new_pose_coords
        new_pose_coords, block_coord_offset = build_missing_leaf_atoms(
            pbt,
            block_types64,
            real_atoms,
            bc,
            missing_atoms,
            inter_residue_connections,
        )

        # slice the coords tensor to create a temp that will avoid a stride of 0
        return torch.sum(new_pose_coords[:, :, :])

    gradcheck(
        coord_score,
        (block_coords.requires_grad_(True),),
        nondet_tol=1e-2,
        eps=1e-2,
        atol=1e-2,
        rtol=1e-2,
    )
