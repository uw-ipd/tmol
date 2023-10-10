import torch
import numpy

from tmol.io.pose_stack_deconstruction import (
    _annotate_packed_block_types_w_termini_types,
    canonical_form_from_pose_stack,
)
from tmol.io.canonical_ordering import (
    canonical_form_from_pdb_lines,
    max_n_canonical_atoms,
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms_v2,
)

from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_canonical_form_from_ubq_pose(ubq_pdb, torch_device):
    canonical_form = canonical_form_from_pdb_lines(ubq_pdb, torch_device)
    # unpack
    (cf_orig_chain_id, cf_orig_res_types, cf_orig_coords, cf_orig_at_is_pres) = tuple(
        x.clone() for x in canonical_form
    )

    assert not torch.any(torch.isnan(cf_orig_coords[cf_orig_at_is_pres.to(torch.bool)]))

    pose_stack = pose_stack_from_canonical_form(*canonical_form)

    restored_canonical_form = canonical_form_from_pose_stack(pose_stack)

    # unpack
    (
        cf_orig_chain_id2,
        cf_orig_res_types2,
        cf_orig_coords2,
        cf_orig_at_is_pres2,
    ) = canonical_form

    (
        chain_id,
        cf_res_types,
        cf_coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = restored_canonical_form

    numpy.testing.assert_equal(
        cf_orig_chain_id.cpu().numpy(), cf_orig_chain_id2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_res_types.cpu().numpy(), cf_orig_res_types2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_coords.cpu().numpy(), cf_orig_coords2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_at_is_pres.cpu().numpy(), cf_orig_at_is_pres2.cpu().numpy()
    )

    assert chain_id.device == torch_device
    assert cf_res_types.device == torch_device
    assert cf_coords.device == torch_device
    assert atom_is_present.device == torch_device
    assert res_not_connected.device == torch_device
    numpy.testing.assert_equal(chain_id.cpu().numpy(), cf_orig_chain_id.cpu().numpy())
    numpy.testing.assert_equal(
        cf_res_types.cpu().numpy(), cf_orig_res_types.cpu().numpy()
    )
    numpy.testing.assert_equal(cf_coords.cpu().numpy(), cf_orig_coords.cpu().numpy())
    numpy.testing.assert_equal(
        atom_is_present.cpu().numpy(), cf_orig_at_is_pres.cpu().numpy()
    )
    gold_disulfides = numpy.empty((0, 3), dtype=numpy.int64)
    numpy.testing.assert_equal(gold_disulfides, disulfides.cpu().numpy())

    gold_res_not_connected = numpy.zeros((1, chain_id.shape[1], 2), dtype=numpy.bool)
    numpy.testing.assert_equal(res_not_connected.cpu().numpy(), gold_res_not_connected)


def test_canonical_form_from_pertuzumab_pose(pertuzumab_pdb, torch_device):
    canonical_form = canonical_form_from_pdb_lines(pertuzumab_pdb, torch_device)
    # unpack and copy before pose stack construction
    (cf_orig_chain_id, cf_orig_res_types, cf_orig_coords, cf_orig_at_is_pres) = tuple(
        x.clone() for x in canonical_form
    )
    assert not torch.any(torch.isnan(cf_orig_coords[cf_orig_at_is_pres.to(torch.bool)]))

    pose_stack = pose_stack_from_canonical_form(*canonical_form)

    restored_canonical_form = canonical_form_from_pose_stack(pose_stack)

    # unpack
    (
        cf_orig_chain_id2,
        cf_orig_res_types2,
        cf_orig_coords2,
        cf_orig_at_is_pres2,
    ) = canonical_form
    (
        chain_id,
        cf_res_types,
        cf_coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = restored_canonical_form

    numpy.testing.assert_equal(
        cf_orig_chain_id.cpu().numpy(), cf_orig_chain_id2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_res_types.cpu().numpy(), cf_orig_res_types2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_coords.cpu().numpy(), cf_orig_coords2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_at_is_pres.cpu().numpy(), cf_orig_at_is_pres2.cpu().numpy()
    )

    assert chain_id.device == torch_device
    assert cf_res_types.device == torch_device
    assert cf_coords.device == torch_device
    assert atom_is_present.device == torch_device
    assert res_not_connected.device == torch_device
    numpy.testing.assert_equal(chain_id.cpu().numpy(), cf_orig_chain_id.cpu().numpy())
    numpy.testing.assert_equal(
        cf_res_types.cpu().numpy(), cf_orig_res_types.cpu().numpy()
    )

    np_cf_coords = cf_coords.cpu().numpy()
    np_cf_orig_coords = cf_orig_coords.cpu().numpy()
    np_cf_orig_at_is_pres = cf_orig_at_is_pres.to(torch.bool).cpu().numpy()

    numpy.testing.assert_equal(
        np_cf_coords[np_cf_orig_at_is_pres], np_cf_orig_coords[np_cf_orig_at_is_pres]
    )

    gold_disulfides = torch.tensor(
        [(0, 22, 87), (0, 133, 193), (0, 213, 435), (0, 235, 309), (0, 359, 415)],
        dtype=torch.int64,
        device=torch_device,
    )

    numpy.testing.assert_equal(gold_disulfides.cpu().numpy(), disulfides.cpu().numpy())

    gold_res_not_connected = numpy.zeros((1, chain_id.shape[1], 2), dtype=numpy.bool)
    numpy.testing.assert_equal(res_not_connected.cpu().numpy(), gold_res_not_connected)


def test_canonical_form_from_pertuzumab_and_antigen_pose(
    pertuzumab_and_nearby_erbb2_pdb_and_segments, torch_device
):
    (
        pert_and_erbb2_lines,
        res_not_connected_orig_np,
    ) = pertuzumab_and_nearby_erbb2_pdb_and_segments
    canonical_form = canonical_form_from_pdb_lines(pert_and_erbb2_lines, torch_device)
    res_not_connected_orig = torch.tensor(
        res_not_connected_orig_np, device=torch_device
    )
    # unpack and copy before pose stack construction
    (cf_orig_chain_id, cf_orig_res_types, cf_orig_coords, cf_orig_at_is_pres) = tuple(
        x.clone() for x in canonical_form
    )
    assert not torch.any(torch.isnan(cf_orig_coords[cf_orig_at_is_pres.to(torch.bool)]))

    pose_stack, chain_id_from_constr = pose_stack_from_canonical_form(
        *canonical_form, res_not_connected=res_not_connected_orig, return_chain_ind=True
    )

    restored_canonical_form = canonical_form_from_pose_stack(
        pose_stack, chain_id_from_constr
    )

    # unpack
    (
        cf_orig_chain_id2,
        cf_orig_res_types2,
        cf_orig_coords2,
        cf_orig_at_is_pres2,
    ) = canonical_form
    (
        chain_id,
        cf_res_types,
        cf_coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = restored_canonical_form

    numpy.testing.assert_equal(
        cf_orig_chain_id.cpu().numpy(), cf_orig_chain_id2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_res_types.cpu().numpy(), cf_orig_res_types2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_coords.cpu().numpy(), cf_orig_coords2.cpu().numpy()
    )
    numpy.testing.assert_equal(
        cf_orig_at_is_pres.cpu().numpy(), cf_orig_at_is_pres2.cpu().numpy()
    )

    assert chain_id.device == torch_device
    assert cf_res_types.device == torch_device
    assert cf_coords.device == torch_device
    assert atom_is_present.device == torch_device
    assert res_not_connected.device == torch_device
    numpy.testing.assert_equal(chain_id.cpu().numpy(), cf_orig_chain_id.cpu().numpy())
    numpy.testing.assert_equal(
        cf_res_types.cpu().numpy(), cf_orig_res_types.cpu().numpy()
    )

    np_cf_coords = cf_coords.cpu().numpy()
    np_cf_orig_coords = cf_orig_coords.cpu().numpy()
    np_cf_orig_at_is_pres = cf_orig_at_is_pres.to(torch.bool).cpu().numpy()

    numpy.testing.assert_equal(
        np_cf_coords[np_cf_orig_at_is_pres], np_cf_orig_coords[np_cf_orig_at_is_pres]
    )

    gold_disulfides = torch.tensor(
        [
            (0, 22, 87),
            (0, 133, 193),
            (0, 213, 435),
            (0, 235, 309),
            (0, 359, 415),
            (0, 447, 466),
            (0, 475, 481),
            (0, 484, 488),
        ],
        dtype=torch.int64,
        device=torch_device,
    )

    numpy.testing.assert_equal(gold_disulfides.cpu().numpy(), disulfides.cpu().numpy())
    numpy.testing.assert_equal(
        res_not_connected.cpu().numpy(), res_not_connected_orig_np
    )


####
#     numpy.set_printoptions(threshold=100000)
#     # print("nan coords")
#     # print(numpy.isnan(np_cf_coords[np_cf_orig_at_is_pres]))
#
#     can_coord_is_nan = numpy.zeros((chain_id.shape[0], chain_id.shape[1], max_n_canonical_atoms), dtype=numpy.bool)
#     # print("numpy.isnan(np_cf_coords[np_cf_orig_at_is_pres])", numpy.isnan(np_cf_coords[np_cf_orig_at_is_pres]).shape)
#     can_coord_is_nan[np_cf_orig_at_is_pres] = numpy.any(numpy.isnan(np_cf_coords[np_cf_orig_at_is_pres]), axis=1)
#
#     for i in range(chain_id.shape[0]):
#         for j in range(chain_id.shape[1]):
#             can_rt = ordered_canonical_aa_types[cf_orig_res_types[i, j]]
#             can_rt_at_names = ordered_canonical_aa_atoms_v2[can_rt]
#             for k, at in enumerate(can_rt_at_names):
#                 if can_coord_is_nan[i, j, k]:
#                     print("nan for", i, j, k, can_rt, at, cf_orig_coords[i, j, k])
#
#
#
#     # print("coords"),
#     # print(np_cf_coords[np_cf_orig_at_is_pres] - np_cf_orig_coords[np_cf_orig_at_is_pres])
