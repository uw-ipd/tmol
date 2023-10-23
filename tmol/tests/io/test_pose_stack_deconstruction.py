import torch
import numpy

from tmol.io.pose_stack_deconstruction import (
    canonical_form_from_pose_stack,
)
from tmol.io.canonical_ordering import (
    canonical_form_from_pdb_lines,
    max_n_canonical_atoms,
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


def test_canonical_form_from_jagged_ubq_pose(ubq_pdb, torch_device):
    ubq_lines4 = ubq_pdb[: 81 * 75]
    ubq_lines6 = ubq_pdb[: 81 * 113]

    canonical_form4 = canonical_form_from_pdb_lines(ubq_lines4, torch_device)
    # unpack
    (
        cf_orig_chain_id4,
        cf_orig_res_types4,
        cf_orig_coords4,
        cf_orig_at_is_pres4,
    ) = tuple(x.clone() for x in canonical_form4)

    canonical_form6 = canonical_form_from_pdb_lines(ubq_lines6, torch_device)
    # unpack
    (
        cf_orig_chain_id6,
        cf_orig_res_types6,
        cf_orig_coords6,
        cf_orig_at_is_pres6,
    ) = tuple(x.clone() for x in canonical_form6)

    orig_chain_id = torch.full((2, 6), -1, dtype=torch.int32, device=torch_device)
    orig_res_types = torch.full((2, 6), -1, dtype=torch.int32, device=torch_device)
    orig_coords = torch.full(
        (2, 6, max_n_canonical_atoms, 3), numpy.nan, device=torch_device
    )
    orig_at_is_pres = torch.full(
        (2, 6, max_n_canonical_atoms), 0, dtype=torch.int32, device=torch_device
    )
    orig_res_not_connected = torch.zeros(
        (2, 6, 2), dtype=torch.bool, device=torch_device
    )

    orig_chain_id[0, :4] = cf_orig_chain_id4
    orig_chain_id[1, :6] = cf_orig_chain_id6
    orig_res_types[0, :4] = cf_orig_res_types4
    orig_res_types[1, :6] = cf_orig_res_types6
    orig_coords[0, :4] = cf_orig_coords4
    orig_coords[1, :6] = cf_orig_coords6
    orig_at_is_pres[0, :4] = cf_orig_at_is_pres4
    orig_at_is_pres[1, :6] = cf_orig_at_is_pres6
    # orig_res_not_connected[0, 0, 0] = True
    # orig_res_not_connected[1, 0, 0] = True
    orig_res_not_connected[0, 3, 1] = True  # don't add OXT
    orig_res_not_connected[1, 5, 1] = True  # don't add OXT

    pose_stack = pose_stack_from_canonical_form(
        orig_chain_id,
        orig_res_types,
        orig_coords,
        orig_at_is_pres,
        res_not_connected=orig_res_not_connected,
    )

    restored_canonical_form = canonical_form_from_pose_stack(pose_stack)

    (
        chain_id,
        cf_res_types,
        cf_coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = restored_canonical_form

    assert chain_id.device == torch_device
    assert cf_res_types.device == torch_device
    assert cf_coords.device == torch_device
    assert atom_is_present.device == torch_device
    assert res_not_connected.device == torch_device
    numpy.testing.assert_equal(chain_id.cpu().numpy(), orig_chain_id.cpu().numpy())
    numpy.testing.assert_equal(cf_res_types.cpu().numpy(), orig_res_types.cpu().numpy())

    numpy.testing.assert_equal(cf_coords.cpu().numpy(), orig_coords.cpu().numpy())
    numpy.testing.assert_equal(
        atom_is_present.cpu().numpy(), orig_at_is_pres.cpu().numpy()
    )
    gold_disulfides = numpy.empty((0, 3), dtype=numpy.int64)
    numpy.testing.assert_equal(gold_disulfides, disulfides.cpu().numpy())

    numpy.testing.assert_equal(
        res_not_connected.cpu().numpy(), orig_res_not_connected.cpu().numpy()
    )


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
