import torch
import numpy

from functools import partial
from tmol.io.canonical_ordering import (
    canonical_form_from_pdb,
    default_packed_block_types,
    default_canonical_ordering,
)
from tmol.io.details.left_justify_canonical_form import left_justify_canonical_form

from tmol.pose.pose_stack_builder import PoseStackBuilder


def get_add_two_fill_shape(x):
    if len(x.shape) >= 3:
        fill_shape = (x.shape[0], 2, *x.shape[2:])
    else:
        fill_shape = (x.shape[0], 2)
    return fill_shape


def add_two_res_at_gap(gap_pos, x, fill_value):
    if isinstance(x, numpy.ndarray):
        fill_shape = get_add_two_fill_shape(x)
        return numpy.concatenate(
            [
                x[:, :gap_pos],
                numpy.full(fill_shape, fill_value, dtype=x.dtype),
                x[:, gap_pos:],
            ],
            axis=1,
        )
    else:
        fill_shape = get_add_two_fill_shape(x)
        return torch.cat(
            [
                x[:, :gap_pos],
                torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device),
                x[:, gap_pos:],
            ],
            dim=1,
        )


def add_two_res_at_end(x, fill_value):
    if isinstance(x, numpy.ndarray):
        fill_shape = get_add_two_fill_shape(x)
        return numpy.concatenate(
            [
                x,
                numpy.full(fill_shape, fill_value, dtype=x.dtype),
            ],
            axis=1,
        )
    else:
        fill_shape = get_add_two_fill_shape(x)
        return torch.cat(
            [
                x,
                torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device),
            ],
            dim=1,
        )


def cf_as_tuple_from_pdb_lines(co, pdblines, device):
    cf = canonical_form_from_pdb(co, pdblines, device)
    return tuple([*cf])


def not_any_nancoord(coords):
    return torch.logical_not(torch.any(torch.isnan(coords), dim=3))


def test_assign_block_types_with_gaps(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    # take ten residues
    (
        ch_id_10,
        can_rts_10,
        coords_10,
        res_lab_10,
        res_ins_10,
        ch_lab_10,
        occ_10,
        bf_10,
        dslf_10,
        rnc_10,
    ) = cf_as_tuple_from_pdb_lines(co, ubq_pdb[: 81 * 167], torch_device)
    assert dslf_10 is None
    assert rnc_10 is None

    # put two empty residues in between res 5 and 6
    add_two_res = partial(add_two_res_at_gap, 5)

    ch_id = add_two_res(ch_id_10, 0)
    can_rts = add_two_res(can_rts_10, -1)
    coords = add_two_res(coords_10, float("nan"))
    res_lab = add_two_res(res_lab_10, 0)
    res_ins = add_two_res(res_ins_10, "")
    ch_lab = add_two_res(ch_lab_10, "")
    occ = add_two_res(occ_10, 1.0)
    bf = add_two_res(bf_10, 0.0)
    at_is_pres_10 = not_any_nancoord(coords_10)
    at_is_pres = not_any_nancoord(coords)

    ch_id, can_rts, coords, at_is_pres, _1, _2, res_lab, res_ins, ch_lab, occ, bf = (
        left_justify_canonical_form(
            ch_id,
            can_rts,
            coords,
            at_is_pres,
            disulfides=dslf_10,
            res_not_connected=rnc_10,
            res_labels=res_lab,
            res_ins_codes=res_ins,
            chain_labels=ch_lab,
            atom_occupancy=occ,
            atom_b_factor=bf,
        )
    )

    ch_id_lj_gold = add_two_res_at_end(ch_id_10, -1).cpu().numpy()
    can_rts_lj_gold = add_two_res_at_end(can_rts_10, -1).cpu().numpy()
    coords_lj_gold = add_two_res_at_end(coords_10, float("nan")).cpu().numpy()
    at_is_pres_lj_gold = add_two_res_at_end(at_is_pres_10, False).cpu().numpy()
    res_lab_lj_gold = add_two_res_at_end(res_lab_10, 0)
    res_ins_lj_gold = add_two_res_at_end(res_ins_10, "")
    ch_lab_lj_gold = add_two_res_at_end(ch_lab_10, "")
    occ_lj_gold = add_two_res_at_end(occ_10, 1.0)
    bf_lj_gold = add_two_res_at_end(bf_10, 0.0)

    numpy.testing.assert_equal(ch_id_lj_gold, ch_id.cpu().numpy())
    numpy.testing.assert_equal(can_rts_lj_gold, can_rts.cpu().numpy())
    numpy.testing.assert_equal(coords_lj_gold, coords.cpu().numpy())
    numpy.testing.assert_equal(at_is_pres_lj_gold, at_is_pres.cpu().numpy())
    numpy.testing.assert_equal(res_lab_lj_gold, res_lab)
    numpy.testing.assert_equal(res_ins_lj_gold, res_ins)
    numpy.testing.assert_equal(ch_lab_lj_gold, ch_lab)
    numpy.testing.assert_equal(occ_lj_gold, occ)
    numpy.testing.assert_equal(bf_lj_gold, bf)


def test_left_justify_can_form_with_gaps_in_dslf(pertuzumab_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    (
        orig_ch_id,
        orig_can_rts,
        orig_coords,
        orig_res_lab,
        orig_res_ins,
        orig_ch_lab,
        orig_occ,
        orig_bf,
        orig_dslf,
        orig_rnc,
    ) = cf_as_tuple_from_pdb_lines(co, pertuzumab_pdb, torch_device)

    # the actual disulfides
    disulfides = torch.tensor(
        [[0, 22, 87], [0, 213, 435], [0, 133, 193], [0, 235, 309], [0, 359, 415]],
        dtype=torch.int64,
        device=torch_device,
    )
    disulfides_shifted = torch.tensor(
        [[0, 22, 89], [0, 215, 437], [0, 135, 195], [0, 237, 311], [0, 361, 417]],
        dtype=torch.int64,
        device=torch_device,
    )

    # put two empty residues in between res 50 and 51
    add_two_res = partial(add_two_res_at_gap, 50)

    ch_id = add_two_res(orig_ch_id, 0)
    can_rts = add_two_res(orig_can_rts, -1)
    coords = add_two_res(orig_coords, float("nan"))
    ch_lab = add_two_res(orig_ch_lab, "")
    at_is_pres = not_any_nancoord(coords)  # add_two_res(orig_at_is_pres, 0)

    (
        lj_ch_id,
        lj_can_rts,
        lj_coords,
        lj_at_is_pres,
        lj_dslf,
        _2,
        _res_lab,
        _res_ins,
        ch_lab,
        _occ,
        _bf,
    ) = left_justify_canonical_form(
        ch_id, can_rts, coords, at_is_pres, disulfides_shifted, chain_labels=ch_lab
    )

    ch_id_lj_gold = add_two_res_at_end(orig_ch_id, -1).cpu().numpy()
    can_rts_lj_gold = add_two_res_at_end(orig_can_rts, -1).cpu().numpy()
    coords_lj_gold = add_two_res_at_end(orig_coords, float("nan")).cpu().numpy()
    orig_at_is_pres = not_any_nancoord(orig_coords)
    at_is_pres_lj_gold = add_two_res_at_end(orig_at_is_pres, 0).cpu().numpy()
    ch_lab_lj_gold = add_two_res_at_end(orig_ch_lab, "")

    numpy.testing.assert_equal(ch_id_lj_gold, lj_ch_id.cpu().numpy())
    numpy.testing.assert_equal(can_rts_lj_gold, lj_can_rts.cpu().numpy())
    numpy.testing.assert_equal(coords_lj_gold, lj_coords.cpu().numpy())
    numpy.testing.assert_equal(at_is_pres_lj_gold, lj_at_is_pres.cpu().numpy())
    numpy.testing.assert_equal(ch_lab_lj_gold, ch_lab)

    numpy.testing.assert_equal(disulfides.cpu().numpy(), lj_dslf.cpu().numpy())


def test_assign_block_types_for_pert_and_antigen(
    pertuzumab_and_nearby_erbb2_pdb_and_segments, torch_device
):
    co = default_canonical_ordering()
    (
        pert_and_erbb2_lines,
        res_not_connected,
    ) = pertuzumab_and_nearby_erbb2_pdb_and_segments
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    (
        orig_ch_id,
        orig_can_rts,
        orig_coords,
        orig_res_lab,
        orig_res_ins,
        orig_ch_lab,
        orig_occ,
        orig_bf,
        orig_dslf,
        orig_rnc,
    ) = cf_as_tuple_from_pdb_lines(co, pert_and_erbb2_lines, torch_device)

    orig_res_not_connected = torch.tensor(res_not_connected, device=torch_device)

    # put two empty residues in between res 50 and 51
    add_two_res = partial(add_two_res_at_gap, 50)

    ch_id = add_two_res(orig_ch_id, 0)
    can_rts = add_two_res(orig_can_rts, -1)
    coords = add_two_res(orig_coords, float("nan"))
    at_is_pres = not_any_nancoord(coords)  # add_two_res(orig_at_is_pres, 0)
    res_not_connected = add_two_res(orig_res_not_connected, False)
    ch_lab = add_two_res(orig_ch_lab, "")
    (
        lj_ch_id,
        lj_can_rts,
        lj_coords,
        lj_at_is_pres,
        _1,
        lj_res_not_connected,
        lj_res_lab,
        lj_res_ins,
        lj_ch_lab,
        lj_occ,
        lj_bf,
    ) = left_justify_canonical_form(
        ch_id, can_rts, coords, at_is_pres, None, res_not_connected, chain_labels=ch_lab
    )

    res_not_connected_lj_gold = (
        add_two_res_at_end(orig_res_not_connected, False).cpu().numpy()
    )
    numpy.testing.assert_equal(
        res_not_connected_lj_gold, lj_res_not_connected.cpu().numpy()
    )
