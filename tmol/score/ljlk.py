import cattr

import pandas

from toolz.dicttoolz import merge
import numpy
import numpy as np

import torch

import tmol.utility.genericnumeric as gn

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from .interatomic_distance import InteratomicDistanceGraphBase
from .total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from .types import RealTensor

import tmol.database
from tmol.database.scoring import LJLKDatabase

pair_param_dtype = np.dtype([
    ("lj_rad1", np.float),
    ("lj_rad2", np.float),
    ("lj_sigma", np.float),
    ("lj_wdepth", np.float),
    ("lj_coeff_sigma6", np.float),
    ("lj_coeff_sigma12", np.float),
    ("lj_switch_intercept", np.float),
    ("lj_switch_slope", np.float),
    ("lj_spline_y0", np.float),
    ("lj_spline_dy0", np.float),
    ("lk_coeff1", np.float),
    ("lk_coeff2", np.float),
    ("lk_inv_lambda2_1", np.float),
    ("lk_inv_lambda2_2", np.float),
    ("lk_spline_close_x0", np.float),
    ("lk_spline_close_x1", np.float),
    ("lk_spline_close_y0", np.float),
    ("lk_spline_close_y1", np.float),
    ("lk_spline_close_dy1", np.float),
    ("lk_spline_far_y0", np.float),
    ("lk_spline_far_dy0", np.float),
])


def render_ljlk_pair_parameters(global_params, params):
    # update derived parameters
    # could 1/2 this calculation
    A = np.arange(len(params)).reshape(1, -1)
    B = np.arange(len(params)).reshape(-1, 1)
    lj_lk_pair_data = np.empty(
        np.broadcast(A, B).shape, dtype=pair_param_dtype
    )

    # lj
    # these are only dependent on atom1/atom2 ... can this be more efficient?
    lj_lk_pair_data["lj_rad1"] = params[A]["lj_radius"]
    lj_lk_pair_data["lj_rad2"] = params[B]["lj_radius"]

    sigma = params[A]["lj_radius"] + params[B]["lj_radius"]
    # exception 1: acc :: non-OH donor radii
    don_acc_pair_mask = (
        (params[A]["is_donor"] & ~params[A]["is_hydroxyl"] & params[B]["is_acceptor"]) |
        (params[B]["is_donor"] & ~params[B]["is_hydroxyl"] & params[A]["is_acceptor"])
    )  # yapf:disable
    sigma[don_acc_pair_mask] = global_params["lj_hbond_dis"]  # lj_hbond_dis

    # exception 2: acc :: OH donor radii
    don_acc_pair_mask = (
        (params[A]["is_donor"] & params[A]["is_hydroxyl"] & params[B]["is_acceptor"]) |
        (params[B]["is_donor"] & params[B]["is_hydroxyl"] & params[A]["is_acceptor"])
    )  # yapf: disable
    sigma[don_acc_pair_mask] = global_params["lj_hbond_OH_donor_dis"]

    # exception 3: acc :: donor H radii
    don_acc_pair_mask = (
        (params[A]["is_polarh"] & params[B]["is_acceptor"]) |
        (params[B]["is_polarh"] & params[A]["is_acceptor"])
    )  # yapf: disable
    sigma[don_acc_pair_mask] = global_params["lj_hbond_hdis"]  # lj_hbond_hdis

    # lj
    sigma6 = sigma * sigma * sigma
    sigma6 = sigma6 * sigma6
    sigma12 = sigma6 * sigma6
    wdepth = np.sqrt(params[A]["lj_wdepth"] * params[B]["lj_wdepth"])

    lj_lk_pair_data["lj_sigma"] = sigma
    lj_lk_pair_data["lj_wdepth"] = wdepth
    lj_lk_pair_data["lj_coeff_sigma6"] = -2 * wdepth * sigma6
    lj_lk_pair_data["lj_coeff_sigma12"] = wdepth * sigma12

    # linear part
    # (slope@switch_point*sigma/wdepth)
    lj_switch_value2wdepth = (
        global_params["lj_switch_dis2sigma"]**-12 -
        2 * global_params["lj_switch_dis2sigma"]**-6
    )
    lj_switch_slope_sigma2wdepth = (
        -12.0 * global_params["lj_switch_dis2sigma"]**-13 +
        12.0 * global_params["lj_switch_dis2sigma"]**-7
    )

    lj_lk_pair_data["lj_switch_slope"
                    ] = (wdepth / sigma * lj_switch_slope_sigma2wdepth)
    lj_lk_pair_data["lj_switch_intercept"] = wdepth * lj_switch_value2wdepth - \
        lj_lk_pair_data["lj_switch_slope"] * sigma * global_params["lj_switch_dis2sigma"]

    lbx = global_params["spline_start"]
    # ubx = global_params["max_dis"]
    lj_lk_pair_data["lj_spline_y0"] = (
        lj_lk_pair_data["lj_coeff_sigma12"] *
        (lbx**-12) + lj_lk_pair_data["lj_coeff_sigma6"] * (lbx**-6)
    )
    lj_lk_pair_data["lj_spline_dy0"] = (
        -12 * lj_lk_pair_data["lj_coeff_sigma12"] *
        (lbx**-13) - 6 * lj_lk_pair_data["lj_coeff_sigma6"] * (lbx**-7)
    )

    # lk
    inv_neg2_times_pi_sqrt_pi = -0.089793561062583294
    inv_lambda_1 = 1.0 / (params[A]["lk_lambda"])
    inv_lambda2_1 = inv_lambda_1 * inv_lambda_1
    lj_lk_pair_data["lk_inv_lambda2_1"] = inv_lambda2_1
    lj_lk_pair_data["lk_coeff1"] = (
        inv_neg2_times_pi_sqrt_pi *
        params[A]["lk_dgfree"] *
        inv_lambda_1 *
        params[B]["lk_volume"]
    )  # yapf: disable

    inv_lambda_2 = 1.0 / (params[B]["lk_lambda"])
    inv_lambda2_2 = inv_lambda_2 * inv_lambda_2
    lj_lk_pair_data["lk_inv_lambda2_2"] = inv_lambda2_2
    lj_lk_pair_data["lk_coeff2"] = (
        inv_neg2_times_pi_sqrt_pi *
        params[B]["lk_dgfree"] *
        inv_lambda_2 *
        params[A]["lk_volume"]
    )  # yapf: disable

    thresh_dis = global_params["lj_switch_dis2sigma"] * sigma
    inv_thresh_dis2 = 1.0 / (thresh_dis * thresh_dis)
    dis_rad1 = thresh_dis - lj_lk_pair_data["lj_rad1"]
    x_thresh1 = (dis_rad1 * dis_rad1) * lj_lk_pair_data["lk_inv_lambda2_1"]
    dis_rad2 = thresh_dis - lj_lk_pair_data["lj_rad2"]
    x_thresh2 = (dis_rad2 * dis_rad2) * lj_lk_pair_data["lk_inv_lambda2_2"]

    spline_close1_y0 = (
        gn.exp(-x_thresh1) * lj_lk_pair_data["lk_coeff1"] * inv_thresh_dis2
    )
    spline_close2_y0 = (
        gn.exp(-x_thresh2) * lj_lk_pair_data["lk_coeff2"] * inv_thresh_dis2
    )
    lj_lk_pair_data["lk_spline_close_y0"] = spline_close1_y0 + spline_close2_y0

    ##
    # near spline
    # fd: in code this is "rounded" to the nearest gridpoint
    switch = np.minimum(spline_close1_y0, spline_close2_y0)
    lj_lk_pair_data["lk_spline_close_x0"] = np.sqrt(
        np.maximum(switch * switch - 1.5, 0.0)
    )
    lj_lk_pair_data["lk_spline_close_x1"] = np.sqrt(switch * switch + 1.0)

    invdist_close = 1 / (lj_lk_pair_data["lk_spline_close_x1"])
    invdist2_close = invdist_close * invdist_close

    dis_rad_x1 = (
        lj_lk_pair_data["lk_spline_close_x1"] - lj_lk_pair_data["lj_rad1"]
    )
    x_x1 = (dis_rad_x1 * dis_rad_x1) * lj_lk_pair_data["lk_inv_lambda2_1"]
    y_1 = gn.exp(-x_x1) * lj_lk_pair_data["lk_coeff1"] * invdist2_close
    dy_1 = -2 * (
        dis_rad_x1 * lj_lk_pair_data["lk_inv_lambda2_1"] + invdist_close
    ) * y_1

    dis_rad_x2 = (
        lj_lk_pair_data["lk_spline_close_x1"] - lj_lk_pair_data["lj_rad2"]
    )
    x_x2 = (dis_rad_x2 * dis_rad_x2) * lj_lk_pair_data["lk_inv_lambda2_2"]
    y_2 = gn.exp(-x_x2) * lj_lk_pair_data["lk_coeff2"] * invdist2_close
    dy_2 = -2 * (
        dis_rad_x2 * lj_lk_pair_data["lk_inv_lambda2_2"] + invdist_close
    ) * y_2

    lj_lk_pair_data["lk_spline_close_y1"] = (y_1 + y_2)
    lj_lk_pair_data["lk_spline_close_dy1"] = (dy_1 + dy_2)

    ##
    # far spline
    invdist_far = 1 / (global_params["spline_start"])
    invdist2_far = invdist_far * invdist_far

    dis_rad_x3 = global_params["spline_start"] - lj_lk_pair_data["lj_rad1"]
    x_x3 = (dis_rad_x3 * dis_rad_x3) * lj_lk_pair_data["lk_inv_lambda2_1"]
    y_3 = gn.exp(-x_x3) * lj_lk_pair_data["lk_coeff1"] * invdist2_far
    dy_3 = -2 * (
        dis_rad_x3 * lj_lk_pair_data["lk_inv_lambda2_1"] + invdist_far
    ) * y_3

    dis_rad_x4 = global_params["spline_start"] - lj_lk_pair_data["lj_rad2"]
    x_x4 = (dis_rad_x4 * dis_rad_x4) * lj_lk_pair_data["lk_inv_lambda2_2"]
    y_4 = gn.exp(-x_x4) * lj_lk_pair_data["lk_coeff2"] * invdist2_far
    dy_4 = -2 * (
        dis_rad_x4 * lj_lk_pair_data["lk_inv_lambda2_2"] + invdist_far
    ) * y_4

    lj_lk_pair_data["lk_spline_far_y0"] = (y_3 + y_4)
    lj_lk_pair_data["lk_spline_far_dy0"] = (dy_3 + dy_4)

    return lj_lk_pair_data


def lj_score(
        # Pair conf/bond dependent inputs
        dist,
        interaction_weight,

        # Pair score parameters
        lj_sigma,
        lj_switch_slope,
        lj_switch_intercept,
        lj_coeff_sigma12,
        lj_coeff_sigma6,
        lj_spline_y0,
        lj_spline_dy0,

        # Global score parameters
        lj_switch_dis2sigma,
        spline_start,
        max_dis,
):
    invdist2 = 1 / (dist * dist)
    invdist6 = invdist2 * invdist2 * invdist2
    invdist12 = invdist6 * invdist6

    # linear part
    shortrange_dcut = lj_switch_dis2sigma * lj_sigma
    shortrange_selector = (dist < shortrange_dcut)
    shortrange_component = dist * lj_switch_slope + lj_switch_intercept

    # analytic 12-6 part
    analytic_selector = (dist >= shortrange_dcut) & (dist < spline_start)
    analytic_component = ((lj_coeff_sigma12 * invdist12) +
                          (lj_coeff_sigma6 * invdist6))

    # lr spline fade part
    x0 = spline_start
    x1 = max_dis
    spline_fade_selector = ((dist >= x0) & (dist < x1))

    x = dist
    y0 = lj_spline_y0
    dy0 = lj_spline_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    spline_fade_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    raw_lj = ((shortrange_component * shortrange_selector.type(RealTensor)) +
              (analytic_component * analytic_selector.type(RealTensor)) +
              (spline_fade_component * spline_fade_selector.type(RealTensor)))

    return torch.where(
        interaction_weight > 0, interaction_weight * raw_lj,
        torch.autograd.Variable(RealTensor([0.0]), requires_grad=False)
    )


def lk_score(
        # Pair conf/bond dependent inputs
        dist,
        interaction_weight,

        # Pair score parameters
        lj_rad1,
        lj_rad2,
        lk_coeff1,
        lk_coeff2,
        lk_inv_lambda2_1,
        lk_inv_lambda2_2,
        lk_spline_close_dy1,
        lk_spline_close_x0,
        lk_spline_close_x1,
        lk_spline_close_y0,
        lk_spline_close_y1,
        lk_spline_far_dy0,
        lk_spline_far_y0,

        # Global score parameters
        spline_start,
        max_dis,
):
    invdist2 = 1 / (dist * dist)

    flat_selector = (dist < lk_spline_close_x0)
    flat_component = lk_spline_close_y0

    # "near" spline part
    # we sum both spline coeffs together rather than summing splines
    near_spline_selector = ((dist >= lk_spline_close_x0) &
                            (dist < lk_spline_close_x1))
    x = dist
    x0 = lk_spline_close_x0
    x1 = lk_spline_close_x1
    y0 = lk_spline_close_y0
    y1 = lk_spline_close_y1
    dy1 = lk_spline_close_dy1
    u0 = (3.0 / (x1 - x0)) * ((y1 - y0) / (x1 - x0))
    u1 = (3.0 / (x1 - x0)) * (dy1 - (y1 - y0) / (x1 - x0))
    near_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    # analytic LK part
    analytic_selector = ((dist >= lk_spline_close_x1) & (dist < spline_start))

    dis1 = dist - lj_rad1
    dis2 = dist - lj_rad2
    x1 = dis1 * dis1 * lk_inv_lambda2_1
    x2 = dis2 * dis2 * lk_inv_lambda2_2
    analytic_component = invdist2 * (
        gn.exp(-x1) * lk_coeff1 + gn.exp(-x2) * lk_coeff2
    )

    # "far" spline  part
    # we sum both spline coeffs together rather than summing splines
    x0 = spline_start
    x1 = max_dis
    far_spline_selector = ((dist >= x0) & (dist < x1))
    x = dist
    y0 = lk_spline_far_y0
    dy0 = lk_spline_far_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    far_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    raw_lk = ((flat_component * flat_selector.type(RealTensor)) +
              (near_spline_component * near_spline_selector.type(RealTensor)) +
              (analytic_component * analytic_selector.type(RealTensor)) +
              (far_spline_component * far_spline_selector.type(RealTensor)))

    return torch.where(
        interaction_weight > 0, interaction_weight * raw_lk,
        torch.autograd.Variable(RealTensor([0.0]), requires_grad=False)
    )


@reactive_attrs(auto_attribs=True)
class LJLKScoreGraph(InteratomicDistanceGraphBase, TotalScoreComponentsGraph):
    ljlk_database: LJLKDatabase = tmol.database.default.scoring.ljlk

    @property
    def component_total_score_terms(self):
        return (
            ScoreComponentAttributes("lk", "total_lk", None),
            ScoreComponentAttributes("lj", "total_lj", None),
        )

    @property
    def component_atom_pair_dist_threshold(self):
        return self.ljlk_database.global_parameters.max_dis

    @reactive_property
    def _type_ljlk_params(ljlk_database) -> pandas.DataFrame:
        """per-atom-type lk/lj score parameters last entry (-1) is nan-filled for a 'dummy' value"""
        param_records = pandas.DataFrame.from_records(
            cattr.unstructure(ljlk_database.atom_type_parameters)
        )

        nan_record = numpy.full(
            1, numpy.nan, param_records.to_records(index=False).dtype
        )
        param_records.loc[len(param_records)
                          ] = pandas.DataFrame.from_records(nan_record).loc[0]

        return param_records

    @reactive_property
    def _type_pair_ljlk_params(
            ljlk_database,
            _type_ljlk_params,
    ) -> NDArray(pair_param_dtype)[:, :]:
        "per-atom-pair-type lk/lj score parameters",
        return render_ljlk_pair_parameters(
            global_params=cattr.unstructure(ljlk_database.global_parameters),
            params=_type_ljlk_params.to_records()
        )

    @reactive_property
    def type_pair_ljlk_params(_type_pair_ljlk_params) -> dict:
        "pairwise lj/lk parameters, dict[param -> Tensor])",
        return {
            n: RealTensor(_type_pair_ljlk_params[n])
            for n in _type_pair_ljlk_params.dtype.names
        }

    @reactive_property
    def name_to_idx(_type_ljlk_params) -> pandas.Series:
        """Index of atom type indicies by atom type name"""
        return pandas.Series(
            data=numpy.arange(len(_type_ljlk_params) - 1),
            index=_type_ljlk_params["name"].values[:-1]
        )

    @reactive_property
    def atom_parameter_indices(
            name_to_idx: pandas.Series,
            atom_types: NDArray(object)[:],
    ) -> Tensor(torch.long)[:]:
        """per-atom type indices in parameter arrays"""

        # Lookup atom time indicies in the source atom properties table via index
        # Use "reindex" so "None" entries in the type array convert to 'nan' index,
        # then remask this into the -1 dummy index in the atom parameters array.
        idx = name_to_idx.reindex(atom_types)

        return torch.LongTensor(numpy.where(numpy.isnan(idx), [-1], idx))

    @reactive_property
    def ljlk_interaction_weight(
            bonded_path_length, atom_types, atom_parameter_indices
    ) -> Tensor(torch.float)[:]:
        """lj&lk interaction weight, bonded cutoff"""

        result = numpy.ones_like(bonded_path_length, dtype="f4")
        result[bonded_path_length < 4] = 0
        result[bonded_path_length == 4] = .2

        # TODO extract into abstract logic?
        result[atom_parameter_indices.numpy() == -1, :] = 0
        result[:, atom_parameter_indices.numpy() == -1] = 0

        return RealTensor(result)

    @reactive_property
    @validate_args
    def lj(
            atom_pair_dist: Tensor(torch.float)[:],
            atom_pair_inds: Tensor(torch.long)[2, :],
            atom_parameter_indices: Tensor(torch.long)[:],
            ljlk_interaction_weight: Tensor(torch.float)[:, :],
            ljlk_database: tmol.database.scoring.LJLKDatabase,
            type_pair_ljlk_params: dict,
    ):

        pair_parameters = {
            "dist": atom_pair_dist,
            "interaction_weight":
                ljlk_interaction_weight[atom_pair_inds[0], atom_pair_inds[1]]
        }

        atom_pair_parameter_indicies = atom_parameter_indices[atom_pair_inds]
        pidx = (
            atom_pair_parameter_indicies[0],
            atom_pair_parameter_indicies[1],
        )

        type_pair_parameters = {
            n: type_pair_ljlk_params[n][pidx]
            for n in (
                "lj_sigma",
                "lj_switch_slope",
                "lj_switch_intercept",
                "lj_coeff_sigma12",
                "lj_coeff_sigma6",
                "lj_spline_y0",
                "lj_spline_dy0",
            )
        }

        global_parameters = {
            n: getattr(ljlk_database.global_parameters, n)
            for n in (
                "lj_switch_dis2sigma",
                "spline_start",
                "max_dis",
            )
        }

        return lj_score(
            **merge(pair_parameters, type_pair_parameters, global_parameters)
        )
        # split into atr & rep
        # atrE = np.copy(ljE);
        # selector3 = (dists < lj_lk_pair_params["lj_sigma"])
        # atrE[ selector3  ] = -lj_lk_pair_params["lj_wdepth"][ selector3 ]
        # repE = ljE - atrE

        # atrE *= lj_lk_pair_params["weights"]
        # repE *= lj_lk_pair_params["weights"]

    @reactive_property
    @validate_args
    def lk(
            atom_pair_dist: Tensor(torch.float)[:],
            atom_pair_inds: Tensor(torch.long)[2, :],
            atom_parameter_indices: Tensor(torch.long)[:],
            ljlk_interaction_weight: Tensor(torch.float)[:, :],
            ljlk_database: tmol.database.scoring.LJLKDatabase,
            type_pair_ljlk_params: dict,
    ):

        pair_parameters = {
            "dist": atom_pair_dist,
            "interaction_weight":
                ljlk_interaction_weight[atom_pair_inds[0], atom_pair_inds[1]]
        }

        atom_pair_parameter_indicies = atom_parameter_indices[atom_pair_inds]
        pidx = (
            atom_pair_parameter_indicies[0], atom_pair_parameter_indicies[1]
        )

        type_pair_parameters = {
            n: type_pair_ljlk_params[n][pidx[0], pidx[1]]
            for n in (
                "lj_rad1",
                "lj_rad2",
                "lk_coeff1",
                "lk_coeff2",
                "lk_inv_lambda2_1",
                "lk_inv_lambda2_2",
                "lk_spline_close_dy1",
                "lk_spline_close_x0",
                "lk_spline_close_x1",
                "lk_spline_close_y0",
                "lk_spline_close_y1",
                "lk_spline_far_dy0",
                "lk_spline_far_y0",
            )
        }

        global_parameters = {
            n: getattr(ljlk_database.global_parameters, n)
            for n in (
                "spline_start",
                "max_dis",
            )
        }

        return lk_score(
            **merge(pair_parameters, type_pair_parameters, global_parameters)
        )

    @reactive_property
    def total_lj(lj):
        """total inter-atomic lj"""
        return lj.sum()

    @reactive_property
    def total_lk(lk):
        """total inter-atomic lk"""
        return lk.sum()
