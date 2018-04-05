import cattr

import pandas

from toolz.dicttoolz import merge, valmap
import numpy
import numpy as np

import torch
import torch.autograd

from properties import Instance, Dictionary, StringChoice

from tmol.properties.reactive import derived_from, cached
from tmol.properties.array import Array, VariableT, TensorT

import tmol.genericnumeric as gn

from .interatomic_distance import InteratomicDistanceGraphBase
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
    A = np.arange(len(params)).reshape(1,-1)
    B = np.arange(len(params)).reshape(-1,1)
    lj_lk_pair_data = np.empty( np.broadcast(A,B).shape , dtype=pair_param_dtype )

    # lj
    # these are only dependent on atom1/atom2 ... can this be more efficient?
    lj_lk_pair_data["lj_rad1"] = params[A]["lj_radius"]
    lj_lk_pair_data["lj_rad2"] = params[B]["lj_radius"]

    sigma = params[A]["lj_radius"] + params[B]["lj_radius"]
    # exception 1: acc :: non-OH donor radii
    don_acc_pair_mask = ( \
        ( params[A]["is_donor"] & ~params[A]["is_hydroxyl"] & params[B]["is_acceptor"] ) |
        ( params[B]["is_donor"] & ~params[B]["is_hydroxyl"] & params[A]["is_acceptor"] ))
    sigma[ don_acc_pair_mask ] = global_params["lj_hbond_dis"] #lj_hbond_dis

    # exception 2: acc :: OH donor radii
    don_acc_pair_mask = ( \
        ( params[A]["is_donor"] & params[A]["is_hydroxyl"] & params[B]["is_acceptor"] ) |
        ( params[B]["is_donor"] & params[B]["is_hydroxyl"] & params[A]["is_acceptor"] ))
    sigma[ don_acc_pair_mask ] = global_params["lj_hbond_OH_donor_dis"] # lj_hbond_OH_donor_dis

    # exception 3: acc :: donor H radii
    don_acc_pair_mask = ( \
        ( params[A]["is_polarh"] & params[B]["is_acceptor"] ) |
        ( params[B]["is_polarh"] & params[A]["is_acceptor"] ))
    sigma[ don_acc_pair_mask ] = global_params["lj_hbond_hdis"] #lj_hbond_hdis

    # lj 
    sigma6  = sigma * sigma * sigma;
    sigma6  = sigma6 * sigma6;
    sigma12 = sigma6 * sigma6;
    wdepth = np.sqrt( params[A]["lj_wdepth"] * params[B]["lj_wdepth"] );

    lj_lk_pair_data["lj_sigma"] = sigma
    lj_lk_pair_data["lj_wdepth"] = wdepth
    lj_lk_pair_data["lj_coeff_sigma6"] = -2 * wdepth * sigma6
    lj_lk_pair_data["lj_coeff_sigma12"] = wdepth * sigma12

    # linear part
    ## (slope@switch_point*sigma/wdepth)
    lj_switch_value2wdepth = global_params["lj_switch_dis2sigma"] ** -12 - 2 * global_params["lj_switch_dis2sigma"] ** -6
    lj_switch_slope_sigma2wdepth = \
        -12.0 * global_params["lj_switch_dis2sigma"] ** -13 + \
        12.0 * global_params["lj_switch_dis2sigma"] ** -7

    lj_lk_pair_data["lj_switch_slope"] = wdepth/sigma * lj_switch_slope_sigma2wdepth
    lj_lk_pair_data["lj_switch_intercept"] = wdepth*lj_switch_value2wdepth - \
        lj_lk_pair_data["lj_switch_slope"]*sigma*global_params["lj_switch_dis2sigma"]

    lbx = global_params["spline_start"];
    ubx = global_params["max_dis"];
    lj_lk_pair_data["lj_spline_y0"] = \
        lj_lk_pair_data["lj_coeff_sigma12"] * (lbx ** -12) + lj_lk_pair_data["lj_coeff_sigma6"] * (lbx ** -6)
    lj_lk_pair_data["lj_spline_dy0"] = \
        -12 * lj_lk_pair_data["lj_coeff_sigma12"] * (lbx ** -13) - 6 * lj_lk_pair_data["lj_coeff_sigma6"] * (lbx ** -7)

    # lk
    inv_neg2_times_pi_sqrt_pi = -0.089793561062583294
    inv_lambda_1 = 1.0 / (params[A]["lk_lambda"])
    inv_lambda2_1 = inv_lambda_1*inv_lambda_1
    lj_lk_pair_data["lk_inv_lambda2_1"] = inv_lambda2_1
    lj_lk_pair_data["lk_coeff1"] = \
        inv_neg2_times_pi_sqrt_pi * \
        params[A]["lk_dgfree"] * \
        inv_lambda_1 * \
        params[B]["lk_volume"]

    inv_lambda_2 = 1.0 / (params[B]["lk_lambda"])
    inv_lambda2_2 = inv_lambda_2*inv_lambda_2
    lj_lk_pair_data["lk_inv_lambda2_2"] = inv_lambda2_2
    lj_lk_pair_data["lk_coeff2"] = \
        inv_neg2_times_pi_sqrt_pi * \
        params[B]["lk_dgfree"] * \
        inv_lambda_2* \
        params[A]["lk_volume"]

    thresh_dis = global_params["lj_switch_dis2sigma"]*sigma
    inv_thresh_dis2 = 1.0 / ( thresh_dis * thresh_dis )
    dis_rad1 = thresh_dis - lj_lk_pair_data["lj_rad1"]
    x_thresh1 = ( dis_rad1 * dis_rad1 ) * lj_lk_pair_data["lk_inv_lambda2_1"]
    dis_rad2 = thresh_dis - lj_lk_pair_data["lj_rad2"]
    x_thresh2 = ( dis_rad2 * dis_rad2 ) * lj_lk_pair_data["lk_inv_lambda2_2"]

    spline_close1_y0 = (gn.exp(-x_thresh1) * lj_lk_pair_data["lk_coeff1"] * inv_thresh_dis2);
    spline_close2_y0 = (gn.exp(-x_thresh2) * lj_lk_pair_data["lk_coeff2"] * inv_thresh_dis2);
    lj_lk_pair_data["lk_spline_close_y0"] = spline_close1_y0 + spline_close2_y0

    ##
    # near spline
    # fd: in code this is "rounded" to the nearest gridpoint
    switch = np.minimum( spline_close1_y0, spline_close2_y0)
    lj_lk_pair_data["lk_spline_close_x0"] = np.sqrt( np.maximum( switch*switch - 1.5 , 0.0 ) )
    lj_lk_pair_data["lk_spline_close_x1"] = np.sqrt( switch*switch + 1.0 )

    invdist_close = 1 / (lj_lk_pair_data["lk_spline_close_x1"])
    invdist2_close = invdist_close*invdist_close

    dis_rad_x1 = lj_lk_pair_data["lk_spline_close_x1"] - lj_lk_pair_data["lj_rad1"]
    x_x1 = ( dis_rad_x1 * dis_rad_x1 ) * lj_lk_pair_data["lk_inv_lambda2_1"]
    y_1 = gn.exp(-x_x1) * lj_lk_pair_data["lk_coeff1"] * invdist2_close
    dy_1 = -2 * ( dis_rad_x1 * lj_lk_pair_data["lk_inv_lambda2_1"] + invdist_close ) * y_1

    dis_rad_x2 = lj_lk_pair_data["lk_spline_close_x1"] - lj_lk_pair_data["lj_rad2"]
    x_x2 = ( dis_rad_x2 * dis_rad_x2 ) * lj_lk_pair_data["lk_inv_lambda2_2"]
    y_2 = gn.exp(-x_x2) * lj_lk_pair_data["lk_coeff2"] * invdist2_close
    dy_2 = -2 * ( dis_rad_x2 * lj_lk_pair_data["lk_inv_lambda2_2"] + invdist_close ) * y_2

    lj_lk_pair_data["lk_spline_close_y1"] = (y_1 + y_2)
    lj_lk_pair_data["lk_spline_close_dy1"] = (dy_1 + dy_2)

    ##
    # far spline
    invdist_far = 1 / (global_params["spline_start"])
    invdist2_far = invdist_far*invdist_far

    dis_rad_x3 = global_params["spline_start"] - lj_lk_pair_data["lj_rad1"]
    x_x3 = ( dis_rad_x3 * dis_rad_x3 ) * lj_lk_pair_data["lk_inv_lambda2_1"]
    y_3 = gn.exp(-x_x3) * lj_lk_pair_data["lk_coeff1"] * invdist2_far
    dy_3 = -2 * ( dis_rad_x3 * lj_lk_pair_data["lk_inv_lambda2_1"] + invdist_far ) * y_3

    dis_rad_x4 = global_params["spline_start"] - lj_lk_pair_data["lj_rad2"]
    x_x4 = ( dis_rad_x4 * dis_rad_x4 ) * lj_lk_pair_data["lk_inv_lambda2_2"]
    y_4 = gn.exp(-x_x4) * lj_lk_pair_data["lk_coeff2"] * invdist2_far
    dy_4 = -2 * ( dis_rad_x4 * lj_lk_pair_data["lk_inv_lambda2_2"] + invdist_far ) * y_4

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
    invdist6 = invdist2*invdist2*invdist2
    invdist12 = invdist6*invdist6

    # linear part
    shortrange_dcut = lj_switch_dis2sigma * lj_sigma
    shortrange_selector = (dist < shortrange_dcut)
    shortrange_component = dist * lj_switch_slope + lj_switch_intercept

    # analytic 12-6 part
    analytic_selector = (dist >= shortrange_dcut) & (dist < spline_start)
    analytic_component = (lj_coeff_sigma12 * invdist12) + (lj_coeff_sigma6 * invdist6)

    # lr spline fade part
    x0 = spline_start
    x1 = max_dis
    spline_fade_selector = ((dist >= x0) & (dist < x1))

    x = dist
    y0 = lj_spline_y0
    dy0 = lj_spline_dy0
    u0 = (3.0/(x1-x0))*((-y0)/(x1-x0) - dy0)
    u1 = (3.0/(x1-x0))*( y0/(x1-x0))
    spline_fade_component = ((x-x1)*((x-x0)*(u1*(x0-x) + u0*(x-x1)) + 3*y0)) / (3*(x0-x1))

    raw_lj = (
        (shortrange_component * shortrange_selector.type(RealTensor)) +
        (analytic_component * analytic_selector.type(RealTensor)) +
        (spline_fade_component * spline_fade_selector.type(RealTensor))
    )

    return torch.where(
        interaction_weight > 0,
        interaction_weight * raw_lj,
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
    flat_component    = lk_spline_close_y0

    # "near" spline part
    # we sum both spline coeffs together rather than summing splines
    near_spline_selector = (
        (dist >= lk_spline_close_x0) &
        (dist < lk_spline_close_x1)
    )
    x = dist
    x0 = lk_spline_close_x0
    x1 = lk_spline_close_x1
    y0 = lk_spline_close_y0
    y1 = lk_spline_close_y1
    dy1 = lk_spline_close_dy1
    u0 = (3.0/(x1-x0))*((y1-y0)/(x1-x0))
    u1 = (3.0/(x1-x0))*(dy1 - (y1-y0)/(x1-x0))
    near_spline_component = ((x-x1)*((x-x0)*(u1*(x0-x) + u0*(x-x1)) + 3*y0)) / (3*(x0-x1))

    # analytic LK part
    analytic_selector = (
        (dist >= lk_spline_close_x1) &
        (dist < spline_start)
    )

    dis1 = dist - lj_rad1
    dis2 = dist - lj_rad2
    x1 = dis1 * dis1 * lk_inv_lambda2_1
    x2 = dis2 * dis2 * lk_inv_lambda2_2
    analytic_component = invdist2 * (
        gn.exp(-x1) * lk_coeff1 +
        gn.exp(-x2) * lk_coeff2
    )

    # "far" spline  part
    # we sum both spline coeffs together rather than summing splines
    x0 = spline_start
    x1 = max_dis
    far_spline_selector = ((dist >= x0) & (dist < x1))
    x = dist
    y0 = lk_spline_far_y0
    dy0 = lk_spline_far_dy0
    u0 = (3.0/(x1-x0))*((-y0)/(x1-x0) - dy0)
    u1 = (3.0/(x1-x0))*( y0/(x1-x0))
    far_spline_component = ((x-x1)*((x-x0)*(u1*(x0-x) + u0*(x-x1)) + 3*y0)) / (3*(x0-x1))

    raw_lk = (
        (flat_component * flat_selector.type(RealTensor)) +
        (near_spline_component * near_spline_selector.type(RealTensor)) +
        (analytic_component * analytic_selector.type(RealTensor)) +
        (far_spline_component * far_spline_selector.type(RealTensor))
    )

    return torch.where(
        interaction_weight > 0,
        interaction_weight * raw_lk,
        torch.autograd.Variable(RealTensor([0.0]), requires_grad=False)
    )

class LJLKScoreGraph(InteratomicDistanceGraphBase):

    ljlk_database : LJLKDatabase = Instance(
        "ljlk parameter database",
        LJLKDatabase,
        default=tmol.database.default.scoring.ljlk
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_components.add("total_lk")
        self.score_components.add("total_lj")
        self.atom_pair_dist_thresholds.add(self.ljlk_database.global_parameters.max_dis)

    @derived_from(("ljlk_database"), Instance(
        "per-atom-type lk/lj score parameters, last entry (-1) is nan-filled for a 'dummy' value",
        pandas.DataFrame))
    def _type_ljlk_params(self):
        param_records = pandas.DataFrame.from_records(cattr.unstructure(
                self.ljlk_database.atom_type_parameters
            ))

        nan_record = numpy.full(1, numpy.nan, param_records.to_records(index=False).dtype)
        param_records.loc[len(param_records)] = pandas.DataFrame.from_records(nan_record).loc[0]

        return param_records

    @derived_from(("ljlk_database", "_type_ljlk_params",),
        Array("per-atom-pair-type lk/lj score parameters", dtype=pair_param_dtype)[:,:])
    def _type_pair_ljlk_params(self):
        return render_ljlk_pair_parameters(
            global_params = cattr.unstructure(self.ljlk_database.global_parameters),
            params = self._type_ljlk_params.to_records()
        )

    @derived_from(("_type_pair_ljlk_params"),
        Dictionary("pairwise lj/lk parameters",
            key_prop = StringChoice("param", pair_param_dtype.names),
            value_prop = TensorT("pairwise parameter tensor")
        ))
    def type_pair_ljlk_params(self):
        return {
            n : RealTensor(self._type_pair_ljlk_params[n])
            for n in self._type_pair_ljlk_params.dtype.names
        }

    @derived_from("_type_ljlk_params",
        Instance("Index of atom type indicies by atom type name", pandas.Series))
    def name_to_idx(self):
        return pandas.Series(
            data=numpy.arange(len(self._type_ljlk_params) - 1),
            index=self._type_ljlk_params["name"].values[:-1]
        )

    @derived_from(("ljlk_database", "atom_types"),
        VariableT("per-atom type indices in parameter arrays"))
    def _atom_type_indicies(self):
        # Lookup atom time indicies in the source atom properties table via index
        # Use "reindex" so "None" entries in the type array convert to 'nan' index,
        # then remask this into the -1 dummy index in the atom parameters array.
        idx = self.name_to_idx.reindex(self.atom_types)

        return torch.LongTensor(numpy.where(numpy.isnan(idx), [-1], idx))

    def _pair_param(self, param_name, ind_i=None, ind_j=None):
        if ind_i is None:
            ind_i = self.atom_pair_inds

        if ind_j is None:
            ind_i, ind_j = ind_i[0], ind_i[1]

        return (
            self.type_pair_ljlk_params
                [param_name]
                [self._atom_type_indicies[ind_i], self._atom_type_indicies[ind_j]]
        )

    def _pair_interaction_weight(self, ind_i, ind_j=None):
        if ind_j is None:
            ind_i, ind_j = ind_i[0], ind_i[1]

        return (
            self.ljlk_interaction_weight[ind_i, ind_j]
        )

    @derived_from(
        ("bonded_path_length", "atom_types", "real_atoms"),
        TensorT("lj&lk interaction weight, bonded cutoff")
    )
    def ljlk_interaction_weight(self):
        result = numpy.ones_like(self.bonded_path_length, dtype="f4")
        result[self.bonded_path_length < 4] = 0
        result[self.bonded_path_length == 4] = .2

        # TODO extract into abstract logic?
        result[self._atom_type_indicies.numpy() == -1,:] = 0
        result[:,self._atom_type_indicies.numpy() == -1] = 0

        return RealTensor(result)

    @derived_from(
        ("atom_pair_dist", "atom_types", "chemical_db", "bond_graph"),
        VariableT("inter-atomic lj score"))
    def lj(self):

        pair_parameters = {
            "dist" : self.atom_pair_dist,
            "interaction_weight": self._pair_interaction_weight(self.atom_pair_inds)
        }

        type_pair_parameters = { n : self._pair_param(n) for n in (
            "lj_sigma",
            "lj_switch_slope",
            "lj_switch_intercept",
            "lj_coeff_sigma12",
            "lj_coeff_sigma6",
            "lj_spline_y0",
            "lj_spline_dy0",
        )}

        global_parameters = { n : getattr(self.ljlk_database.global_parameters, n) for n in (
            "lj_switch_dis2sigma",
            "spline_start",
            "max_dis",
        )}

        return lj_score(**merge(
            pair_parameters,
            type_pair_parameters,
            global_parameters
        ))
        # split into atr & rep
        # atrE = np.copy(ljE);
        # selector3 = (dists < lj_lk_pair_params["lj_sigma"])
        # atrE[ selector3  ] = -lj_lk_pair_params["lj_wdepth"][ selector3 ]
        # repE = ljE - atrE

        # atrE *= lj_lk_pair_params["weights"]
        # repE *= lj_lk_pair_params["weights"]

    @derived_from(
        ("atom_pair_dist", "atom_types", "chemical_db", "bond_graph"),
        VariableT("inter-atomic lk score"))
    def lk(self):

        pair_parameters = {
            "dist" : self.atom_pair_dist,
            "interaction_weight": self._pair_interaction_weight(self.atom_pair_inds)
        }

        type_pair_parameters = { n : self._pair_param(n) for n in (
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
        )}

        global_parameters = { n : getattr(self.ljlk_database.global_parameters, n) for n in (
            "spline_start",
            "max_dis",
        )}

        return lk_score(**merge(
            pair_parameters,
            type_pair_parameters,
            global_parameters
        ))

    @derived_from("lj", VariableT("total inter-atomic lj"))
    def total_lj(self):
        return self.lj.sum()

    @derived_from("lk", VariableT("total inter-atomic lk"))
    def total_lk(self):
        return self.lk.sum()
