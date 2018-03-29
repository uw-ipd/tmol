import numpy

import torch
import torch.autograd

import properties
from properties import  Dictionary, StringChoice

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT

import tmol.genericnumeric as gn

from .interatomic_distance import InteratomicDistanceGraphBase

param_dtype = numpy.dtype([
    ("lj_radius", numpy.float),
    ("lj_wdepth", numpy.float),
    ("lk_dgfree", numpy.float),
    ("lk_lambda", numpy.float),
    ("lk_volume", numpy.float),
    ("is_donor", numpy.bool),
    ("is_acceptor", numpy.bool),
    ("is_hydroxyl", numpy.bool),
    ("is_polarh", numpy.bool)
])

pair_param_dtype = numpy.dtype([
    ("lj_rad1", numpy.float),
    ("lj_rad2", numpy.float),
    ("lj_r6_coeff", numpy.float),
    ("lj_r12_coeff", numpy.float),
    ("lj_sigma", numpy.float),
    ("lk_coeff1", numpy.float),
    ("lk_coeff2", numpy.float),
    ("lk_inv_lambda1", numpy.float),
    ("lk_inv_lambda2", numpy.float)
])

class LJLKScoreGraph(InteratomicDistanceGraphBase):

    def __init__(self, **kwargs):
        self.score_components.add("total_lk")
        self.score_components.app("total_lj")
        super().__init__(**kwargs)

    @derived_from(("chemical_db"), Array(
        "per-atom-type lk/lj score parameters, last entry (-1) is nan-filled for a 'dummy' value",
        dtype=param_dtype)[:]
    )
    def _type_ljlk_params(self):
        result = numpy.full(
            len(self.chemical_db.atom_properties.table) + 1,
            numpy.nan,
            dtype=param_dtype,
        )

        for n in result.dtype.names:
            result[n][:-1] = self.chemical_db.atom_properties.table[n]

        return result

    @derived_from(("chemical_db", "_type_ljlk_params",),
        Array("per-atom-pair-type lk/lj score parameters", dtype=pair_param_dtype)[:,:])
    def _type_pair_ljlk_params(self):

        a = self._type_ljlk_params.reshape((-1, 1))
        b = self._type_ljlk_params.reshape((1, -1))

        # update derived parameters
        # could 1/2 this calculation
        type_pair_params = numpy.empty(numpy.broadcast(a, b).shape, dtype=pair_param_dtype )

        # lj
        # these are only dependent on atom1/atom2 ... can this be more efficient?
        type_pair_params["lj_rad1"] = a["lj_radius"]
        type_pair_params["lj_rad2"] = b["lj_radius"]

        sigma = a["lj_radius"] + b["lj_radius"]
        don_acc_pair_mask = ( \
            ( a["is_donor"] & b["is_acceptor"] ) |
            ( b["is_donor"] & a["is_acceptor"] ))
        sigma[ don_acc_pair_mask ] = 1.75 #lj_hbond_hdis
        don_acc_pair_mask = ( \
            ( a["is_donor"] & a["is_hydroxyl"] & b["is_acceptor"] ) |
            ( b["is_donor"] & b["is_hydroxyl"] & a["is_acceptor"] ))
        sigma[ don_acc_pair_mask ] = 2.6 #lj_hbond_OH_donor_dis

        # lj 
        sigma  = sigma * sigma * sigma;
        sigma6  = sigma * sigma;
        sigma12 = sigma6 * sigma6;
        wdepth = numpy.sqrt( a["lj_wdepth"] + b["lj_wdepth"] );

        type_pair_params["lj_sigma"] = sigma
        type_pair_params["lj_r6_coeff"] = -2 * wdepth * sigma6
        type_pair_params["lj_r12_coeff"] = wdepth * sigma12

        # lk
        inv_neg2_tms_pi_sqrt_pi = -0.089793561062583294
        inv_lambda1 = 1.0 / a["lk_lambda"]
        type_pair_params["lk_inv_lambda1"] = inv_lambda1
        type_pair_params["lk_coeff1"] = (
            inv_neg2_tms_pi_sqrt_pi * a["lk_lambda"] * inv_lambda1 * inv_lambda1 * b["lk_volume"]
        )

        inv_lambda2 = 1.0 / b["lk_lambda"]
        type_pair_params["lk_inv_lambda2"] = inv_lambda2
        type_pair_params["lk_coeff2"] = (
            inv_neg2_tms_pi_sqrt_pi * b["lk_lambda"] * inv_lambda2 * inv_lambda2 * a["lk_volume"] )

        return type_pair_params

    @derived_from(("chemical_db", "type_pair_ljlk_params", "atom_types"),
        Dictionary("pairwise lj/lk parameters",
            key_prop = StringChoice("param", pair_param_dtype.names),
            value_prop = TensorT("pairwise parameter tensor")
        ))
    def type_pair_ljlk_params(self):
        return {
            n : torch.Tensor(self._type_pair_ljlk_params[n])
            for n in self._type_pair_ljlk_params.dtype.names
        }

    @derived_from(("chemical_db", "atom_types"),
        TensorT("per-atom type indices in parameter arrays"))
    def _atom_type_indicies(self):
        # Lookup atom time indicies in the source atom properties table via index
        # Use "reindex" so "None" entries in the type array convert to 'nan' index,
        # then remask this into the -1 dummy index in the atom parameters array.
        return torch.LongTensor(numpy.where(
            self.real_atoms,
            self.chemical_db.atom_properties.name_to_idx.reindex(self.atom_types),
            [-1]
        ))

    def _pair_param(self, param_name, ind_i, ind_j=None):
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
        result[~self.real_atoms,:] = 0
        result[:,~self.real_atoms] = 0

        return torch.Tensor(result)

    @derived_from(("atom_pair_dist"), VariableT("1 / dist^2"))
    def atom_pair_invdist2(self):
        return 1 / (self.atom_pair_dist * self.atom_pair_dist)

    @derived_from(
        ("atom_pair_dist", "atom_types", "chemical_db", "bond_graph"),
        VariableT("inter-atomic lj score"))
    def lj(self):
        # lj
        # NOTE:
        #   - no sr or lr smoothing, no lr shift to 0
        #   - no split into atr/rep
        invdist2 = self.atom_pair_invdist2
        invdist6 = invdist2*invdist2*invdist2
        invdist12 = invdist6*invdist6

        interaction_weight = self._pair_interaction_weight(self.atom_pair_inds)
        lj_r12_coeff = self._pair_param("lj_r12_coeff", self.atom_pair_inds)
        lj_r6_coeff = self._pair_param("lj_r6_coeff", self.atom_pair_inds)

        lj = (interaction_weight * (
                lj_r12_coeff * invdist12 +
                lj_r6_coeff * invdist6
        ))

        return lj.where(
            interaction_weight > 0,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )

    @derived_from(
        ("atom_pair_dist", "atom_types", "chemical_db", "bond_graph"),
        VariableT("inter-atomic lk score"))
    def lk(self):
        # lk -- for now, non-smoothed version
        dis1 = self.atom_pair_dist - self._pair_param("lj_rad1", self.atom_pair_inds)
        dis2 = self.atom_pair_dist - self._pair_param("lj_rad2", self.atom_pair_inds)

        x1 = dis1 * dis1 * self._pair_param("lk_inv_lambda1", self.atom_pair_inds)
        x2 = dis2 * dis2 * self._pair_param("lk_inv_lambda2", self.atom_pair_inds)

        interaction_weight = self._pair_interaction_weight(self.atom_pair_inds)

        lk = interaction_weight * self.atom_pair_invdist2 * (
            gn.exp(-x1) * self._pair_param("lk_coeff1", self.atom_pair_inds) +
            gn.exp(-x2) * self._pair_param("lk_coeff2", self.atom_pair_inds)
        )

        return lk.where(
            interaction_weight > 0,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )


    @derived_from("lj", VariableT("total inter-atomic lj"))
    def total_lj(self):
        return self.lj.sum()

    @derived_from("lk", VariableT("total inter-atomic lk"))
    def total_lk(self):
        return self.lk.sum()
