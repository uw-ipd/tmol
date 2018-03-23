import properties
import numpy

import torch
import torch.autograd


from properties import  Dictionary, StringChoice

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT
from tmol.database import ChemicalDatabase

import tmol.genericnumeric as gn

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

class LJLKScoreGraph(properties.HasProperties):
    chemical_db = properties.Instance("parameter database", ChemicalDatabase)

    @derived_from(("chemical_db"), Array(
        "per-atom-type lk/lj score parameters, last entry (-1) is nan-filled for a 'dummy' value",
        dtype=param_dtype)[:]
    )
    def type_ljlk_params(self):
        result = numpy.full(
            len(self.chemical_db.atom_properties.table) + 1,
            numpy.nan,
            dtype=param_dtype,
        )

        for n in result.dtype.names:
            result[n][:-1] = self.chemical_db.atom_properties.table[n]

        return result

    @derived_from(("chemical_db", "type_ljlk_params",),
        Array("per-atom-pair-type lk/lj score parameters", dtype=pair_param_dtype)[:,:])
    def type_pair_ljlk_params(self):

        a = self.type_ljlk_params.reshape((-1, 1))
        b = self.type_ljlk_params.reshape((1, -1))

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

    @derived_from(("chemical_db", "type_pair_ljlk_params", "atom_types"),
        Dictionary("pairwise lj/lk parameters",
            key_prop = StringChoice("param", pair_param_dtype.names),
            value_prop = TensorT("pairwise parameter tensor")
        ))
    def ljlk_pair_params(self):

        # Lookup atom time indicies in the source atom properties table via index
        # Use "reindex" so "None" entries in the type array convert to 'nan' index,
        # then remask this into the -1 dummy index in the atom parameters array.
        type_indices = numpy.where(
            self.real_atoms,
            self.chemical_db.atom_properties.name_to_idx.reindex(self.atom_types),
            [-1]
        ).astype(int)

        pair_parameters = self.type_pair_ljlk_params[
            type_indices.reshape((-1, 1)), 
            type_indices.reshape((1, -1))
        ]

        return {
            n : torch.Tensor(pair_parameters[n])
            for n in pair_parameters.dtype.names
        }

    @derived_from(("dist"), VariableT("1 / dist^2"))
    def invdist2(self):
        return 1 / (self.dist*self.dist)

    @derived_from(
        ("invdist2", "ljlk_pair_params", "ljlk_interaction_weight"),
        VariableT("inter-atomic lj score"))
    def lj(self):
        # lj
        # NOTE:
        #   - no sr or lr smoothing, no lr shift to 0
        #   - no split into atr/rep
        invdist2 = self.invdist2
        invdist6 = invdist2*invdist2*invdist2
        invdist12 = invdist6*invdist6

        lj = (self.ljlk_interaction_weight * (
                self.ljlk_pair_params["lj_r12_coeff"] * invdist12 +
                self.ljlk_pair_params["lj_r6_coeff"] * invdist6
        ))

        return torch.where(
            self.ljlk_interaction_weight > 0,
            lj,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )

    @derived_from(
        ("dist", "invdist2", "ljlk_pair_params", "ljlk_interaction_weight"),
        VariableT("inter-atomic lk score"))
    def lk(self):
        # lk -- for now, non-smoothed version
        dis1 = self.dist - self.ljlk_pair_params["lj_rad1"];
        dis2 = self.dist - self.ljlk_pair_params["lj_rad2"];

        x1 = dis1 * dis1 * self.ljlk_pair_params["lk_inv_lambda1"];
        x2 = dis2 * dis2 * self.ljlk_pair_params["lk_inv_lambda2"];

        lk = self.ljlk_interaction_weight * self.invdist2 * (
            gn.exp(-x1) * self.ljlk_pair_params["lk_coeff1"] +
            gn.exp(-x2) * self.ljlk_pair_params["lk_coeff2"]
        )

        return torch.where(
            self.ljlk_interaction_weight > 0,
            lk,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )
