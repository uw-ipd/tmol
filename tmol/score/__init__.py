import properties

import torch

import numpy

import scipy.sparse
import scipy.sparse.csgraph

import torch
import torch.autograd

from properties import List, Dictionary, StringChoice
from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT
from tmol.properties import eq_by_is

from tmol.database import ChemicalDatabase
from tmol.database.chemical import lj_lk_pair_param_dtype

class ScoreGraph(properties.HasProperties):
    db = properties.Instance("parameter database", ChemicalDatabase)

    bond_graph = eq_by_is(
            properties.Instance(
                "inter-atomic bond graph",
                instance_class=scipy.sparse.spmatrix))
    coords = VariableT("source atomic coordinates")
    types = List("atomic types")

    @derived_from("bond_graph", Array("inter-atomic minimum bonded path length", dtype="f4")[:,:])
    def bonded_path_length(self):
        return scipy.sparse.csgraph.shortest_path(
            self.bond_graph,
            directed=False,
            unweighted=True
        ).astype("f4")

    @derived_from("coords", VariableT("inter-atomic pairwise distance"))
    def dist(self):
        deltas = self.coords.view((-1, 1, 3)) - self.coords.view((1, -1, 3))
        return torch.norm(deltas, 2, -1)

    @derived_from("bonded_path_length", TensorT("lj&lk interaction weight, bonded cutoff"))
    def lj_lk_interaction_weight(self):
        result = numpy.ones_like(self.bonded_path_length, dtype="f4")
        result[self.bonded_path_length < 4] = 0
        result[self.bonded_path_length == 4] = .2
        return torch.Tensor(result)

    @derived_from(("db", "types"),
        Dictionary("pairwise lj/lk parameters",
            key_prop = StringChoice("param", lj_lk_pair_param_dtype.names),
            value_prop = TensorT("pairwise parameter tensor")
        )
    )
    def lj_lk_pair_params(self):
        type_indices = self.db.atom_properties.name_to_idx[self.types].values

        pair_parameters = self.db.atom_properties.pairwise_lj_lk_params[
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
        ("invdist2", "lj_lk_pair_params", "lj_lk_interaction_weight"),
        VariableT("inter-atomic lj score"))
    def lj(self):
        # lj
        # NOTE:
        #   - no sr or lr smoothing, no lr shift to 0
        #   - no split into atr/rep
        invdist2 = self.invdist2
        invdist6 = invdist2*invdist2*invdist2
        invdist12 = invdist6*invdist6

        lj = (self.lj_lk_interaction_weight * (
                self.lj_lk_pair_params["lj_r12_coeff"] * invdist12 +
                self.lj_lk_pair_params["lj_r6_coeff"] * invdist6
        ))

        return torch.where(
            self.lj_lk_interaction_weight > 0,
            lj,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )

    @derived_from(
        ("dist", "invdist2", "lj_lk_pair_params", "lj_lk_interaction_weight"),
        VariableT("inter-atomic lk score"))
    def lk(self):
        # lk -- for now, non-smoothed version
        dis1 = self.dist - self.lj_lk_pair_params["lj_rad1"];
        dis2 = self.dist - self.lj_lk_pair_params["lj_rad2"];

        x1 = dis1 * dis1 * self.lj_lk_pair_params["lk_inv_lambda1"];
        x2 = dis2 * dis2 * self.lj_lk_pair_params["lk_inv_lambda2"];

        lk = self.lj_lk_interaction_weight * self.invdist2 * (
            torch.exp(-x1) * self.lj_lk_pair_params["lk_coeff1"] +
            torch.exp(-x2) * self.lj_lk_pair_params["lk_coeff2"]
        )

        return torch.where(
            self.lj_lk_interaction_weight > 0,
            lk,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )

    @derived_from("lj", TensorT("inter-atomic total score"))
    def atom_scores(self):
        return torch.sum((self.lj + self.lk).data, dim=-1)

    @derived_from("lj", VariableT("system total score"))
    def total_score(self):
        return torch.sum(self.lj + self.lk)

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(dict(name="coords", prev=getattr(self, "coords"), mode="observe_set"))

        self.total_score.backward()
        return self.total_score
