import properties

import torch

import numpy

import scipy.sparse
import scipy.sparse.csgraph

import torch
import torch.autograd

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT
from tmol.properties import eq_by_is

class ScoreGraph(properties.HasProperties):
    r_m = properties.Float("lj minima", default=1.0)
    epsilon = properties.Float("lj score epsilon", default=1.0)

    bond_graph = eq_by_is(
            properties.Instance(
                "inter-atomic bond graph",
                instance_class=scipy.sparse.spmatrix))
    coords = VariableT("source atomic coordinates")

    @derived_from("bond_graph", Array("inter-atomic minimum bonded path length", dtype="f4")[:,:])
    def bonded_path_length(self):
        return scipy.sparse.csgraph.shortest_path(
            self.bond_graph,
            directed=False,
            unweighted=True
        ).astype("f4")
    
    @derived_from("bonded_path_length", TensorT("lj-interaction weight, bonded cutoff"))
    def lj_interaction_weight(self):
        return torch.Tensor((self.bonded_path_length > 3).astype(float))
    
    @derived_from("coords", VariableT("inter-atomic pairwise distance"))
    def dist(self):
        
        deltas = self.coords.view((-1, 1, 3)) - self.coords.view((1, -1, 3))
        return torch.norm(deltas, 2, -1)
    
    @derived_from(("dist", "lj_interaction_weight"), VariableT("inter-atomic lj score"))
    def lj(self):
        fd = (self.r_m / self.dist)
        fd2 = fd * fd
        fd6 = fd2 * fd2 * fd2
        fd12 = fd6 * fd6

        lj = self.lj_interaction_weight * (self.epsilon * (fd12  - 3 * fd6))

        return torch.where(
            self.lj_interaction_weight > 0,
            lj,
            torch.autograd.Variable(torch.Tensor([0.0]), requires_grad=False)
        )

    @derived_from("lj", TensorT("inter-atomic total score"))
    def atom_scores(self):
        return torch.sum(self.lj.data, dim=-1)
        
    @derived_from("lj", VariableT("system total score"))
    def total_score(self):
        return torch.sum(self.lj)
    
    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(dict(name="coords", prev=getattr(self, "coords"), mode="observe_set"))

        self.total_score.backward()
        return self.total_score
