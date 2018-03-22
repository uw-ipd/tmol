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

from .ljlk import LJLKScoreGraph

class ScoreGraph(LJLKScoreGraph, properties.HasProperties):
    bond_graph = eq_by_is(
            properties.Instance(
                "inter-atomic bond graph",
                instance_class=scipy.sparse.spmatrix))
    coords = VariableT("source atomic coordinates")
    types = Array("atomic types", dtype=object)[:]

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

    @derived_from(("lj", "lk"), TensorT("inter-atomic total score"))
    def atom_scores(self):
        return torch.sum((self.lj + self.lk).data, dim=-1)

    @derived_from(("lj", "lk"), VariableT("system total score"))
    def total_score(self):
        return torch.sum(self.lj + self.lk)

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(dict(name="coords", prev=getattr(self, "coords"), mode="observe_set"))

        self.total_score.backward()
        return self.total_score

