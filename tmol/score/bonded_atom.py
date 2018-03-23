import properties

import torch
import numpy

import scipy.sparse

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT
from tmol.properties import eq_by_is

from tmol.database import ChemicalDatabase

class BondedAtomScoreGraph(properties.HasProperties):
    @staticmethod
    def nan_to_num(var):
        return var.where(
            ~numpy.isnan(var.detach()),
            torch.Tensor([0.0])
        )

    bond_graph = eq_by_is(properties.Instance(
        "inter-atomic bond graph",
        instance_class=scipy.sparse.spmatrix))

    coords = VariableT("source atomic coordinates")

    atom_types = Array("atomic types", dtype=object)[:]

    chemical_db = properties.Instance("parameter database", ChemicalDatabase)

    @derived_from("atom_types", VariableT("mask of 'real' atom indicies"))
    def real_atoms(self):
        return torch.ByteTensor((self.atom_types != None).astype(numpy.ubyte))

    @derived_from("bond_graph", Array("inter-atomic minimum bonded path length", dtype="f4")[:,:])
    def bonded_path_length(self):
        return scipy.sparse.csgraph.shortest_path(
            self.bond_graph,
            directed=False,
            unweighted=True
        ).astype("f4")

    @derived_from(("lj", "lk"), TensorT("inter-atomic total score"))
    def atom_scores(self):
        raise NotImplementedError()
        return torch.sum((self.lj + self.lk).data, dim=-1)

    @derived_from(("lj", "lk"), VariableT("system total score"))
    def total_score(self):
        return torch.sum(self.lj + self.lk)

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(dict(name="coords", prev=getattr(self, "coords"), mode="observe_set"))

        self.total_score.backward()
        return self.total_score
