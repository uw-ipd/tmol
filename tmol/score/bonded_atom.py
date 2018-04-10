import properties

import torch
import numpy

import scipy.sparse

from tmol.properties.reactive import derived_from, cached
from tmol.properties.array import Array, VariableT

from .types import RealTensor


class BondedAtomScoreGraph(properties.HasProperties):
    @staticmethod
    def nan_to_num(var):
        return var.where(~numpy.isnan(var.detach()), RealTensor([0.0]))

    system_size = properties.Integer(
        "number of atoms in system", min=1, cast=True
    )

    coords = VariableT("source atomic coordinates")

    atom_types = Array("atomic types", dtype=object)[:]

    bonds = Array("inter-atomic bond indices", dtype=int, cast="unsafe")[:, 2]

    @derived_from("atom_types", VariableT("mask of 'real' atom indicies"))
    def real_atoms(self):
        return (
            torch.ByteTensor((self.atom_types != None).astype(numpy.ubyte))
        )  # noqa: E711 - None != is a vectorized check for None.

    @derived_from(
        "bonds",
        Array("inter-atomic minimum bonded path length", dtype="f4")[:, :]
    )
    def bonded_path_length(self):
        return scipy.sparse.csgraph.shortest_path(
            scipy.sparse.coo_matrix(
                (
                    numpy.ones(self.bonds.shape[0], dtype=bool),
                    (self.bonds[:, 0], self.bonds[:, 1])
                ),
                shape=(self.system_size, self.system_size),
            ),
            directed=False,
            unweighted=True
        ).astype("f4")

    score_components = properties.Set(
        "total score components",
        prop=properties.String("attribute name of scalar tensor property"),
        default=set(),
        observe_mutations=True
    )

    @cached(VariableT("sum of score_components"))
    def total_score(self):
        assert len(self.score_components) > 0
        return sum(
            getattr(self, component) for component in self.score_components
        )

    @properties.observer(properties.everything)
    def on_change(self, change):
        if change["name"] in self.score_components:
            self._set("total_score", properties.undefined)

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(
            dict(
                name="coords",
                prev=getattr(self, "coords"),
                mode="observe_set"
            )
        )

        self.total_score.backward()
        return self.total_score
