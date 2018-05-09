import properties

import torch
import numpy

import scipy.sparse

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT


class BondedAtomScoreGraph(properties.HasProperties):
    system_size = properties.Integer(
        "number of atoms in system", min=1, cast=True
    )
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
