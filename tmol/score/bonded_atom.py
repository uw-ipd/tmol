from functools import singledispatch

import torch
import numpy

import scipy.sparse

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class BondedAtomScoreGraph(Factory):
    @staticmethod
    @singledispatch
    def factory_for(other):
        """`clone`-factory, extract atom types and bonds from other."""
        return dict(
            atom_types=other.atom_types,
            bonds=other.bonds,
        )

    # String atom types
    atom_types: NDArray(object)[:]

    # Inter-atomic bond indices
    bonds: NDArray(int)[:, 2]

    @reactive_property
    def system_size(atom_types) -> int:
        """Number of atom locations within the system."""
        return len(atom_types)

    @reactive_property
    def real_atoms(atom_types: NDArray(object)[:], ) -> Tensor(bool)[:]:
        """Mask of 'real' atomic indices in the system."""
        return (torch.ByteTensor((atom_types != None).astype(numpy.ubyte))
                )  # noqa: E711 - None != is a vectorized check for None.

    @reactive_property
    def bonded_path_length(
            bonds: NDArray(int)[:, 2],
            system_size: int,
    ) -> NDArray("f4")[:, :]:
        """Dense inter-atomic bonded path length distance matrix."""
        return scipy.sparse.csgraph.shortest_path(
            scipy.sparse.coo_matrix(
                (
                    numpy.ones(bonds.shape[0], dtype=bool),
                    (bonds[:, 0], bonds[:, 1])
                ),
                shape=(system_size, system_size),
            ),
            directed=False,
            unweighted=True
        ).astype("f4")
