from functools import singledispatch

import torch
import numpy

import sparse
import scipy.sparse.csgraph as csgraph

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from .database import ParamDB
from .factory import Factory
from .stacked_system import StackedSystem


@reactive_attrs(auto_attribs=True)
class BondedAtomScoreGraph(StackedSystem, ParamDB, Factory):
    """Score graph component describing a system's atom types and bonds.

    Attributes:
        atom_types: [layer, atom_index] String atom type descriptors.
            Type descriptions defined in :py:mod:`tmol.database.chemical`.

        bonds:[layer, atom_index, atom_index] Inter-atomic bond indices.
            Note that bonds are strictly intra-layer, and are defined by a
            single layer index for both atoms of the bond.

        MAX_BONDED_PATH_LENGTH: Maximum relevant inter-atomic path length.
            Limits search depth used in ``bonded_path_length``, all longer
            paths reported as ``inf``.

    """

    MAX_BONDED_PATH_LENGTH = 6

    @staticmethod
    @singledispatch
    def factory_for(other, **_):
        """`clone`-factory, extract atom types and bonds from other."""
        return dict(atom_types=other.atom_types, bonds=other.bonds)

    atom_types: NDArray(object)[:, :]
    bonds: NDArray(int)[:, 3]

    @reactive_property
    @validate_args
    def real_atoms(atom_types: NDArray(object)[:, :],) -> Tensor(bool)[:, :]:
        """Mask of non-null atomic indices in the system."""
        return torch.ByteTensor(
            (atom_types != None).astype(numpy.ubyte)
        )  # noqa: E711 - None != is a vectorized check for None.

    @reactive_property
    @validate_args
    def bonded_path_length(
        bonds: NDArray(int)[:, 3],
        stack_depth: int,
        system_size: int,
        MAX_BONDED_PATH_LENGTH: int,
    ) -> NDArray("f4")[:, :, :]:
        """Dense inter-atomic bonded path length distance tables.

        Returns:
            [layer, from_atom, to_atom]
            Per-layer interatomic bonded path length entries.
        """

        bond_graph = sparse.COO(
            bonds.T,
            data=numpy.full(len(bonds), True),
            shape=(stack_depth, system_size, system_size),
            cache=True,
        )

        result = numpy.empty(bond_graph.shape, dtype="f4")
        for l in range(stack_depth):
            result[l] = csgraph.dijkstra(
                bond_graph[l].tocsr(),
                directed=False,
                unweighted=True,
                limit=MAX_BONDED_PATH_LENGTH,
            )

        return result
