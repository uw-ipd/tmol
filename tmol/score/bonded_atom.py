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
from .device import TorchDevice


@reactive_attrs(auto_attribs=True)
class BondedAtomScoreGraph(StackedSystem, ParamDB, TorchDevice, Factory):
    """Score graph component describing a system's atom types and bonds.

    Attributes:
        atom_types: [layer, atom_index] String atom type descriptors.
            Type descriptions defined in :py:mod:`tmol.database.chemical`.

        atom_elements: [layer, atom_index] String atom element.

        atom_names: [layer, atom_index] String residue-specific atom name.

        res_names: [layer, atom_index] String residue name descriptors.

        res_indices: [layer, atom_index] Integer residue index descriptors.

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
        return dict(
            atom_types=other.atom_types,
            atom_elements=other.atom_elements,
            atom_names=other.atom_names,
            res_names=other.res_names,
            res_indices=other.res_indices,
            bonds=other.bonds,
        )

    atom_types: NDArray(object)[:, :]
    atom_elements: NDArray(object)[:, :]
    atom_names: NDArray(object)[:, :]
    res_names: NDArray(object)[:, :]
    res_indices: NDArray(int)[:, :]
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
        device: torch.device,
        MAX_BONDED_PATH_LENGTH: int,
    ) -> Tensor("f4")[:, :, :]:
        """Dense inter-atomic bonded path length distance tables.

        Returns:
            [layer, from_atom, to_atom]
            Per-layer interatomic bonded path length entries.
        """

        return torch.from_numpy(
            bonded_path_length_stacked(
                bonds, stack_depth, system_size, MAX_BONDED_PATH_LENGTH
            )
        ).to(device)


def bonded_path_length(
    bonds: NDArray(int)[:, 2], system_size: int, limit: int
) -> NDArray("f4")[:, :]:
    bond_graph = sparse.COO(
        bonds.T,
        data=numpy.full(len(bonds), True),
        shape=(system_size, system_size),
        cache=True,
    )

    return csgraph.dijkstra(bond_graph, directed=False, unweighted=True, limit=limit)


def bonded_path_length_stacked(
    bonds: NDArray(int)[:, 3], stack_depth: int, system_size: int, limit: int
) -> NDArray("f4")[:, :, :]:
    bond_graph = sparse.COO(
        bonds.T,
        data=numpy.full(len(bonds), True),
        shape=(stack_depth, system_size, system_size),
        cache=True,
    )

    result = numpy.empty(bond_graph.shape, dtype="f4")
    for l in range(stack_depth):
        result[l] = csgraph.dijkstra(
            bond_graph[l].tocsr(), directed=False, unweighted=True, limit=limit
        )

    return result
