import attr

from functools import singledispatch

import torch
import numpy

import toolz

import sparse
import scipy.sparse.csgraph as csgraph

from tmol.utility.reactive import reactive_property

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from .score_graph import score_graph
from .database import ParamDB
from .stacked_system import StackedSystem
from .device import TorchDevice


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IndexedBonds:
    bonds: Tensor(int)[:, 2]
    bond_spans: Tensor(int)[:, 2]
    src_index: Tensor(int)[:]

    @classmethod
    def from_bonds(cls, src_bonds, minlength=None):

        # Convert undirected (i, j) bond index tuples into sorted, indexed list.

        bonds, src_index = numpy.unique(src_bonds, axis=0, return_index=True)

        if not minlength:
            minlength = numpy.max(bonds)

        # Generate [start_idx, end_idx) spans for contiguous [(i, j_n)...]
        # blocks in the sorted bond table indexed by i
        num_bonds = numpy.cumsum(numpy.bincount(bonds[:, 0], minlength=minlength))

        bond_spans = numpy.empty((len(num_bonds), 2), dtype=int)
        bond_spans[0, 0] = 0
        bond_spans[1:, 0] = num_bonds[:-1]
        bond_spans[:, 1] = num_bonds

        return cls(
            bonds=torch.from_numpy(bonds),
            bond_spans=torch.from_numpy(bond_spans),
            src_index=torch.from_numpy(src_index),
        )

    @classmethod
    def to_directed(cls, src_bonds):
        """Convert a potentially-undirected bond-table into dense, directed bonds.

        Eg. Converts [[0, 1], [0, 2]] into [[0, 1], [1, 0], [0, 1], [2, 0]]
        """

        return numpy.concatenate(
            (
                src_bonds,
                numpy.concatenate((src_bonds[:, -1:], src_bonds[:, -2:-1]), axis=-1),
            ),
            axis=0,
        )

    def to(self, device: torch.device):
        return type(self)(
            **toolz.valmap(lambda t: t.to(device), attr.asdict(self, recurse=False))
        )


@score_graph
class BondedAtomScoreGraph(StackedSystem, ParamDB, TorchDevice):
    """Score graph component describing a system's atom types and bonds.

    Attributes:
        atom_types: [layer, atom_index] String atom type descriptors.
            Type descriptions defined in :py:mod:`tmol.database.chemical`.

        atom_names: [layer, atom_index] String residue-specific atom name.

        res_names: [layer, atom_index] String residue name descriptors.

        res_indices: [layer, atom_index] Integer residue index descriptors.

        bonds: [ind, (layer=0, atom_index=1, atom_index=2)] Inter-atomic bond indices.
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
            atom_names=other.atom_names,
            res_names=other.res_names,
            res_indices=other.res_indices,
            bonds=other.bonds,
        )

    atom_types: NDArray(object)[:, :]
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
    def indexed_bonds(bonds, system_size, device):
        """Sorted, constant time access to bond graph."""
        assert numpy.all(bonds[:, 0] == 0)
        assert bonds.ndim == 2
        assert bonds.shape[1] == 3

        ## fd lkball needs this on the device
        ibonds = IndexedBonds.from_bonds(
            IndexedBonds.to_directed(bonds[..., 1:]), minlength=system_size
        ).to(device)

        return ibonds

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
    print("bonds!", bonds.shape)
    print(bonds)
    bond_graph = sparse.COO(
        bonds.T,
        data=numpy.full(len(bonds), True),
        shape=(stack_depth, system_size, system_size),
        cache=True,
    )

    result = numpy.empty(bond_graph.shape, dtype="f4")
    for l in range(stack_depth):
        print("bond_graph", bond_graph[l])
        result[l] = csgraph.dijkstra(
            bond_graph[l].tocsr(), directed=False, unweighted=True, limit=limit
        )

        print("csgraph:", l, result[l, :10, :10])

    return result
