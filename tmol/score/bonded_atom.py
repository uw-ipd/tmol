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
    # bonds = [ nstacks x nbonds x 2 ]
    #   with 2 = (atom1ind atom2ind)
    # bond_spans = [ nstacks x max-natoms x 2 ]
    #   with 2 = (bond start ind, bond end ind + 1)
    bonds: Tensor(int)[:, :, 2]
    bond_spans: Tensor(int)[:, :, 2]

    @classmethod
    def from_bonds(cls, src_bonds, minlength=None):

        # Convert undirected (stack, i, j) bond index tuples into sorted, indexed list.

        uniq_bonds, src_index = numpy.unique(src_bonds, axis=0, return_index=True)

        nstacks = numpy.max(uniq_bonds[:, 0]) + 1

        if not minlength:
            minlength = numpy.max(uniq_bonds[:, 1:]) + 1

        bond_counts_per_stack = numpy.bincount(
            uniq_bonds[:, 0], minlength=nstacks
        ).reshape((nstacks, 1))
        max_nbonds = numpy.max(bond_counts_per_stack)

        bonds = numpy.full((nstacks, max_nbonds, 2), -1, dtype=int)
        counts = numpy.arange(max_nbonds, dtype=int).reshape((1, max_nbonds))
        lowinds = counts < bond_counts_per_stack
        nz = numpy.nonzero(lowinds)
        bonds[nz[0], nz[1], :] = uniq_bonds[:, 1:]

        bond_spans = numpy.full((nstacks, minlength, 2), -1, dtype=int)

        # bond spans: the starting and stoping index for every bond for an atom;
        # created using the cumsum function after counting how many bonds each
        # atom has, using the bincount function.
        #
        # To use the bincount function, we need to ravel the 1st-atom-in-bond
        # array, and we need to assign atoms in separate stacks to diferent
        # integers (which is what max_atom_offsets is for). We also need to
        # set all of the negative bond inds to an out-of-range value (like
        # (maxats+1)*nstacks -1) and then, after performing the bincount,
        # slice off the part of the array that carries its counts.

        max_atom_offsets = numpy.arange(nstacks, dtype=int).reshape((nstacks, 1)) * (
            minlength + 1
        )
        first_at_shifted = numpy.copy(bonds[:, :, 0])
        first_at_shifted[first_at_shifted < 0] = (minlength + 1) * nstacks - 1
        first_at_shifted = (first_at_shifted + max_atom_offsets).ravel()
        bincounts = numpy.bincount(
            first_at_shifted, minlength=((minlength + 1) * nstacks)
        ).reshape(nstacks, -1)[:, :minlength]
        bonds_cumsum = numpy.cumsum(bincounts, axis=1)
        bond_spans[:, 0, 0] = 0
        bond_spans[:, 1:minlength, 0] = bonds_cumsum[:, :-1]
        bond_spans[:, 0:minlength, 1] = bonds_cumsum

        return cls(
            bonds=torch.from_numpy(bonds), bond_spans=torch.from_numpy(bond_spans)
        )

    @classmethod
    def to_directed(cls, src_bonds):
        """Convert a potentially-undirected bond-table into dense, directed bonds.
        The input "bonds" tensor is a two dimensional array of nbonds x 3,
        where the 2nd dimension holds [stack index, atom 1 index, atom 2 index].

        Eg. Converts
        [[0, 0, 1], [0, 0, 2]]
        into
        [[0, 0, 1], [0, 1, 0], [0, 0, 2], [0, 2, 0]]
        """

        return numpy.concatenate(
            (
                src_bonds,
                numpy.concatenate(
                    (src_bonds[:, 0:1], src_bonds[:, -1:], src_bonds[:, -2:-1]), axis=-1
                ),
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
        assert bonds.ndim == 2
        assert bonds.shape[1] == 3

        ## fd lkball needs this on the device
        ibonds = IndexedBonds.from_bonds(
            IndexedBonds.to_directed(bonds), minlength=system_size
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
