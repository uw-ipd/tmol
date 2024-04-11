import attr


import torch
import numpy

import toolz

import scipy.sparse.csgraph as csgraph
import scipy


from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IndexedBonds:
    # bonds = [ nstacks x nbonds x 2 ]
    #   with 2 = (atom1ind atom2ind)
    # bond_spans = [ nstacks x max-natoms x 2 ]
    #   with 2 = (bond start ind, bond end ind + 1)
    bonds: Tensor[int][:, :, 2]
    bond_spans: Tensor[int][:, :, 2]

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
