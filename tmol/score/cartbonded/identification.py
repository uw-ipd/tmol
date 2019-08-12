import attr

from tmol.types.functional import convert_args
from tmol.types.array import NDArray
from tmol.database.scoring import CartBondedDatabase

from ..bonded_atom import IndexedBonds

import numpy

from numba import jit


# traverse bond graph, generate angles
@jit(nopython=True)
def find_angles(bonds, bond_spans):
    nstacks = bonds.shape[0]
    nbonds = bonds.shape[1]
    max_angles = int(nbonds * 3 / 2)  # assumes each node has at most 4 connections
    angles = numpy.full((nstacks, max_angles, 3), -9999, dtype=numpy.int64)

    # use atm1:atm3 ordering to ensure no duplication
    nangles_list = numpy.zeros((nstacks,), dtype=numpy.int64)
    for stack in range(nstacks):
        nangles = 0
        for i in range(nbonds):
            atm1, atm2 = bonds[stack, i]
            if atm1 < 0 or atm2 < 0:
                continue
            for j in range(bond_spans[stack, atm2, 0], bond_spans[stack, atm2, 1]):
                atm3 = bonds[stack, j, 1]
                if atm3 > atm1:
                    angles[stack, nangles, :] = [atm1, atm2, atm3]
                    nangles += 1
        nangles_list[stack] = nangles
    max_nangles = numpy.max(nangles_list)

    return angles[:, :max_nangles]


# traverse bond graph, generate torsions
@jit(nopython=True)
def find_torsions(bonds, bond_spans):
    nstacks = bonds.shape[0]
    nbonds = bonds.shape[1]
    max_torsions = int(nbonds * 9 / 2)  # assumes each node has at most 4 connections
    torsions = numpy.full((nstacks, max_torsions, 4), -9999, dtype=numpy.int64)

    # use atm0:atm3 ordering to ensure no duplication
    # note: cycles like A-B-C-A are valid!
    ntorsions_list = numpy.zeros((nstacks,), dtype=numpy.int64)
    for stack in range(nstacks):
        ntorsions = 0
        for i in range(nbonds):
            atm1, atm2 = bonds[stack, i]
            if atm1 < 0 or atm2 < 0:
                continue

            for j in range(bond_spans[stack, atm2, 0], bond_spans[stack, atm2, 1]):
                atm3 = bonds[stack, j, 1]
                if atm3 == atm1:
                    continue

                for k in range(bond_spans[stack, atm1, 0], bond_spans[stack, atm1, 1]):
                    atm0 = bonds[stack, k, 1]
                    if atm0 == atm2:
                        continue

                    if (atm0 < atm3) or (atm0 == atm3 and atm1 < atm2):
                        torsions[stack, ntorsions, :] = [atm0, atm1, atm2, atm3]
                        ntorsions += 1

        ntorsions_list[stack] = ntorsions
    max_ntorsions = numpy.max(ntorsions_list)

    return torsions[:, :max_ntorsions]


# traverse bond graph, generate improper torsions
#   ABCD combinations with bond pattern:
#         A
#          \
#           C-D
#          /
#         B
@jit(nopython=True)
def find_impropers(bonds, bond_spans):
    nstacks = bonds.shape[0]
    nbonds = bonds.shape[1]
    max_impropers = nbonds * 6  # assumes each node has at most 4 connections
    improper = numpy.full((nstacks, max_impropers, 4), -9999, dtype=numpy.int64)

    # note: ABCD and BACD are considered separate terms
    nimproper_list = numpy.zeros((nstacks,), dtype=numpy.int64)
    for stack in range(nstacks):
        nimproper = 0
        for i in range(nbonds):
            atm1, atm2 = bonds[stack, i]
            if atm1 < 0 or atm2 < 0:
                continue

            for j in range(bond_spans[stack, atm2, 0], bond_spans[stack, atm2, 1]):
                atm3 = bonds[stack, j, 1]
                if atm3 == atm1:
                    continue
                for k in range(bond_spans[stack, atm2, 0], bond_spans[stack, atm2, 1]):
                    atm4 = bonds[stack, k, 1]
                    if atm4 == atm1 or atm4 == atm3:
                        continue

                    improper[stack, nimproper, :] = [atm4, atm3, atm2, atm1]
                    nimproper += 1
        nimproper_list[stack] = nimproper
    max_nimproper = numpy.max(nimproper_list)

    return improper[:, :max_nimproper]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CartBondedIdentification:
    """Expands a bondgraph to export all bondangles and bondlengths
    """

    lengths: NDArray(int)[:, :, 2]
    angles: NDArray(int)[:, :, 3]
    torsions: NDArray(int)[:, :, 4]
    impropers: NDArray(int)[:, :, 4]

    @classmethod
    @convert_args
    def setup(
        cls, cartbonded_database: CartBondedDatabase, indexed_bonds: IndexedBonds
    ):
        bonds = indexed_bonds.bonds.cpu().numpy()
        spans = indexed_bonds.bond_spans.cpu().numpy()

        selected_bonds = [
            bonds[i, :, 0] < bonds[i, :, 1] for i in range(bonds.shape[0])
        ]
        max_bonds = max(numpy.sum(selected) for selected in selected_bonds)
        lengths = numpy.full((bonds.shape[0], max_bonds, 2), -9999, dtype=int)
        for i in range(bonds.shape[0]):
            ireal = numpy.sum(selected_bonds[i])
            lengths[i, :ireal, :] = bonds[i, selected_bonds[i], :]

        angles = find_angles(bonds, spans)
        torsions = find_torsions(bonds, spans)
        impropers = find_impropers(bonds, spans)

        return cls(
            lengths=lengths, angles=angles, torsions=torsions, impropers=impropers
        )
