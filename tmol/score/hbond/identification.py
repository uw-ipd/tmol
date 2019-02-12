import torch
import attr

import numba

from tmol.types.functional import convert_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

import numpy

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring import HBondDatabase
from tmol.score.chemical_database import AcceptorHybridization, AtomTypeParamResolver


acceptor_dtype = numpy.dtype(
    [("a", int), ("b", int), ("b0", int), ("acceptor_type", object)]
)

donor_dtype = numpy.dtype([("d", int), ("h", int), ("donor_type", object)])


@numba.jit(nopython=True)
def sp2_acceptor_base(A, bonds, atom_bond_span, atom_is_hydrogen):
    """ Identify acceptor bases for sp2 acceptors.

        A = sp2 acceptor =>
            B = bonded to A, not H
            B0 = bonded to B, not A

        Returns:
            (A, B, B0) Index Tuple
    """
    B = -1
    B0 = -1

    for bidx in range(atom_bond_span[A][0], atom_bond_span[A][1]):
        other_atom = bonds[bidx][1]
        if not atom_is_hydrogen[other_atom]:
            B = other_atom
            break

    if B == -1:
        return (A, -1, -1)

    for bidx in range(atom_bond_span[B][0], atom_bond_span[B][1]):
        other_atom = bonds[bidx][1]
        if not other_atom == A:
            B0 = other_atom
            break

    if B0 == -1:
        return (A, -1, -1)

    return (A, B, B0)


@numba.jit(nopython=True)
def sp3_acceptor_base(A, bonds, atom_bond_span, atom_is_hydrogen):
    """ Identify acceptor bases for sp3 acceptors.

        A = sp3 acceptor =>
            B0 = bonded to A, is H
            B = bonded to A, not B0

        Returns:
            (A, B, B0) Index Tuple
    """
    B = -1
    B0 = -1

    for bidx in range(atom_bond_span[A][0], atom_bond_span[A][1]):
        other_atom = bonds[bidx][1]
        if atom_is_hydrogen[other_atom]:
            B0 = other_atom
            break

    if B0 == -1:
        return (A, -1, -1)

    for bidx in range(atom_bond_span[A][0], atom_bond_span[A][1]):
        other_atom = bonds[bidx][1]
        if not other_atom == B0:
            B = other_atom
            break

    if B == -1:
        return (A, -1, -1)

    return (A, B, B0)


@numba.jit(nopython=True)
def ring_acceptor_base(A, bonds, atom_bond_span, atom_is_hydrogen):
    """ Identify acceptor bases for ring acceptors.

        A = ring acceptor =>
            B = bonded to A, not H
            B0 = bonded to A, not H, not B

        Returns:
            (A, B, B0) Index Tuple
    """
    B = -1
    B0 = -1

    for bidx in range(atom_bond_span[A][0], atom_bond_span[A][1]):
        other_atom = bonds[bidx][1]
        if not atom_is_hydrogen[other_atom]:
            B = other_atom
            break

    if B == -1:
        return (A, -1, -1)

    for bidx in range(atom_bond_span[A][0], atom_bond_span[A][1]):
        other_atom = bonds[bidx][1]
        if not atom_is_hydrogen[other_atom] and other_atom != B:
            B0 = other_atom
            break

    if B0 == -1:
        return (A, -1, -1)

    return (A, B, B0)


@numba.jit
def id_acceptor_bases(
    A_idx, B_idx, B0_idx, A_hyb, bonds, atom_bond_span, atom_is_hydrogen
):
    """Given A_idx and bond graph metadata, calculate B, B0 idx.

    Params:
        A_idx: int[N] Populated Aatom index.
        B_idx: int[N] Empty B atom index.
        B0_idx: int[N] Empty B0 atom index.
        A_hyb: int[N] Acceptor hybridization class.
        bonds: int[nbond, 2] (i,j) sorted, symmetric bond tuples.
        atom_bond_span: int[natom, 2] [start,end) bond index i span.
        atom_is_hydrogen: bool[natom] Flag is atom is hydrogen.

    Populates A_idx, B_idx, B0_idx with identified atom indices, B, B0 = -1 if
    no valid acceptor bond pattern found.
    """

    for ai in range(A_idx.shape[0]):
        if A_hyb[ai] == AcceptorHybridization.sp2:
            A_idx[ai], B_idx[ai], B0_idx[ai] = sp2_acceptor_base(
                A_idx[ai], bonds, atom_bond_span, atom_is_hydrogen
            )
        elif A_hyb[ai] == AcceptorHybridization.sp3:
            A_idx[ai], B_idx[ai], B0_idx[ai] = sp3_acceptor_base(
                A_idx[ai], bonds, atom_bond_span, atom_is_hydrogen
            )
        elif A_hyb[ai] == AcceptorHybridization.ring:
            A_idx[ai], B_idx[ai], B0_idx[ai] = ring_acceptor_base(
                A_idx[ai], bonds, atom_bond_span, atom_is_hydrogen
            )
        else:
            A_idx[ai], B_idx[ai], B0_idx[ai] = A_idx[ai], -1, -1

    return A_idx, B_idx, B0_idx


@attr.s(frozen=True, slots=True, auto_attribs=True)
class HBondElementAnalysis(ValidateAttrs):
    donors: NDArray(donor_dtype)[:]
    acceptors: NDArray(acceptor_dtype)[:]

    @classmethod
    @convert_args
    def setup(
        cls,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:],
        atom_is_acceptor: NDArray(bool)[:],
        atom_acceptor_hybridization: NDArray(int)[:],
        atom_is_donor: NDArray(bool)[:],
        atom_is_hydrogen: NDArray(bool)[:],
        bonds: NDArray(int)[:, 2],
    ):
        # Sort (i,j) bond indicies in ascending order.
        bonds = bonds[numpy.lexsort((bonds[:, 1], bonds[:, 0]))]

        # Generate [start_idx, end_idx) spans for contiguous [(i, j_n)...]
        # blocks in the sorted bond table indexed by i
        num_bonds = numpy.cumsum(numpy.bincount(bonds[:, 0], minlength=len(atom_types)))

        atom_bond_span = numpy.empty((len(num_bonds), 2))
        atom_bond_span[0, 0] = 0
        atom_bond_span[1:, 0] = num_bonds[:-1]
        atom_bond_span[:, 1] = num_bonds

        # Filter donors/acceptors to those with type definitions in hbond
        # database, this logic should likely be moved to parameter resolution.
        atom_type_acceptor_type = {
            p.a: p.acceptor_type for p in hbond_database.acceptor_atom_types
        }
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        atom_acceptor_type = numpy.array(
            [atom_type_acceptor_type.get(at, None) for at in atom_types]
        )
        atom_donor_type = numpy.array(
            [atom_type_donor_type.get(at, None) for at in atom_types]
        )

        # Get the acceptor indicies and allocate base idx buffers
        A_idx = numpy.flatnonzero(
            atom_is_acceptor & atom_acceptor_type.astype(bool)  # None->False
        )
        B_idx = numpy.empty_like(A_idx)
        B0_idx = numpy.empty_like(A_idx)

        # Yeeeehaw
        id_acceptor_bases(
            A_idx,
            B_idx,
            B0_idx,
            atom_acceptor_hybridization[A_idx],
            bonds,
            atom_bond_span,
            atom_is_hydrogen,
        )

        assert not numpy.any(B_idx == -1), "Invalid acceptor atom type."

        acceptors = numpy.empty(A_idx.shape, dtype=acceptor_dtype)
        acceptors["a"] = A_idx
        acceptors["b"] = B_idx
        acceptors["b0"] = B0_idx
        acceptors["acceptor_type"] = atom_acceptor_type[acceptors["a"]]

        # Identify donor groups via donor-hydrogen bonds.
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        donor_pair_idx = bonds[
            atom_is_donor[bonds[:, 0]]
            & atom_donor_type.astype(bool)[bonds[:, 0]]  # None -> False
            & atom_is_hydrogen[bonds[:, 1]]
        ]

        donors = numpy.empty(donor_pair_idx.shape[0], dtype=donor_dtype)
        donors["d"] = donor_pair_idx[:, 0]
        donors["h"] = donor_pair_idx[:, 1]
        donors["donor_type"] = atom_donor_type[donors["d"]]

        return cls(donors=donors, acceptors=acceptors)

    @classmethod
    def setup_from_database(
        cls,
        chemical_database: ChemicalDatabase,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:],
        bonds: NDArray(int)[:, 2],
    ):

        atom_resolver = AtomTypeParamResolver.from_database(
            chemical_database, torch.device("cpu")
        )
        atom_type_idx = atom_resolver.type_idx(atom_types)
        atom_type_params = atom_resolver.params[atom_type_idx]

        return cls.setup(
            hbond_database=hbond_database,
            atom_types=atom_types,
            atom_is_acceptor=atom_type_params.is_acceptor.numpy(),
            atom_acceptor_hybridization=atom_type_params.acceptor_hybridization.numpy(),
            atom_is_donor=atom_type_params.is_donor.numpy(),
            atom_is_hydrogen=atom_type_params.is_hydrogen.numpy(),
            bonds=bonds,
        )
