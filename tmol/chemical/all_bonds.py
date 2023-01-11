import numpy
import numba

from typing import Tuple
from tmol.types.array import NDArray


@numba.jit(nopython=True)
def bonds_and_bond_ranges(
    n_atoms: int,
    intra_res_bonds: NDArray[numpy.int32][:, 2],
    ordered_connection_atoms: NDArray[numpy.int32][:],
) -> Tuple[NDArray[numpy.int32][:, 3], NDArray[numpy.int32][:, 2]]:
    """Concatenate the set of intra- and inter-block bonds

    The intra-block bonds should list each bond twice, once for each of the
    two atoms it connects. The ordered-connection-atoms array lists the atom
    on this block that connects to another block such that the ith position
    in this array represents the ith inter-block connection; the cases when
    a single atom serves as a connection point to multiple other blocks
    (as might happen with metal ions) is handled correctly.

    This function returns a pair of arrays. The first array lists each bond
    as a tuple of (atom-ind, (other-atom)) where "other-atom" is similar to
    an unresolved-atom-id: it is itself a 2-tuple where if the indicated atom
    is a member of this block, then the first value in the tuple is a
    non-negative integer index of that atom, and if not, then the sentinel
    value of -1; if the atom is not a member of this block, then the second
    value in the tuple is the non-negative connection id, which can then
    be used to look up which other block, and which connection on that block
    is this block connected to in the PoseStack's data. The second array gives
    for each atom on this block the start and end indices of its bonds in
    the first array (all bonds in the first array for a single atom are
    contiguous).
    """

    int_dtype = intra_res_bonds.dtype
    n_intra_bonds = intra_res_bonds.shape[0]
    n_inter_bonds = ordered_connection_atoms.shape[0]
    n_all_bonds = n_intra_bonds + n_inter_bonds

    count_intra_bonds = numpy.zeros((n_atoms,), dtype=int_dtype)
    count_inter_bonds = numpy.zeros((n_atoms,), dtype=int_dtype)

    for i in range(n_intra_bonds):
        count_intra_bonds[intra_res_bonds[i, 0]] += 1
    for i in range(n_inter_bonds):
        count_inter_bonds[ordered_connection_atoms[i]] += 1

    bond_ranges = numpy.full((n_atoms, 2), -1, dtype=int_dtype)
    for i in range(n_atoms):
        i_n_bonds = count_intra_bonds[i] + count_inter_bonds[i]
        if i == 0:
            if i_n_bonds > 0:
                bond_ranges[i, 0] = 0
                bond_ranges[i, 1] = i_n_bonds
        else:
            bond_ranges[i, 0] = bond_ranges[i - 1, 1]
            bond_ranges[i, 1] = bond_ranges[i - 1, 1] + i_n_bonds

    bonds = numpy.zeros((n_all_bonds, 3), dtype=int_dtype)
    count_bond = numpy.zeros((n_atoms,), dtype=int_dtype)
    for i in range(n_intra_bonds):
        i_atm = intra_res_bonds[i, 0]
        i_count = count_bond[i_atm]
        count_bond[i_atm] += 1
        bond = bond_ranges[i_atm, 0] + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = intra_res_bonds[i, 1]
        bonds[bond, 2] = -1
    for i in range(n_inter_bonds):
        i_atm = ordered_connection_atoms[i]
        i_count = count_bond[i_atm]
        count_bond[i_atm] += 1
        bond = bond_ranges[i_atm, 0] + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = -1  # unresolved; on another residue
        bonds[bond, 2] = i  # i is the index of the inter-residue connection

    return bonds, bond_ranges
