import numpy

from typing import Tuple
from tmol.types import NDArray


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

    if n_all_bonds == 0:
        return numpy.empty((0, 3), dtype=int_dtype), numpy.full(
            (n_atoms, 2), -1, dtype=int_dtype
        )

    sources = numpy.concatenate((intra_res_bonds[:, 0], ordered_connection_atoms))
    if sources.max() >= n_atoms:
        raise IndexError("bond source atom index is out of bounds")
    counts = numpy.bincount(sources, minlength=n_atoms).astype(int_dtype, copy=False)
    ends = numpy.cumsum(counts, dtype=int_dtype)
    # Preserve the established range origin when the first atom is isolated.
    isolated_first = counts[0] == 0
    if isolated_first:
        ends -= 1
    bond_ranges = numpy.column_stack((ends - counts, ends))

    bonds = numpy.full((n_all_bonds, 3), -1, dtype=int_dtype)
    bonds[:, 0] = sources
    bonds[:n_intra_bonds, 1] = intra_res_bonds[:, 1]
    bonds[n_intra_bonds:, 2] = numpy.arange(n_inter_bonds, dtype=int_dtype)
    # Keep input order within an atom, with intra-block bonds before connections.
    bonds = bonds[numpy.argsort(sources, kind="stable")]
    if isolated_first:
        bonds = numpy.roll(bonds, -1, axis=0)

    return bonds, bond_ranges
