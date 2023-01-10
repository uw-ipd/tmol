import numpy
import numba


@numba.jit(nopython=True)
def bonds_and_bond_ranges(n_atoms, intra_res_bonds, ordered_connection_atoms):
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
