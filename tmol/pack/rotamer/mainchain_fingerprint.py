import numpy
import torch

from typing import List

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.numeric.dihedrals import coord_dihedrals
from tmol.database.chemical import ChemicalDatabase
from tmol.system.restypes import RefinedResidueType


@validate_args
def create_non_sidechain_fingerprint(
    rt: RefinedResidueType,
    parents: NDArray(numpy.int32)[:],
    sc_atoms: NDArray(numpy.int32)[:],
    chem_db: ChemicalDatabase,
):
    non_sc_atoms = numpy.nonzero(sc_atoms == 0)[0]
    print(non_sc_atoms)
    mc_at_names = rt.properties.polymer.mainchain_atoms
    mc_atoms = numpy.array(
        [rt.atom_to_idx[at] for at in mc_at_names], dtype=numpy.int32
    )
    mc_ind = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    mc_ind[mc_atoms] = numpy.arange(mc_atoms.shape[0], dtype=numpy.int32)

    # count the number of bonds for each atom
    n_bonds = numpy.zeros(rt.n_atoms, dtype=numpy.int32)
    for i in range(rt.bond_indices.shape[0]):
        n_bonds[rt.bond_indices[i, 0]] += 1
    for conn in rt.connection_to_idx:
        n_bonds[rt.connection_to_idx[conn]] += 1

    mc_ancestors = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    chiralities = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    non_sc_atom_fingerprints = []

    for nsc_at in non_sc_atoms:
        # find the index of the mc atom this branches from using the kintree
        mc_anc = mc_ind[nsc_at]
        bonds_from_mc = 0
        atom = nsc_at
        for i in range(rt.n_atoms):
            if mc_anc != -1:
                break
            par = parents[atom]
            mc_anc = mc_ind[par]
            atom = par
            bonds_from_mc += 1

        # now lets figure out the chirality of this atom??
        if bonds_from_mc == 0:
            chirality = 0
        elif bonds_from_mc == 1:
            # ok, let's figure out the number of bonds
            # that the mc atom has
            mc_n_bonds = n_bonds[mc_anc]
            if mc_n_bonds == 4:
                # now we need to measure the chirality of the atom
                # or, rather, whether this atom is on the "left"
                # or "right" of the chiral backbone atom.
                # Measure the improper dihedral given by the
                # mc atom and the two mc atoms it is bonded to.

                # ok, who is the first "lower" neighbor?
                prev_neighb = -1
                for i, at in enumerate(mc_atoms):
                    if at == mc_anc:
                        prev_neighb = i - 1
                        break
                if prev_neighb == -1:
                    mc1_icoor_ind = rt.icoors_index["down"]
                else:
                    mc1_icoor_ind = rt.at_to_icoor_ind[mc_atoms[prev_neighb]]

                # ok, who is the first "upper" neighbor?
                next_neighb = -1
                for i, at in enumerate(mc_atoms):
                    if at == mc_anc:
                        next_neighb = i + 1
                        break
                if next_neighb == -1 or next_neighb == mc_atoms.shape[0]:
                    mc2_icoor_ind = rt.icoors_index["up"]
                else:
                    mc2_icoor_ind = rt.at_to_icoor_ind[mc_atoms[next_neighb]]

                mc_anc_icoor_ind = rt.at_to_icoor_ind[mc_anc]

                def t64(coord):
                    return torch.tensor([coord], dtype=torch.float64)

                at1_coord = t64(rt.ideal_coords[mc1_icoor_ind])
                at2_coord = t64(rt.ideal_coords[mc_anc_icoor_ind])
                at3_coord = t64(rt.ideal_coords[mc2_icoor_ind])
                at4_coord = t64(rt.ideal_coords[rt.at_to_icoor_ind[nsc_at]])

                # now we have four coordinates, measure the dihedral
                dihe = numpy.degrees(
                    coord_dihedrals(at4_coord, at2_coord, at1_coord, at3_coord).numpy()[
                        0
                    ]
                )

                if dihe > 0:
                    chirality = 1
                else:
                    chirality = 2
            else:
                chirality = 0

        atom_type_name = rt.atoms[nsc_at].atom_type
        elem_name = next(
            at.element for at in chem_db.atom_types if at.name == atom_type_name
        )
        atomic_number = next(
            el.atomic_number for el in chem_db.element_types if el.name == elem_name
        )
        non_sc_atom_fingerprints.append(
            (mc_anc, bonds_from_mc, chirality, atomic_number)
        )
    return non_sc_atoms, non_sc_atom_fingerprints
