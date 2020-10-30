import attr
import numpy
import torch

from typing import Tuple, Mapping

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.numeric.dihedrals import coord_dihedrals
from tmol.database.chemical import ChemicalDatabase
from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes

from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms_jit


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomFingerprint:
    mc_ind: int
    mc_bond_dist: int
    chirality: int
    element: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprint:
    mc_ats: NDArray(numpy.int32)[:]
    mc_at_fingerprints: Tuple[AtomFingerprint, ...]
    fingerprint: Tuple[AtomFingerprint, ...]
    at_for_fingerprint: Mapping[AtomFingerprint, int]


@validate_args
def create_non_sidechain_fingerprint(
    rt: RefinedResidueType,
    parents: NDArray(numpy.int32)[:],
    sc_atoms: NDArray(numpy.int32)[:],
    chem_db: ChemicalDatabase,
):
    non_sc_atoms = numpy.nonzero(sc_atoms == 0)[0]
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
    at_for_fingerprint = {}

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
        at_fingerprint = AtomFingerprint(
            mc_ind=mc_anc,
            mc_bond_dist=bonds_from_mc,
            chirality=chirality,
            element=atomic_number,
        )

        non_sc_atom_fingerprints.append(at_fingerprint)
        at_for_fingerprint[at_fingerprint] = nsc_at
    return non_sc_atoms, tuple(non_sc_atom_fingerprints), at_for_fingerprint


@validate_args
def create_mainchain_fingerprint(
    rt: RefinedResidueType, sc_roots: Tuple[str, ...], chem_db: ChemicalDatabase
):
    id = rt.kintree_id
    parents = rt.kintree_parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    sc_roots = tuple(rt.atom_to_idx[at] for at in sc_roots)

    sidechain_atoms = bfs_sidechain_atoms_jit(
        parents, numpy.array(sc_roots, dtype=numpy.int32)
    )
    return create_non_sidechain_fingerprint(rt, parents, sidechain_atoms, chem_db)


def annotate_residue_type_with_sampler_fingerprints(
    restype: RefinedResidueType,
    samplers: Tuple[ChiSampler, ...],
    chem_db: ChemicalDatabase,
):
    for sampler in samplers:
        if sampler.defines_rotamers_for_rt(restype):
            if hasattr(restype, "mc_fingerprints"):
                if sampler.sampler_name() in restype.mc_fingerprints:
                    continue
            else:
                setattr(restype, "mc_fingerprints", {})

            sc_roots = sampler.first_sc_atoms_for_rt(restype)
            mc_ats, mc_at_fingerprints, at_for_fingerprint = create_mainchain_fingerprint(
                restype, sc_roots, chem_db
            )
            fingerprint = tuple(sorted(mc_at_fingerprints))
            restype.mc_fingerprints[sampler.sampler_name()] = MCFingerprint(
                mc_ats=mc_ats,
                mc_at_fingerprints=mc_at_fingerprints,
                fingerprint=fingerprint,
                at_for_fingerprint=at_for_fingerprint,
            )


def find_unique_fingerprints(pbt: PackedBlockTypes,):
    builder_types = set()
    for rt in pbt.active_residues:
        if hasattr(rt, "mc_fingerprints"):
            for builder in rt.mc_fingerprints:
                builder_types.add(builder)

    # builder_types = set([builder for rt in pbt.active_residues if hasattr(rt, "mc_fingerprints") for builder in rt.mc_fingerprints.keys()])

    fp_sets = set()
    for rt in pbt.active_residues:
        if hasattr(rt, "mc_fingerprints"):
            for builder, mcfps in rt.mc_fingerprints.items():
                fp_sets.add(mcfps.fingerprint)

    # we do not need to re-annotate this PackedBlockTypes object if there
    # are no sidechain builders that it has not encountered before
    if hasattr(pbt, "mc_atom_mapping"):
        all_found = True
        for bt in builder_types:
            if bt not in pbt.mc_atom_mapping:
                all_found = False
                break
        if all_found:
            return

    fp_sets = sorted(fp_sets)

    n_mcs = len(fp_sets)
    max_n_mc_atoms = max(len(fp) for fp in fp_sets)

    # ok, we need have n mainchain types
    # and we have m residue types
    # we have n x n different ways to map atoms from one mainchain
    # type onto another mainchain type

    # we will create an n x m x n-ats array that says which atom k
    # on mc-type i maps to which atom on residue type j

    mc_atom_inds_for_rt_for_builder = {}
    for builder in builder_types:
        mc_at_ind_for_rt = numpy.full(
            (n_mcs, pbt.n_types, max_n_mc_atoms), -1, dtype=numpy.int32
        )
        for i, fp in enumerate(fp_sets):
            for j, rt in enumerate(pbt.active_residues):
                # now we'er going to find the index of the mainchain atom
                if not hasattr(rt, "mc_fingerprints"):
                    continue
                if not builder in rt.mc_fingerprints:
                    continue
                rt_fingerprint = rt.mc_fingerprints[builder]
                for k, at_fp in enumerate(fp):
                    mc_at_ind_for_rt[i, j, k] = rt_fingerprint.at_for_fingerprint.get(
                        at_fp, -1
                    )
        mc_atom_inds_for_rt_for_builder[builder] = mc_at_ind_for_rt

    setattr(pbt, "mc_atom_mapping", mc_atom_inds_for_rt_for_builder)
