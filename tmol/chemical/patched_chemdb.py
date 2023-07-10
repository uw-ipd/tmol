from typing import Tuple

from tmol.database.chemical import (
    Element,
    AtomType,
    RawResidueType,
    ChemicalDatabase,
    Icoor,
)

from tmol.extern.pysmiles.read_smiles import read_smiles

import attr
import copy
from enum import IntEnum

import networkx as nx
import networkx.algorithms.isomorphism as iso


# enum for validator code
class ResTypeValidatorErrorCodes(IntEnum):
    success = 0
    undefined_field = 1
    remove_nonreference_atom = 2
    modify_nonreference_atom = 3
    illegal_bond = 4
    illegal_icoor = 5
    illegal_torsion = 6
    illegal_connection = 7
    duplicate_atom_name = 8


# build a graph from a restype
class RestypeGraphBuilder:
    def __init__(self, atomtypedict):
        self.atomtypedict = atomtypedict

    def from_raw_res(self, r):
        mol = nx.Graph()
        for x in r.atoms:
            mol.add_node(x.name, element=self.atomtypedict[x.atom_type])
        for x in r.bonds:
            mol.add_edge(x[0], x[1])
        for x in r.connections:
            mol.add_node(x.name, element="{" + x.name + "}")
            mol.add_edge(x.name, x.atom)
        return mol


# remove all atom references from a raw restype
# delete atoms, bonds, torsions, and connections involving atom
def remove_atom(res, atom):
    res.atoms = tuple(x for x in res.atoms if x.name != atom)
    res.bonds = tuple((x, y) for x, y in res.bonds if x != atom and y != atom)
    res.torsions = tuple(
        x for x in res.torsions if atom not in [x.a.atom, x.b.atom, x.c.atom, x.d.atom]
    )
    res.torsions = tuple(
        x
        for x in res.torsions
        if atom not in [x.a.connection, x.b.connection, x.c.connection, x.d.connection]
    )
    res.connections = tuple(x for x in res.connections if x.name != atom)
    return res


def modify_atom(res, atom):
    res.atoms = tuple(x if x.name != atom.name else atom for x in res.atoms)
    return res


# update icoors corresponding to a patch definition
def update_icoor(res, patch, atoms_remove, namemap):
    new_icoor = []
    for target_i in patch:
        if target_i.source:
            source_name_i = namemap[target_i.source]
            source_idx = [i for i, x in enumerate(res) if x.name == source_name_i]
            source_i = res[source_idx[0]]
        else:
            source_i = Icoor(
                name=None,
                phi=None,
                theta=None,
                d=None,
                parent=None,
                grand_parent=None,
                great_grand_parent=None,
            )

        # map p, gp, ggp names
        p_i = source_i.parent if not target_i.parent else target_i.parent
        if p_i in namemap:
            p_i = namemap[p_i]
        gp_i = (
            source_i.grand_parent
            if not target_i.grand_parent
            else target_i.grand_parent
        )
        if gp_i in namemap:
            gp_i = namemap[gp_i]
        ggp_i = (
            source_i.great_grand_parent
            if not target_i.great_grand_parent
            else target_i.great_grand_parent
        )
        if ggp_i in namemap:
            ggp_i = namemap[ggp_i]

        icoor_i = Icoor(
            name=target_i.name,
            phi=source_i.phi if not target_i.phi else target_i.phi,
            theta=source_i.theta if not target_i.theta else target_i.theta,
            d=source_i.d if not target_i.d else target_i.d,
            parent=p_i,
            grand_parent=gp_i,
            great_grand_parent=ggp_i,
        )

        new_icoor.append(icoor_i)

    # create final icoor list
    remove = [namemap[i] for i in atoms_remove]
    icoor = (*(x for x in res if x.name not in remove), *new_icoor)

    return icoor


# get a list of atoms modified by a patch
def get_modified_atoms(patch):
    added, modded, deleted = [], [], []

    for i in patch.remove_atoms:
        deleted.append(i)

    for i in patch.add_atoms:
        added.append(i.name)
    for i in patch.modify_atoms:
        modded.append(i.name)
    for i in patch.add_connections:
        modded.append(i.atom)
        added.append(i.name)

    # modded finds all atoms whose CONNECTIVITY or COORDINATES have changed
    for i, j in patch.add_bonds:
        if i[0] == "<" and i[-1] == ">" and i not in modded:
            modded.append(i)
        if j[0] == "<" and j[-1] == ">" and j not in modded:
            modded.append(j)

    for i in patch.icoors:
        if i.name[0] == "<" and i.name[-1] == ">" and i.name not in modded:
            modded.append(i.name)

    return added, modded, deleted


# validate raw residues
def validate_raw_residue(res):
    allatoms = set([i.name for i in res.atoms])
    allconns = set([i.name for i in res.connections])

    # No duplicate atom names
    if len(allatoms) != len(res.atoms):
        return ResTypeValidatorErrorCodes.duplicate_atom_name

    # illegal bonds
    for i, j in res.bonds:
        if i not in allatoms or j not in allatoms:
            return ResTypeValidatorErrorCodes.illegal_bond

    # illegal torsions
    for i in res.torsions:
        for a_i in (i.a, i.b, i.c, i.d):
            if a_i.atom is None:
                if a_i.connection is None or a_i.connection not in allconns:
                    return ResTypeValidatorErrorCodes.illegal_torsion
            else:
                if a_i.atom not in allatoms:
                    return ResTypeValidatorErrorCodes.illegal_torsion

    # illegal connections
    for i in res.connections:
        if i.atom not in allatoms:
            return ResTypeValidatorErrorCodes.illegal_connection

    # illegal icoors
    for i in res.icoors:
        if i.name not in allatoms and i.name not in allconns:
            return ResTypeValidatorErrorCodes.illegal_icoor
        for a_i in (i.parent, i.grand_parent, i.great_grand_parent):
            if a_i not in allatoms and a_i not in allconns:
                return ResTypeValidatorErrorCodes.illegal_icoor

    return ResTypeValidatorErrorCodes.success


# validate patches
def validate_patch(patch):
    # make sure all fields are defined
    if (
        patch.name is None
        or patch.display_name is None
        or patch.pattern is None
        or patch.remove_atoms is None
        or patch.add_atoms is None
        or patch.add_connections is None
        or patch.add_bonds is None
        or patch.icoors is None
    ):
        return ResTypeValidatorErrorCodes.undefined_field

    # make sure all removed atoms are references
    for i in patch.remove_atoms:
        if i[0] != "<" or i[-1] != ">":
            return ResTypeValidatorErrorCodes.remove_nonreference_atom

    # make sure all modded atoms are references
    for i in patch.modify_atoms:
        if i.name[0] != "<" or i.name[-1] != ">":
            return ResTypeValidatorErrorCodes.modify_nonreference_atom

    # make sure all bonds are references or added atoms
    addedatoms = [i.name for i in patch.add_atoms]
    addedatoms.extend([i.name for i in patch.add_connections])
    for i, j in patch.add_bonds:
        if (i[0] != "<" or i[-1] != ">") and (i not in addedatoms):
            return ResTypeValidatorErrorCodes.illegal_bond

    # make sure all icoors are references or added atoms
    for i in patch.icoors:
        if (i.name[0] != "<" or i.name[-1] != ">") and (i.name not in addedatoms):
            return ResTypeValidatorErrorCodes.illegal_icoor

    return ResTypeValidatorErrorCodes.success


# apply a patch to a rawresidue
#    res, resgraph - base residue, graph
#    variant, patchgraph - patch variant, patch graph
#    marked - atoms modified in the base residue
# returns:
#    newreses - list of new residues produced by the patch (currently only support for 1)
#    newmarked - updated list of modified atoms in new residue
def do_patch(res, variant, resgraph, patchgraph, marked):
    atoms_match = (
        lambda x, y: ("element" not in y)
        or ("element" not in x)
        or x["element"] == y["element"]
    )
    gm = iso.GraphMatcher(resgraph, patchgraph, node_match=atoms_match)

    added, modded, deleted = get_modified_atoms(variant)
    assert len(modded) + len(deleted) > 0, (
        "Patch " + variant.name + " does not modify any atoms!"
    )

    # find patchsets that are unique w.r.t. list of modded atoms
    mod_unique = []
    namemaps = []
    for i, subgraph_x in enumerate(gm.subgraph_monomorphisms_iter()):
        namemap = {
            "<" + patchgraph.nodes[y]["name"] + ">": x for x, y in subgraph_x.items()
        }
        mod_i = [namemap[x] for x in modded]
        if mod_i not in mod_unique:
            mod_unique.append(mod_i)
            namemaps.append(namemap)

    # fd: this could be supported in the future, with a few issues to work out
    #    -if the patch adds atoms, we need to figure out how to ensure unique names
    #    -need a patch naming scheme if a patch applied twice
    assert len(mod_unique) <= 1, (
        "Patch " + variant.name + " applies to residue " + res.name + " multiple times!"
    )

    # apply patches
    newreses, newmarked = [], []
    for namemap in namemaps:
        newmark = copy.deepcopy(marked)
        modded = [namemaps[0][x] for x in modded]
        deleted = [namemaps[0][x] for x in deleted]

        # -1. Add atoms bonded to deleted atoms to modded set
        #     needs to be done after name map
        for i, j in res.bonds:
            if i in deleted and j not in deleted:
                modded.append(j)
            if j in deleted and i not in deleted:
                modded.append(i)

        # 0. check if we've already modified any of these atoms
        if set(modded) & set(newmark):
            continue

        newres = RawResidueType(
            name=res.name + ":" + variant.display_name,
            base_name=res.base_name,
            name3=res.base_name,
            atoms=res.atoms,
            bonds=res.bonds,
            connections=res.connections,
            torsions=res.torsions,
            icoors=res.icoors,
            properties=res.properties,
            chi_samples=res.chi_samples,
        )

        # 1. remove atoms
        for atom in variant.remove_atoms:
            atom = namemap[atom]
            newres = remove_atom(newres, atom)

        for atom in variant.modify_atoms:
            atom = attr.evolve(atom, name=namemap[atom.name])
            newres = modify_atom(newres, atom)

        # 2. add atoms
        newres.atoms = (*newres.atoms, *variant.add_atoms)

        # 3. add connections
        newconnections = []
        for i in variant.add_connections:
            if i.atom in namemap:
                i = attr.evolve(i, atom=namemap[i.atom])
            newconnections.append(i)
        newres.connections = (*newres.connections, *newconnections)

        # 4. add bonds
        newbonds = []
        for i, j in variant.add_bonds:
            if i in namemap:
                i = namemap[i]
            if j in namemap:
                j = namemap[j]
            newbonds.append((i, j))
        newres.bonds = (*newres.bonds, *newbonds)

        # 5. update icoors
        newres.icoors = update_icoor(
            newres.icoors, variant.icoors, variant.remove_atoms, namemap
        )

        # 6. update modified atoms
        # a) directly modified/added
        newmark.extend(modded)
        newmark.extend(added)

        # b) bonded to deleted atoms

        # c) removed atoms
        newmark = list(filter(lambda x: x not in deleted, newmark))

        newreses.append(newres)
        newmarked.append(newmark)

    return newreses, newmarked


# takes a ChemicalDatabase containing Tuple[RawResidueType] and Tuple[VariantType]
# applies all patches to all residues types
# returns PatchedChemicalDatabase containing only Tuple[RawResidueType]
@attr.s(auto_attribs=True)
class PatchedChemicalDatabase:
    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RawResidueType, ...]

    @classmethod
    def from_chem_db(cls, chemdb: ChemicalDatabase):
        G = RestypeGraphBuilder({x.name: x.element for x in chemdb.atom_types})

        for variant in chemdb.variants:
            val_id = validate_patch(variant)
            if val_id != ResTypeValidatorErrorCodes.success:
                assert False, (
                    "Bad patch: " + variant.name + "\nError code: " + str(val_id)
                )

        for res in chemdb.residues:
            val_id = validate_raw_residue(res)
            if val_id != ResTypeValidatorErrorCodes.success:
                assert False, (
                    "Bad raw residue: " + res.name + "\nError code: " + str(val_id)
                )

        patched_residues, patched_residues_names = [], []
        for res in chemdb.residues:
            done = False

            resvariants, resvariantnames, marked_atoms = [res], [""], [[]]
            while not done:
                done = True

                # newly added variants, variant names, marked atoms
                rv_new, rvn_new, ma_new = [], [], []
                for res_i, name_i, mark_i in zip(
                    resvariants, resvariantnames, marked_atoms
                ):
                    resgraph = G.from_raw_res(res_i)
                    for variant in chemdb.variants:
                        newtag = [*name_i, variant.name]
                        newtag.sort()
                        if newtag in resvariantnames or newtag in rvn_new:
                            continue  # we already made this variant

                        patchgraph = read_smiles(
                            variant.pattern,
                            explicit_hydrogen=True,
                            do_fill_valence=False,
                        )
                        patched_reses, mark_i_new = do_patch(
                            res_i, variant, resgraph, patchgraph, mark_i
                        )

                        if len(patched_reses) > 0:
                            rv_new.extend(patched_reses)
                            rvn_new.extend((newtag,) * len(patched_reses))
                            ma_new.extend(mark_i_new)
                            done = False  # added new residues

                resvariants.extend(rv_new)
                resvariantnames.extend(rvn_new)
                marked_atoms.extend(ma_new)
            patched_residues.extend(resvariants)
            patched_residues_names.extend(resvariantnames)

        for res in patched_residues:
            val_id = validate_raw_residue(res)
            if val_id != ResTypeValidatorErrorCodes.success:
                assert False, (
                    "Bad raw residue: " + res.name + "\nError code: " + str(val_id)
                )

        return cls(
            element_types=chemdb.element_types,
            atom_types=chemdb.atom_types,
            residues=patched_residues,
        )
