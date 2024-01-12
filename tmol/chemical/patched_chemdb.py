from typing import Tuple
from collections import defaultdict

from tmol.database.chemical import (
    Element,
    AtomType,
    RawResidueType,
    VariantType,
    ChemicalDatabase,
    Icoor,
)

from tmol.extern.pysmiles.read_smiles import read_smiles

import attr
import copy

import networkx as nx
import networkx.algorithms.isomorphism as iso


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

    _validate_raw_residue_atoms(res, allatoms)
    _validate_raw_residue_bonds(res, allatoms)
    _validate_raw_residue_torsions(res, allatoms, allconns)
    _validate_raw_residue_connections(res, allatoms)
    _validate_raw_residue_icoors(res, allatoms, allconns)


def _validate_raw_residue_atoms(res, allatoms):
    # No duplicate atom names
    if len(allatoms) != len(res.atoms):
        duplicated = defaultdict(int)
        for i, at in enumerate(res.atoms):
            if at.name in duplicated:
                continue
            for j, at2 in enumerate(res.atoms[(i + 1) :]):
                if at.name == at2.name:
                    duplicated[at.name] += 1

        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: duplicated_atom_name; atoms may appear only once\n",
                "Offending atoms:\n",
                "\n".join(
                    [f'    "{x}" appears {duplicated[x]+1} times' for x in duplicated]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_bonds(res, allatoms):
    # illegal bonds
    bad_bonds = []
    for i, j in res.bonds:
        if i not in allatoms:
            bad_bonds.append((i, i, j))
        if j not in allatoms:
            bad_bonds.append((j, i, j))
    if len(bad_bonds) > 0:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_bond; must be between declared atoms\n"
                "Offending atoms:\n",
                "\n".join(
                    [
                        f'    Undeclared atom "{x[0]}" in bond ("{x[1]}", "{x[2]}")'
                        for x in bad_bonds
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_torsions(res, allatoms, allconns):
    bad_torsions = []
    for i in res.torsions:
        for a_i in (i.a, i.b, i.c, i.d):
            if a_i.atom is None:
                if a_i.connection is None or a_i.connection not in allconns:
                    bad_torsions.append(("connection", a_i, i))
            else:
                if a_i.atom not in allatoms:
                    bad_torsions.append(("atom", a_i, i))
    if len(bad_torsions) > 0:

        def str_ua(x):
            return x.atom if x.atom is not None else x.connection

        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_torsion; Torsion atoms must be either ",
                "previously-declared connections or previously-declared atoms\n",
                f'Offending atom{"s" if len(bad_torsions) > 1 else ""}:\n',
                "\n".join(
                    [
                        f'    atom "{str_ua(x[1])}" of'
                        f" ({str_ua(x[2].a)}, {str_ua(x[2].b)},"
                        f" {str_ua(x[2].c)}, {str_ua(x[2].d)})"
                        f" is not a declared {x[0]}"
                        for x in bad_torsions
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_connections(res, allatoms):
    bad_conns = []
    for i in res.connections:
        if i.atom not in allatoms:
            bad_conns.append(i)
    if len(bad_conns) > 0:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_connection; connection atom must be previously declared\n",
                f'Offending connection{"s" if len(bad_conns) > 1 else ""}:\n',
                "\n".join(
                    [
                        f'    connection "{x.name}" with undeclared atom "{x.atom}"'
                        for x in bad_conns
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_icoors(res, allatoms, allconns):
    bad_icoors = []
    for i in res.icoors:
        if i.name not in allatoms and i.name not in allconns:
            bad_icoors.append((i, "name"))
        for anc in ("parent", "grand_parent", "great_grand_parent"):
            a_i = getattr(i, anc)
            if a_i not in allatoms and a_i not in allconns:
                bad_icoors.append((i, anc))
    if len(bad_icoors) > 0:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_icoor; must reference previoulsy-declared atoms or connections only.\n",
                f'Offending icoor{"s" if len(bad_icoors) > 1 else ""}\n',
                "\n".join(
                    [
                        f'    icoor for {x[0].name}: {x[1]} atom "{getattr(x[0],x[1])}" undeclared'
                        for x in bad_icoors
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def validate_patch(patch):
    """Validate a given patch object or raise a RuntimeException"""
    addedatoms = set([i.name for i in patch.add_atoms])
    added_ats_and_conns = addedatoms.union(set([i.name for i in patch.add_connections]))

    _validate_patch_fields(patch)
    _validate_patch_atom_references(
        patch, "remove_atoms", "remove_nonreference_atom", lambda x: x
    )
    _validate_patch_atom_references(
        patch, "modify_atoms", "modify_nonreference_atom", lambda x: x.name
    )
    _validate_patch_atom_aliases(patch, addedatoms)
    _validate_patch_bonds(patch, added_ats_and_conns)
    _validate_patch_icoors(patch, added_ats_and_conns)


def _validate_patch_fields(patch):
    # make sure all fields are defined
    missing = []
    required_attrs = [
        "name",
        "display_name",
        "pattern",
        "remove_atoms",
        "add_atoms",
        "add_atom_aliases",
        "modify_atoms",
        "add_connections",
        "add_bonds",
        "icoors",
    ]
    for attribute in required_attrs:
        if getattr(patch, attribute) is None:
            missing.append(attribute)
    if len(missing) > 0:
        err_msg = "".join(
            [
                f"Bad patch: {patch.name}\n",
                "Error: Undefined field",
                ("s: " if len(missing) > 1 else ": "),
                ", ".join(missing),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_atom_references(
    patch, atom_list_attribute_name, error_name, get_atom_name
):
    bad_atom_inds = []
    atom_list = getattr(patch, atom_list_attribute_name)
    for i in range(len(atom_list)):
        if (
            get_atom_name(atom_list[i])[0] != "<"
            or get_atom_name(atom_list[i])[-1] != ">"
        ):
            bad_atom_inds.append(i)
    if len(bad_atom_inds) > 0:
        err_msg = "".join(
            [
                f"Bad patch: {patch.name}\n",
                f"Error: {error_name}; ",
                f'atoms listed with "{atom_list_attribute_name}" must begin with "<" and end with ">".\n',
                "Offending atoms: ",
                ",".join([get_atom_name(atom_list[i]) for i in bad_atom_inds]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_atom_aliases(patch, addedatoms):
    # make sure that aliases are for new atoms
    bad_aliases = []
    for i in patch.add_atom_aliases:
        if i.name not in addedatoms:
            bad_aliases.append(i)

    if len(bad_aliases) > 0:
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal_add_alias. "
                "Added atom alias must refer to newly added atoms. "
                "Bad add_atom_aliases from: ",
                ", ".join([f'"{x.name}" --> "{x.alt_name}"' for x in bad_aliases]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_bonds(patch, added_ats_and_conns):
    # make sure all bonds are references or added atoms
    bad_bonds = []
    for i, j in patch.add_bonds:
        if (i[0] != "<" or i[-1] != ">") and (i not in added_ats_and_conns):
            bad_bonds.append((i, j))
    if len(bad_bonds) > 0:
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal bond; "
                "first atom in each bond must be either atom reference ",
                '(start with "<" and end with ">") or an added atom.\n',
                "Offending bonds: ",
                ", ".join([f'("{x[0]}" "{x[1]}")' for x in bad_bonds]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_icoors(patch, added_ats_and_conns):
    # make sure all icoors are references or added atoms
    bad_icoors = []
    for i in patch.icoors:
        if (i.name[0] != "<" or i.name[-1] != ">") and (
            i.name not in added_ats_and_conns
        ):
            bad_icoors.append(("name", i))
        for anc in ("source", "parent", "grand_parent", "great_grand_parent"):
            a_i = getattr(i, anc)
            if a_i is None:
                # ok for source to be None, ok for p, gp, ggp to be None if source is given
                if anc != "source" and getattr(i, "source") is None:
                    bad_icoors.append((anc, i))
                continue
            if (a_i[0] != "<" or a_i[-1] != ">") and (a_i not in added_ats_and_conns):
                bad_icoors.append((anc, i))
    if len(bad_icoors) > 0:
        added_ats_and_conns_list = sorted(added_ats_and_conns)
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal_icoor; ",
                "icoor atoms must be for either atom reference ",
                '(start with "<" and end with ">") or an added atom / connection,\n'
                'and ancestor atoms may only be omitted (i.e. "None") if the "source" atom is provided\n'
                "Offending icoors:\n",
                "\n".join(
                    [
                        f'    Icoor for {x[1].name} with atom reference "{x[0]}" of "{getattr(x[1],x[0])}"'
                        for x in bad_icoors
                    ]
                ),
                "\nWhere the added atom / connection list is: ",
                ", ".join([f'"{x}"' for x in added_ats_and_conns_list]),
            ]
        )
        raise RuntimeError(err_msg)


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
            name3=res.name3,
            io_equiv_class=res.io_equiv_class,
            atoms=res.atoms,
            atom_aliases=res.atom_aliases,
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

        # 2b. add atom alias
        newres.atom_aliases = (*newres.atom_aliases, *variant.add_atom_aliases)

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
    variants: Tuple[VariantType, ...]

    @classmethod
    def from_chem_db(cls, chemdb: ChemicalDatabase):
        G = RestypeGraphBuilder({x.name: x.element for x in chemdb.atom_types})

        for variant in chemdb.variants:
            validate_patch(variant)

        for res in chemdb.residues:
            validate_raw_residue(res)

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
            validate_raw_residue(res)

        return cls(
            element_types=chemdb.element_types,
            atom_types=chemdb.atom_types,
            residues=patched_residues,
            variants=chemdb.variants,
        )
