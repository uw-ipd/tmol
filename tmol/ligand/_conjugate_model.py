"""Complete, capped chemical models for non-polymer connections in an input.

Models retain residue-instance and source-atom identity. Caps replace ordinary
polymer neighbours, while every attachment within a connected group is kept.
They retain observed stereochemistry but no conformers for parameter generation.
"""

from dataclasses import dataclass

import attr
import biotite.structure as struc
import networkx as nx
import numpy as np
from rdkit import Chem
from atomworks.io.tools.rdkit import (
    ccd_template_to_rdkit,
    transfer_tetrahedral_stereochemistry,
)
from atomworks.io.utils.atom_array_plus import concatenate_atom_array_plus
from atomworks.io.utils.leaving_atoms import get_leaving_atom_groups

from tmol.ligand._polymer_profile import cap_residue, profile_for_atom_array
from tmol.ligand._conjugation_patches import connection_name
from tmol.ligand._rdkit_mol import rdkit_mol_from_ligand_atom_array


@dataclass(frozen=True)
class CappedConjugateModel:
    atom_array: struc.AtomArray
    # Coordinate-free chemistry retaining stereochemistry from resolved atoms.
    molecule: Chem.Mol
    # Original input indices; synthetic cap atoms have index -1.
    source_atom_indices: np.ndarray
    source_residue_indices: np.ndarray
    # Original input atom indices and declared Biotite bond order.
    connections: tuple[tuple[int, int, int], ...]
    connection_names: tuple[tuple[str, str], ...]


def _polymer_connection(residue, atom):
    if not residue.properties.polymer.is_polymer:
        return None
    return next(
        (
            c.name
            for c in residue.connections
            if c.name in ("up", "down") and c.atom == atom
        ),
        None,
    )


def attachment_connection_name(
    atom_array, index, partner, residue, partner_residue=None
):
    """Classify a cross-residue endpoint from both atoms' chemistry.

    A polymer nitrogen's ``down`` connection means an incoming carbonyl, not
    every possible bond at that atom. Alkyl carbon and phosphorus partners are
    ordinary conjugations, including when the nitrogen is at a chain end.
    Likewise an ``up`` atom bonded to a known polymer residue anywhere but its
    ``down`` atom (a sidechain amine) is a conjugation.
    """
    atom = str(atom_array.atom_name[index])
    declared = _polymer_connection(residue, atom)
    if (
        declared == "up"
        and partner_residue is not None
        and partner_residue.properties.polymer.is_polymer
        and _polymer_connection(partner_residue, str(atom_array.atom_name[partner]))
        != "down"
    ):
        return connection_name(atom)
    if declared != "down" or str(atom_array.element[index]).strip().upper() != "N":
        return declared or connection_name(atom)
    if str(atom_array.element[partner]).strip().upper() != "C":
        return connection_name(atom)
    neighbors, orders = atom_array.bonds.get_bonds(partner)
    carbonyl = any(
        str(atom_array.element[neighbor]).strip().upper() == "O"
        and int(order) == int(struc.BondType.DOUBLE)
        for neighbor, order in zip(neighbors, orders)
        if neighbor != index
    )
    return declared if carbonyl else connection_name(atom)


def conjugate_residue_definitions(chemical_database):
    """Residue definitions a cross-residue bond can be keyed on, by input name."""
    definitions = {
        r.name: r for r in chemical_database.residues if r.name == r.base_name
    }
    for residue in list(definitions.values()):
        definitions.setdefault(residue.io_equiv_class, residue)
    for alias in chemical_database.name3_aliases:
        if alias.read_as in definitions:
            definitions.setdefault(alias.name3, definitions[alias.read_as])
    return definitions


def _cross_residue_bonds(atom_array):
    """The input's bonds whose two atoms belong to different residues."""
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    indices = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    bonds = atom_array.bonds.as_array()
    return bonds[indices[bonds[:, 0]] != indices[bonds[:, 1]]]


def unprepared_conjugate_partners(atom_array, chemical_database):
    """Residue names at a cross-residue bond that no definition describes.

    These are exactly the residues :func:`iter_capped_conjugate_models` refuses
    to guess polymer connectivity for. A residue with no cross-residue bond is
    never looked up, so its absence from the database does not matter here.
    """
    if atom_array.bonds is None:
        return frozenset()
    cross = _cross_residue_bonds(atom_array)
    if not len(cross):
        return frozenset()
    # Every peptide bond is a cross-residue bond, so reduce to the distinct
    # names before building the definition table or comparing any strings.
    names = np.unique(atom_array.res_name[cross[:, :2].reshape(-1)])
    definitions = conjugate_residue_definitions(chemical_database)
    return frozenset(str(name) for name in names if str(name) not in definitions)


def without_cross_bonds_to(atom_array, residue_names):
    """A copy with every cross-residue bond reaching ``residue_names`` removed.

    The residue itself is left in place: dropping a ligand tmol cannot prepare
    is the caller's business, and severing the covalent link is what stops a
    conjugate model from having to describe it.
    """
    trimmed = atom_array.copy()
    bonds = trimmed.bonds.as_array()
    starts = struc.get_residue_starts(trimmed, add_exclusive_stop=True)
    indices = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    unwanted = np.isin(np.asarray(trimmed.res_name), list(residue_names))
    reaches = unwanted[bonds[:, 0]] | unwanted[bonds[:, 1]]
    cut = reaches & (indices[bonds[:, 0]] != indices[bonds[:, 1]])
    trimmed.bonds = struc.BondList(len(trimmed), bonds[~cut])
    return trimmed


def iter_capped_conjugate_models(atom_array, chemical_database):
    """Yield owned models using a database in which input residues are prepared.

    Ordinary up/down polymer bonds are cut and replaced by the existing cap
    profiles. Covalent groups may contain repeated residue names, multiple
    polymer anchors, or no polymer anchor. No coordinates or residue-name
    matching are used to infer attachment bonds. Unknown residue definitions
    at a cross-residue bond raise instead of guessing polymer connectivity.
    """
    if atom_array.bonds is None:
        raise ValueError("Conjugate models require explicit input bonds")
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    indices = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    definitions = conjugate_residue_definitions(chemical_database)
    residue_definitions = {}

    def definition(ri):
        if ri not in residue_definitions:
            name = str(atom_array.res_name[starts[ri]])
            if name not in definitions:
                raise ValueError(
                    f"Unprepared residue {ri} ({name}) in a conjugate model"
                )
            residue_definitions[ri] = definitions[name]
        return residue_definitions[ri]

    links_by_residue, occupied = {}, {}
    graph = nx.Graph()
    bonds = atom_array.bonds.as_array()
    cross = bonds[indices[bonds[:, 0]] != indices[bonds[:, 1]]]

    def partner_definition(partner):
        return definitions.get(str(atom_array.res_name[partner]))

    def port(index, partner):
        ri = int(indices[index])
        return (
            ri,
            attachment_connection_name(
                atom_array, index, partner, definition(ri), partner_definition(partner)
            ),
        )

    def label(index, partner):
        ri, name = port(index, partner)
        return f"residue {ri} {atom_array.res_name[index]}.{atom_array.atom_name[index]} ({name})"

    def slot(index, partner):
        # a declared up/down atom has one partner whichever connection it takes
        ri, name = port(index, partner)
        atom = str(atom_array.atom_name[index])
        return ri, _polymer_connection(definition(ri), atom) or name

    polymer_attached = set()
    for first, second, order in cross:
        first, second = int(first), int(second)
        ri, rj = int(indices[first]), int(indices[second])
        for endpoint, partner in ((first, second), (second, first)):
            previous = occupied.setdefault(slot(endpoint, partner), partner)
            if previous != partner:
                raise ValueError(
                    f"{label(endpoint, partner)} has multiple declared partners: "
                    f"{label(previous, endpoint)} and {label(partner, endpoint)}. "
                    "Each connection accepts one partner; resolve the input bond graph."
                )
        ci = attachment_connection_name(
            atom_array, first, second, definition(ri), partner_definition(second)
        )
        cj = attachment_connection_name(
            atom_array, second, first, definition(rj), partner_definition(first)
        )
        if {ci, cj} == {"up", "down"}:
            polymer_attached.update(
                (
                    (ri, str(atom_array.atom_name[first])),
                    (rj, str(atom_array.atom_name[second])),
                )
            )
            continue
        if str(atom_array.element[first]) in ("H", "D") or str(
            atom_array.element[second]
        ) in ("H", "D"):
            raise ValueError(
                "Cross-residue hydrogen bonds are not covalent heavy-atom attachments"
            )
        links_by_residue.setdefault(ri, []).append((first, second, int(order)))
        graph.add_edge(ri, rj)

    references = {}
    for members in nx.connected_components(graph):
        group_links = tuple(
            link for ri in sorted(members) for link in links_by_residue.get(ri, ())
        )
        yield _capped_group(
            atom_array,
            starts,
            members,
            residue_definitions,
            chemical_database,
            group_links,
            tuple((port(a, b)[1], port(b, a)[1]) for a, b, _ in group_links),
            references,
            polymer_attached,
        )


def _capped_group(
    atom_array,
    starts,
    members,
    residue_definitions,
    chemical_database,
    group_links,
    connection_names,
    references,
    polymer_attached,
):
    # Release per-residue slices and cap buffers before the caller parameterizes
    # the returned model; a suspended generator would otherwise retain them.
    arrays, source_atoms, source_residues = [], [], []
    for ri in sorted(members):
        start, stop = starts[ri : ri + 2]
        source = atom_array[start:stop]
        names = list(map(str, source.atom_name))
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate atom names in residue instance {ri}")
        source_by_name = {name: int(start + i) for i, name in enumerate(names)}
        residue_type = residue_definitions[ri]
        if residue_type.properties.polymer.is_polymer:
            connections = frozenset(
                c.atom for c in residue_type.connections if c.name in ("up", "down")
            )
            profile = profile_for_atom_array(source, connections, chemical_database)
            if profile is None:
                raise ValueError(
                    f"No cap profile for polymer residue {ri} ({residue_type.name})"
                )
            # A polymer port may itself be an attachment (e.g. a depsipeptide
            # ester). Keep that real partner instead of adding a second cap.
            occupied = {
                str(atom_array.atom_name[i])
                for a, b, _ in group_links
                for i in (a, b)
                if start <= i < stop
            }
            caps = []
            for cap in profile.caps:
                if (
                    cap.bond_to in occupied
                    and (ri, cap.bond_to) not in polymer_attached
                ):
                    occupied.add(cap.name)
                else:
                    caps.append(cap)
            profile = attr.evolve(profile, caps=tuple(caps))
            prepared, caps = cap_residue(source, profile, include_coordinates=False)
            cap_names = set(caps.values())
        else:
            prepared = source[~np.isin(source.element, ("H", "D"))]
            prepared.coord[:] = np.nan
            cap_names = set()
        arrays.append(prepared)
        source_atoms.extend(
            -1 if name in cap_names else source_by_name[str(name)]
            for name in prepared.atom_name
        )
        source_residues.extend([ri] * len(prepared))
    combined = concatenate_atom_array_plus(arrays, on_annotation_mismatch_policy="drop")
    remap = {old: new for new, old in enumerate(source_atoms) if old >= 0}
    if any(a not in remap or b not in remap for a, b, _ in group_links):
        raise ValueError("Capping removed an atom involved in an attachment")
    connections = np.asarray(
        [(remap[a], remap[b], order) for a, b, order in group_links], dtype=np.int64
    )
    combined.bonds = struc.BondList(
        len(combined), np.concatenate((combined.bonds.as_array(), connections))
    )
    # Observe chirality before discarding coordinates; generated equilibrium
    # geometry must not use these experimental positions as its targets.
    source_atoms = np.asarray(source_atoms, dtype=np.int64)
    retained = source_atoms >= 0
    combined.coord[retained] = atom_array.coord[source_atoms[retained]]
    molecule = rdkit_mol_from_ligand_atom_array(combined)
    _restore_template_stereochemistry(
        molecule, atom_array, source_atoms, starts, members, references
    )
    molecule.RemoveAllConformers()
    combined.coord[:] = np.nan
    return CappedConjugateModel(
        combined,
        molecule,
        source_atoms,
        np.asarray(source_residues, dtype=np.int64),
        group_links,
        connection_names,
    )


def _restore_template_stereochemistry(
    molecule, source, source_atoms, starts, members, references
):
    """Use authored templates and declared leaving groups for unresolved centers."""
    templates = getattr(source, "_custom_ccd_registry", {})
    if not templates or np.isfinite(molecule.GetConformer().GetPositions()).all():
        return
    if molecule.GetNumAtoms() != len(source_atoms):
        raise ValueError("Template stereo requires retained conjugate atom indices")
    for ri in sorted(members):
        start, stop = starts[ri : ri + 2]
        name = str(source.res_name[start])
        template = templates.get(name)
        if template is None:
            continue
        if name not in references:
            reference = ccd_template_to_rdkit(template, hydrogen_policy="remove")
            references[name] = (
                reference,
                {
                    atom.GetProp("atom_name"): atom.GetIdx()
                    for atom in reference.GetAtoms()
                },
                get_leaving_atom_groups(template),
            )
        reference, reference_names, leaving = references[name]
        retained = np.flatnonzero((source_atoms >= start) & (source_atoms < stop))
        actual = {str(source.atom_name[source_atoms[i]]): int(i) for i in retained}
        if not actual.keys() <= reference_names.keys():
            continue
        mapping = {reference_names[name]: index for name, index in actual.items()}
        for atom in reference.GetAtoms():
            center = mapping.get(atom.GetIdx())
            if center is None or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            target = molecule.GetAtomWithIdx(center)
            if target.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            neighbors = {
                n.GetIdx(): mapping.get(n.GetIdx()) for n in atom.GetNeighbors()
            }
            missing = [i for i, mapped in neighbors.items() if mapped is None]
            if missing:
                if len(missing) != 1:
                    continue
                missing_name = reference.GetAtomWithIdx(missing[0]).GetProp("atom_name")
                if not any(
                    missing_name in group and actual.keys().isdisjoint(group)
                    for group in leaving.get(atom.GetProp("atom_name"), ())
                ):
                    continue
                replacements = [
                    n.GetIdx()
                    for n in target.GetNeighbors()
                    if n.GetIdx() not in neighbors.values()
                ]
                if len(replacements) != 1:
                    continue
                replacement = replacements[0]
                original = source_atoms[replacement]
                if original < 0 or start <= original < stop:
                    continue
                neighbors[missing[0]] = replacement
            transfer_tetrahedral_stereochemistry(
                molecule,
                reference,
                {atom.GetIdx(): center, **neighbors},
                replaced_atoms=missing,
            )


def capped_conjugate_models(atom_array, chemical_database):
    """Collect :func:`iter_capped_conjugate_models` into an owned tuple."""
    return tuple(iter_capped_conjugate_models(atom_array, chemical_database))
