"""Programmatic specialization of generated polymer components.

The ligand pipeline perceives atom types, charges, bonds, and internal
coordinates for every unknown chemical component.  This module adds only the
polymer semantics that cannot be inferred by the general ligand builder:
backbone identity, upper/lower connections, and canonical backbone torsions.
No modified-residue or carbohydrate name table is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import attr

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType
from tmol.database.scoring import CartRes
from tmol.ligand._detect import (
    NonStandardResidueInfo,
    get_chem_comp_one_letter_code,
    get_chem_comp_parent,
)
from tmol.ligand._registry import LigandPreparation


class ComponentKind(str, Enum):
    """Preparation route selected from CCD metadata and molecular topology."""

    PROTEIN = "protein"
    NUCLEIC_ACID = "nucleic_acid"
    CARBOHYDRATE = "carbohydrate"
    GENERAL = "general"


@dataclass(frozen=True)
class ComponentProfile:
    """Structure-derived component classification used by preparation."""

    kind: ComponentKind
    parent_name: str | None = None
    one_letter_code: str | None = None


_PROTEIN_BACKBONE = ("N", "CA", "C")
_NA_BACKBONE = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")


def _atom_names(info: NonStandardResidueInfo) -> set[str]:
    return {str(name).strip() for name in info.atom_names}


def _bond_names(info: NonStandardResidueInfo) -> set[frozenset[str]]:
    if info.atom_array.bonds is None:
        return set()
    names = info.atom_array.atom_name
    return {
        frozenset((str(names[a]).strip(), str(names[b]).strip()))
        for a, b, _ in info.atom_array.bonds.as_array()
    }


def _contains_path(info: NonStandardResidueInfo, path: tuple[str, ...]) -> bool:
    names = _atom_names(info)
    if not set(path) <= names:
        return False
    bonds = _bond_names(info)
    # Some PDB inputs omit intra-residue bond tables. Atom-name recognition is
    # still useful there; when bonds are present, require the claimed path.
    return not bonds or all(
        frozenset((left, right)) in bonds for left, right in zip(path, path[1:])
    )


def classify_component(info: NonStandardResidueInfo) -> ComponentProfile:
    """Classify an unknown component without a residue-name allowlist."""

    ccd_type = info.ccd_type.upper()
    parent = get_chem_comp_parent(info.res_name)
    one_letter = get_chem_comp_one_letter_code(info.res_name)

    if "PEPTIDE LINKING" in ccd_type and _contains_path(info, _PROTEIN_BACKBONE):
        return ComponentProfile(ComponentKind.PROTEIN, parent, one_letter)
    if ("RNA LINKING" in ccd_type or "DNA LINKING" in ccd_type) and _contains_path(
        info, _NA_BACKBONE
    ):
        return ComponentProfile(ComponentKind.NUCLEIC_ACID, parent, one_letter)
    if "SACCHARIDE" in ccd_type:
        return ComponentProfile(ComponentKind.CARBOHYDRATE, parent, one_letter)
    return ComponentProfile(ComponentKind.GENERAL, parent, one_letter)


def _raw_by_name(
    param_db: ParameterDatabase, name: str | None
) -> RawResidueType | None:
    if name is None:
        return None
    for residue in param_db.chemical.residues:
        if residue.name == name:
            return residue
    return None


def _nucleic_parent_name(
    profile: ComponentProfile, ccd_type: str, param_db: ParameterDatabase
) -> str | None:
    if _raw_by_name(param_db, profile.parent_name) is not None:
        return profile.parent_name
    if profile.one_letter_code is None or len(profile.one_letter_code) != 1:
        return None
    prefix = "D" if "DNA" in ccd_type.upper() else "R"
    candidate = prefix + profile.one_letter_code.upper()
    return candidate if _raw_by_name(param_db, candidate) is not None else None


def _specialize_from_parent(
    preparation: LigandPreparation,
    parent: RawResidueType,
    info: NonStandardResidueInfo,
    param_db: ParameterDatabase,
) -> LigandPreparation:
    """Give generated chemistry the parent's polymer topology and torsions."""

    generated = preparation.residue_type
    names = {atom.name for atom in generated.atoms}
    mainchain = parent.properties.polymer.mainchain_atoms
    if mainchain is None or not set(mainchain) <= names:
        raise ValueError(
            f"{info.res_name}: CCD parent {parent.name} backbone is not present"
        )

    parent_connections = tuple(
        connection
        for connection in parent.connections
        if connection.name in ("down", "up")
    )
    if {connection.name for connection in parent_connections} != {"down", "up"}:
        raise ValueError(
            f"{parent.name}: canonical parent lacks upper/lower connections"
        )

    element_for_type = {
        atom_type.name: atom_type.element for atom_type in param_db.chemical.atom_types
    }
    element_for_type.update(preparation.atom_type_elements or {})

    def heavy_neighbors(residue, ignored=()):
        atom_by_name = {atom.name: atom for atom in residue.atoms}
        result = {atom.name: set() for atom in residue.atoms}
        for atom1, atom2, *_ in residue.bonds:
            if (
                atom2 not in ignored
                and element_for_type[atom_by_name[atom2].atom_type] != "H"
            ):
                result[atom1].add(atom2)
            if (
                atom1 not in ignored
                and element_for_type[atom_by_name[atom1].atom_type] != "H"
            ):
                result[atom2].add(atom1)
        return result

    parent_atom = {atom.name: atom for atom in parent.atoms}

    def hydrogen_neighbors(residue, atom_name):
        atom_by_name = {atom.name: atom for atom in residue.atoms}
        neighbors = []
        for atom1, atom2, *_ in residue.bonds:
            neighbor = (
                atom2 if atom1 == atom_name else atom1 if atom2 == atom_name else None
            )
            if (
                neighbor is not None
                and element_for_type[atom_by_name[neighbor].atom_type] == "H"
            ):
                neighbors.append(neighbor)
        return sorted(neighbors)

    leaving_hydrogens = []
    for connection in parent_connections:
        expected = len(hydrogen_neighbors(parent, connection.atom))
        actual = hydrogen_neighbors(generated, connection.atom)
        leaving_hydrogens.extend(actual[expected:])
    terminal_heavy_atoms = {
        atom.name
        for variant in param_db.chemical.variants
        if variant.applies_to.matches(parent)
        and ({"<{down}>", "<{up}>"} & set(variant.remove_atoms))
        for atom in variant.add_atoms
        if atom.name not in parent_atom and element_for_type[atom.atom_type] != "H"
    }
    removed_atoms = set(leaving_hydrogens) | (names & terminal_heavy_atoms)
    generated_heavy_neighbors = heavy_neighbors(generated, removed_atoms)
    parent_heavy_neighbors = heavy_neighbors(parent)
    stable_atoms = {
        atom.name
        for atom in generated.atoms
        if atom.name not in removed_atoms
        and atom.name in parent_atom
        and generated_heavy_neighbors[atom.name] == parent_heavy_neighbors[atom.name]
    }
    specialized_atoms = tuple(
        (
            attr.evolve(atom, atom_type=parent_atom[atom.name].atom_type)
            if atom.name in stable_atoms
            else atom
        )
        for atom in generated.atoms
        if atom.name not in removed_atoms
    )
    specialized_bonds = tuple(
        bond for bond in generated.bonds if not removed_atoms.intersection(bond[:2])
    )
    specialized_icoors = tuple(
        icoor for icoor in generated.icoors if icoor.name not in removed_atoms
    )
    if any(
        removed_atoms.intersection(
            (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)
        )
        for icoor in specialized_icoors
    ):
        raise ValueError(f"{info.res_name}: a removed terminus atom is not a leaf atom")
    specialized_torsions = tuple(
        torsion
        for torsion in generated.torsions
        if not removed_atoms.intersection(
            unresolved.atom
            for unresolved in (torsion.a, torsion.b, torsion.c, torsion.d)
            if unresolved.atom is not None
        )
    )
    names -= removed_atoms
    parent_icoors = tuple(
        icoor for icoor in parent.icoors if icoor.name in ("down", "up")
    )
    parent_torsions = tuple(
        torsion
        for torsion in parent.torsions
        if torsion.name in ("phi", "psi", "omega")
        or (
            torsion.name.startswith("chi")
            and all(
                unresolved.atom is None or unresolved.atom in names
                for unresolved in (torsion.a, torsion.b, torsion.c, torsion.d)
            )
        )
    )
    inherited_names = {torsion.name for torsion in parent_torsions}
    # Generated extra torsions remain available to generic samplers, but do not
    # masquerade as parent Dunbrack dimensions.
    extra_torsions = tuple(
        attr.evolve(torsion, name=f"component_{torsion.name}")
        for torsion in specialized_torsions
        if torsion.name not in inherited_names
    )
    properties = attr.evolve(
        generated.properties,
        polymer=parent.properties.polymer,
        chemical_modifications=tuple(
            dict.fromkeys(
                (*generated.properties.chemical_modifications, info.res_name.lower())
            )
        ),
        virtual=tuple(
            atom for atom in generated.properties.virtual if atom not in removed_atoms
        ),
    )
    specialized = attr.evolve(
        generated,
        base_name=parent.base_name,
        atoms=specialized_atoms,
        atom_aliases=tuple(
            alias for alias in generated.atom_aliases if alias.name not in removed_atoms
        ),
        bonds=specialized_bonds,
        connections=parent_connections,
        torsions=(*parent_torsions, *extra_torsions),
        icoors=(*specialized_icoors, *parent_icoors),
        properties=properties,
        chi_samples=(),
        default_jump_connection_atom=(
            parent.default_jump_connection_atom
            if parent.default_jump_connection_atom in names
            else generated.default_jump_connection_atom
        ),
        one_letter_code=parent.one_letter_code,
    )
    parent_charges = {
        parameter.atom: parameter.charge
        for parameter in param_db.scoring.elec.atom_charge_parameters
        if parameter.res == parent.name
    }
    charges = {
        atom_name: charge
        for atom_name, charge in preparation.partial_charges.items()
        if atom_name not in removed_atoms
    }
    # Removing terminal atoms leaves the ligand model within rounding error of
    # the modified polymer's integral formal charge. Preserve that total while
    # retaining canonical charges on the chemically unchanged parent region.
    target_charge = round(sum(charges.values()))
    charges.update(
        {
            atom_name: parent_charges[atom_name]
            for atom_name in stable_atoms
            if atom_name in parent_charges
        }
    )
    modified_heavy_atoms = [
        atom.name
        for atom in specialized_atoms
        if atom.name not in stable_atoms and element_for_type[atom.atom_type] != "H"
    ]
    charge_correction = target_charge - sum(charges.values())
    if modified_heavy_atoms:
        per_atom_correction = charge_correction / len(modified_heavy_atoms)
        for atom_name in modified_heavy_atoms:
            charges[atom_name] += per_atom_correction

    def keep_parameter(parameter):
        return not removed_atoms.intersection(
            getattr(parameter, field)
            for field in ("atm1", "atm2", "atm3", "atm4")
            if hasattr(parameter, field)
        )

    cartbonded = preparation.cartbonded_params
    cartbonded = CartRes(
        length_parameters=tuple(filter(keep_parameter, cartbonded.length_parameters)),
        angle_parameters=tuple(filter(keep_parameter, cartbonded.angle_parameters)),
        torsion_parameters=tuple(filter(keep_parameter, cartbonded.torsion_parameters)),
        improper_parameters=tuple(
            filter(keep_parameter, cartbonded.improper_parameters)
        ),
        hxltorsion_parameters=tuple(
            filter(keep_parameter, cartbonded.hxltorsion_parameters)
        ),
    )
    return replace(
        preparation,
        residue_type=specialized,
        partial_charges=charges,
        cartbonded_params=cartbonded,
    )


def specialize_component_preparation(
    preparation: LigandPreparation,
    info: NonStandardResidueInfo,
    param_db: ParameterDatabase,
) -> tuple[LigandPreparation, ComponentProfile]:
    """Apply a protein/NA backbone profile; leave sugars/general chemistry generic.

    Carbohydrates are intentionally not promoted to a linear polymer here:
    their upper/lower/branch roles depend on the explicit inter-component graph
    and are assigned by the covalent-topology layer instead of a residue table.
    """

    profile = classify_component(info)
    if profile.kind is ComponentKind.PROTEIN:
        parent = _raw_by_name(param_db, profile.parent_name)
        if parent is None:
            raise ValueError(
                f"{info.res_name}: CCD peptide parent {profile.parent_name!r} "
                "is unavailable in the parameter database"
            )
        return _specialize_from_parent(preparation, parent, info, param_db), profile
    if profile.kind is ComponentKind.NUCLEIC_ACID:
        parent_name = _nucleic_parent_name(profile, info.ccd_type, param_db)
        parent = _raw_by_name(param_db, parent_name)
        if parent is None:
            raise ValueError(
                f"{info.res_name}: no canonical nucleic-acid parent can be resolved"
            )
        return _specialize_from_parent(preparation, parent, info, param_db), profile
    return preparation, profile
