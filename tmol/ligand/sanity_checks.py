"""Sanity checks for ligand preparations.

``sanity_check`` is the entry point called during ligand registration;
new checks can be added here without cluttering the registration machinery.
"""

from tmol.ligand.registry import LigandPreparation


def check_OH_has_H(prep: LigandPreparation) -> None:
    """Verify that every atom with type 'OH' has at least one hydrogen neighbour.

    Args:
        prep: The ligand preparation to validate.

    Raises:
        ValueError: If an atom with atom type ``OH`` has no bonded hydrogen.
    """
    oh_atoms = {a.name for a in prep.residue_type.atoms if a.atom_type == "OH"}
    if not oh_atoms:
        return

    # Identify which atoms are hydrogens.
    h_atoms: set[str] = set()
    for a in prep.residue_type.atoms:
        if prep.atom_type_elements is not None:
            elem = prep.atom_type_elements.get(a.atom_type)
            if elem == "H":
                h_atoms.add(a.name)
                continue
        # Fallback heuristic: atom name or type starts with "H".
        if a.name.startswith("H") or a.atom_type.startswith("H"):
            h_atoms.add(a.name)

    # Build bond adjacency.
    bonded_to: dict[str, set[str]] = {}
    for bond in prep.residue_type.bonds:
        a_name, b_name = bond[0], bond[1]
        bonded_to.setdefault(a_name, set()).add(b_name)
        bonded_to.setdefault(b_name, set()).add(a_name)

    for oh_name in sorted(oh_atoms):
        neighbors = bonded_to.get(oh_name, set())
        if not (neighbors & h_atoms):
            raise ValueError(
                f"Ligand {prep.residue_type.name}: atom '{oh_name}' has "
                f"atom type 'OH' (hydroxyl oxygen) but no hydrogen is "
                f"bonded to it. Neighbors: {sorted(neighbors)}"
            )


def sanity_check(preps: list[LigandPreparation]) -> None:
    """Run all sanity checks on a batch of ligand preparations.

    Currently checks that every atom with atom type ``OH`` has at least
    one bonded hydrogen.  Additional checks can be added here.

    Args:
        preps: Ligand preparations to validate.

    Raises:
        ValueError: If any sanity check fails.
    """
    for prep in preps:
        check_OH_has_H(prep)
