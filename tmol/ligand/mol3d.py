"""Authoritative partial-charge mapping for prepared ligands.

The unified ligand path derives every ligand's partial charges from its SMILES
via OpenBabel MMFF94 (see
:func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`). Those
charges arrive on the detected ligand as an ``{atom_name: charge}`` map in
source-atom order. This module maps them onto the prepared molecule purely by
stable atom index, so charges are wholly independent of any later atom renaming
and no force-field recomputation is ever attempted.

If authoritative charges are missing, incomplete, or mis-sized, preparation
fails loudly rather than guessing -- there is no RDKit/Gasteiger fallback.
"""

from typing import Mapping, Optional, Sequence

from rdkit import Chem


def authoritative_charges_by_index(
    source_atom_names: Sequence[str],
    partial_charges: Optional[Mapping[str, float]],
    mol: Chem.Mol,
    *,
    ligand_name: str = "",
) -> dict[int, float]:
    """Return ``{atom_index: charge}`` mapping source charges onto ``mol``.

    ``source_atom_names[i]`` must name atom ``i`` of ``mol``. The SMILES -> mol2
    reader preserves atom order from the OpenBabel mol2 through to the prepared
    molecule, so the per-atom MMFF94 charges can be applied directly by index --
    independent of any downstream atom renaming.

    Args:
        source_atom_names: Atom names in source (OpenBabel mol2) order.
        partial_charges: Authoritative ``{atom_name: charge}`` map from the
            SMILES -> OpenBabel MMFF94 step.
        mol: The prepared RDKit molecule (same atom order as the source).
        ligand_name: Optional residue name for error messages.

    Returns:
        ``{rdkit_atom_index: partial_charge}`` for every atom in ``mol``.

    Raises:
        ValueError: If charges are absent, incomplete, or atom counts disagree.
    """
    prefix = f"{ligand_name}: " if ligand_name else ""
    if not partial_charges:
        raise ValueError(
            f"{prefix}no authoritative partial charges available. The unified "
            "ligand path requires OpenBabel MMFF94 charges from the SMILES step; "
            "no RDKit/Gasteiger charge fallback is used."
        )
    n_atoms = mol.GetNumAtoms()
    if len(source_atom_names) != n_atoms:
        raise ValueError(
            f"{prefix}atom-count mismatch mapping charges by index: "
            f"{len(source_atom_names)} source names vs {n_atoms} prepared atoms."
        )
    by_index: dict[int, float] = {}
    missing: list[str] = []
    for index, name in enumerate(source_atom_names):
        if name in partial_charges:
            by_index[index] = float(partial_charges[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"{prefix}authoritative partial charges missing for atoms: {missing}."
        )
    return by_index
