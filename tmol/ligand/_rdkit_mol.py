"""RDKit molecule construction for ligands.

Builds an RDKit ``Mol`` from a ligand ``AtomArray`` while preserving the
source's explicit bond orders and aromatic/subtype annotations. Protonation
and partial-charge generation are handled upstream by the SMILES -> OpenBabel
mol2 step (:func:`tmol.ligand._detect.nonstandard_residue_info_from_smiles_via_mol2`),
so this module does not protonate or recompute chemistry.
"""

import logging

import biotite.structure as struc
import numpy as np
from atomworks.io.tools.rdkit import (
    BIOTITE_BOND_TYPE_TO_RDKIT,
    atom_array_to_rdkit,
    fix_charge_based_on_valence,
)
from atomworks.io.utils.ccd import get_custom_ccd_entries
from biotite.structure import AtomArray
from rdkit.Chem.rdchem import Mol
from collections.abc import Collection, Mapping
from typing import Literal

from rdkit import Chem

from tmol.ligand._detect import NonStandardResidueInfo, _strip_metals
from tmol.ligand._input_repair import correct_carboxylate_bond_orders

logger = logging.getLogger(__name__)


_SOURCE_KEKULE_PROP = "_tmol_source_kekule"
_SOURCE_AROMATIC_PROP = "_tmol_source_aromatic"


_SOURCE_SUBTYPE_PROP = "_tmol_source_subtype"


def _apply_source_subtypes(mol: Chem.Mol, atom_array: struc.AtomArray) -> None:
    """Stamp source subtype tags on atoms before H removal.

    The source AtomArray and pre-RemoveHs RDKit mol share atom indices,
    so this is the most reliable point to transfer subtype hints.
    """
    if not hasattr(atom_array, "tmol_source_subtype"):
        return
    subtypes = atom_array.tmol_source_subtype
    for idx, atom in enumerate(mol.GetAtoms()):
        if idx >= len(subtypes):
            break
        sub = str(subtypes[idx])
        if sub and sub != "?":
            atom.SetProp(_SOURCE_SUBTYPE_PROP, sub)


def _kekulize_non_ring_aromatic_bonds(mol: Chem.Mol) -> None:
    """De-aromatize non-ring aromatic bonds to explicit singles.

    Aromatic semantics are ring-based in RDKit/biotite. Non-ring aromatic
    bonds are treated as delocalization placeholders and must be explicit
    non-aromatic bonds for robust downstream handling.
    """
    changed = False
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
            changed = True
    if not changed:
        return
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            if not any(b.GetIsAromatic() for b in atom.GetBonds()):
                atom.SetIsAromatic(False)


def normalize_non_ring_aromatic_bonds(mol: Chem.Mol) -> None:
    """Normalize non-ring aromatic placeholders before RDKit sanitize."""
    _kekulize_non_ring_aromatic_bonds(mol)


def normalize_cumulated_azide(mol: Chem.Mol) -> Chem.Mol:
    """Strip a spurious H from a charge-separated azide/diazo terminus.

    Convert N=N=N-H (which RDKit does not understand) to =[N+]=[N-]
    Do nothing if this group is not found.
    """
    h_idx = [
        h.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7
        and atom.GetFormalCharge() == -1
        and any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())
        for h in atom.GetNeighbors()
        if h.GetAtomicNum() == 1
    ]
    if not h_idx:
        return mol
    rw = Chem.RWMol(mol)
    for i in sorted(set(h_idx), reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()


def _apply_atom_array_annotations(
    mol: Chem.Mol, atom_array: struc.AtomArray, arr_indices: list[int]
) -> None:
    """Apply source CIF/mol2 annotations onto ``mol`` at ``arr_indices``.

    Each ``arr_indices[i]`` is the AtomArray index for RDKit atom index ``i``.
    After ``RemoveHs``, ``arr_indices`` lists only heavy-atom source indices;
    when explicit hydrogens are kept, it is ``range(n_atoms)``.
    """
    if mol.GetNumAtoms() != len(arr_indices):
        return

    if hasattr(atom_array, "tmol_source_subtype"):
        subtypes = atom_array.tmol_source_subtype
        for mol_idx, arr_idx in enumerate(arr_indices):
            if arr_idx >= len(subtypes):
                continue
            sub = str(subtypes[arr_idx])
            if sub and sub != "?":
                mol.GetAtomWithIdx(mol_idx).SetProp(_SOURCE_SUBTYPE_PROP, sub)

    if not hasattr(atom_array, "tmol_aromatic"):
        return
    flags = atom_array.tmol_aromatic
    for mol_idx, arr_idx in enumerate(arr_indices):
        a = mol.GetAtomWithIdx(mol_idx)
        a.SetIsAromatic(bool(flags[arr_idx]) and a.IsInRing())
    for bond in mol.GetBonds():
        bond.SetIsAromatic(
            bond.IsInRing()
            and bond.GetBeginAtom().GetIsAromatic()
            and bond.GetEndAtom().GetIsAromatic()
        )
    mol.SetProp(_SOURCE_AROMATIC_PROP, "1")


def source_subtype(atom: Chem.Atom) -> str:
    """Return the source mol2 atom-type subtype tag (e.g. ``ar``, ``2``,
    ``cat``, ``pl3``, ``3``) when known, else ``""``."""
    if atom.HasProp(_SOURCE_SUBTYPE_PROP):
        return atom.GetProp(_SOURCE_SUBTYPE_PROP)
    return ""


def source_carried_kekule(mol: Chem.Mol) -> bool:
    """True iff the source molecule was constructed with Kekulé bond orders.

    Set when the input AtomArray carried
    explicit ``SINGLE`` / ``DOUBLE`` (or biotite's ``AROMATIC_SINGLE`` /
    ``AROMATIC_DOUBLE``) ring bonds — typical for mol2 files written
    with ``C.2`` (sp2). SMILES inputs come through with only
    ``AROMATIC`` bonds and leave this flag unset.
    """
    return mol.HasProp(_SOURCE_KEKULE_PROP) and mol.GetProp(_SOURCE_KEKULE_PROP) == "1"


def source_has_aromatic_annotations(mol: Chem.Mol) -> bool:
    """True iff aromatic atom flags were provided by the source input."""
    return (
        mol.HasProp(_SOURCE_AROMATIC_PROP) and mol.GetProp(_SOURCE_AROMATIC_PROP) == "1"
    )


def _remove_hs_tolerant(mol: Chem.Mol) -> Chem.Mol:
    """Remove explicit hydrogens, retrying without sanitize on failure.

    Kekulization can fail mid-pipeline for ligands with formal-charge
    nitrogens or unusual ring patterns. Falling back to ``sanitize=False``
    preserves the bond orders we already set (e.g. by
    the shared AtomWorks converter); running ``sanitize_tolerant`` here
    instead silently rewrites DOUBLE bonds back to SINGLE via the cleanup
    pass.
    """
    options = Chem.RemoveHsParameters()
    options.removeDegreeZero = True
    try:
        return Chem.RemoveHs(mol, options)
    except (
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        Chem.rdchem.AtomValenceException,
    ):
        return Chem.RemoveHs(mol, options, sanitize=False)


def _normalize_nitro(mol: Chem.Mol) -> None:
    """Rewrite pentavalent nitro N(=O)=O to [N+](=O)[O-].

    Some inputs draw nitro with two N=O double bonds (valence-5 neutral N), which
    RDKit rejects. Demote one N=O to a single bond and set the charges that make
    it valid. Runs before formal-charge inference and sanitize.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        dbl_term_o = [
            b
            for b in atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(atom).GetAtomicNum() == 8
            and b.GetOtherAtom(atom).GetDegree() == 1
        ]
        if len(dbl_term_o) < 2:
            continue
        for bond in dbl_term_o[1:]:
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.GetOtherAtom(atom).SetFormalCharge(-1)
        atom.SetFormalCharge(1)


def _normalize_carboxylate_formal_charges(mol: Chem.Mol) -> None:
    """Place a carboxylate charge on its singly bonded oxygen."""
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        oxygen_bonds = [
            bond
            for bond in carbon.GetBonds()
            if bond.GetOtherAtom(carbon).GetAtomicNum() == 8
            and bond.GetOtherAtom(carbon).GetDegree() == 1
        ]
        singles = [
            bond for bond in oxygen_bonds if bond.GetBondType() == Chem.BondType.SINGLE
        ]
        doubles = [
            bond for bond in oxygen_bonds if bond.GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(singles) != 1 or len(doubles) != 1:
            continue
        single = singles[0].GetOtherAtom(carbon)
        double = doubles[0].GetOtherAtom(carbon)
        if double.GetFormalCharge() == -1 and single.GetFormalCharge() == 0:
            double.SetFormalCharge(0)
            single.SetFormalCharge(-1)


def _normalize_exocyclic_aromatic_imine(mol: Chem.Mol) -> None:
    """Demote an exocyclic aromatic C=N to single (amino tautomer).

    Some mol2 files annotate these as a double-bond, leading to
    RDKit failures in converting to smiles.  Downgrade to C-N"""

    def arom_c(a):
        return a.GetAtomicNum() == 6 and a.GetIsAromatic()

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if (arom_c(a) and b.GetAtomicNum() == 7) or (
            arom_c(b) and a.GetAtomicNum() == 7
        ):
            bond.SetBondType(Chem.BondType.SINGLE)


# Neutral (uncharged) valence per element
_NEUTRAL_VALENCE = {7: 3, 8: 2}


def _is_counterion_oxygen(atom: Chem.Atom) -> bool:
    """True for a terminal O bonded to a cationic N (nitro / N-oxide oxygen).

    Such an O is the counter-anion of a hard N cation, never an omitted-H
    hydroxyl, so it must be -1 even without explicit H. The N is cationic when
    aromatic (pyridinium N-oxide) or over-valent (nitro/tertiary N-oxide);
    a neutral hydroxylamine R2N-OH (valence-3 N) is excluded.
    """
    if atom.GetAtomicNum() != 8 or atom.GetDegree() != 1:
        return False
    n = atom.GetNeighbors()[0]
    if n.GetAtomicNum() != 7:
        return False
    if n.GetIsAromatic():
        return True
    return int(round(sum(b.GetBondTypeAsDouble() for b in n.GetBonds()))) > 3


def _assign_formal_charges_from_valence(mol: Chem.Mol) -> None:
    """Infer formal charge from the explicit bond orders (Lewis rule).

    This is only run on structures containing explicit H.  For others, we
    let dimorphite infer both protonation state and charge -- except a nitro /
    N-oxide oxygen, which stays -1 even without H (see _is_counterion_oxygen).
    """
    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    for atom in mol.GetAtoms():
        neutral = _NEUTRAL_VALENCE.get(atom.GetAtomicNum())
        if neutral is None or atom.GetFormalCharge() != 0:
            continue
        bonds = atom.GetBonds()
        if any(b.GetBondType() == Chem.BondType.AROMATIC for b in bonds):
            continue
        valence = sum(b.GetBondTypeAsDouble() for b in bonds)
        charge = int(round(valence)) - neutral
        if charge < 0 and not has_explicit_h and not _is_counterion_oxygen(atom):
            continue  # ambiguous: omitted H, not a real anion -- leave neutral
        if charge != 0:
            atom.SetFormalCharge(charge)


def rdkit_mol_from_ligand_atom_array(
    atom_array: struc.AtomArray,
    *,
    res_name: str = "ligand",
    keep_hydrogens: bool = False,
    repair_chemistry: bool = False,
) -> Chem.Mol:
    """Build an RDKit Mol from a ligand AtomArray's explicit bond table.

    The single AtomArray -> RDKit builder for the ligand pipeline: it preserves
    the source's explicit bond orders (restoring Kekulé forms biotite's
    ``to_mol`` collapses) and aromatic/subtype annotations. Bond perception from
    geometry is intentionally unsupported — the input must carry chemistry-level
    bond orders.

    Args:
        atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
        res_name: Residue code, used only for log/error messages.
        keep_hydrogens: When True, retain explicit hydrogens from the input
            (used for ``skip_protonation`` — preserve mol2/CIF protonation).
        repair_chemistry: When True, apply last-resort chemistry normalizations
            that rewrite source bond orders to make an otherwise-unrepresentable
            input build.
    """
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if len(atom_array) == 0:
        raise ValueError(f"{res_name}: empty atom array")
    if not has_bonds and len(atom_array) != 1:
        raise ValueError(
            f"{res_name}: ligand bond inference is unsupported. "
            "Input must provide explicit bond orders (CIF with "
            "_chem_comp_bond.value_order / aromatic annotations). "
            "PDB/topology-only ligand chemistry is not supported."
        )

    raw_types = [int(t) for _, _, t in atom_array.bonds.as_array()] if has_bonds else []
    unsupported = sorted(
        set(t for t in raw_types if t not in BIOTITE_BOND_TYPE_TO_RDKIT)
    )
    if unsupported:
        logger.warning(
            "%s: unsupported bond type codes %s in ligand input; "
            "preserving original to_mol bond typing for those edges.",
            res_name,
            unsupported,
        )

    # BondType.ANY (0) marks bonds whose order was perceived from geometry
    # (PDB/topology); explicit SINGLE (1) is a real, provided bond order, so a
    # fully saturated ligand is not topology-only.
    has_custom_aromatic_flags = hasattr(atom_array, "tmol_aromatic")
    has_topology_only_bonds = any(t == int(struc.BondType.ANY) for t in raw_types)
    if has_topology_only_bonds and not has_custom_aromatic_flags:
        raise ValueError(
            f"{res_name}: ligand has topology-only bonds with no "
            "chemistry-level bond-order/aromatic annotations. "
            "PDB ligand chemistry inference is unsupported; provide ligand as CIF "
            "with explicit bond orders."
        )

    try:
        mol = atom_array_to_rdkit(
            atom_array,
            set_coord=True,
            hydrogen_policy="keep",
            annotations_to_keep=[],
            sanitize=False,
            attempt_fixing_corrupted_molecules=False,
            infer_bonds=False,
        )
        if keep_hydrogens:
            # Regeneration fills valence after covalent groups are separated;
            # as-is preparation preserves the supplied hydrogen count exactly.
            for atom in mol.GetAtoms():
                atom.SetNoImplicit(True)
    except Exception as exc:
        raise ValueError(
            f"{res_name}: failed to read explicit ligand bond chemistry "
            f"from input ({exc}). Provide a CIF with explicit bond orders."
        ) from exc
    if any(
        t
        in (
            int(struc.BondType.AROMATIC_SINGLE),
            int(struc.BondType.AROMATIC_DOUBLE),
            int(struc.BondType.AROMATIC_TRIPLE),
        )
        for t in raw_types
    ):
        mol.SetProp(_SOURCE_KEKULE_PROP, "1")
    mol = normalize_cumulated_azide(mol)
    normalize_non_ring_aromatic_bonds(mol)
    _apply_source_subtypes(mol, atom_array)
    _normalize_nitro(mol)
    _normalize_carboxylate_formal_charges(mol)
    if repair_chemistry:
        # Last-resort normalizations that rewrite source bond orders; only the
        # SMILES fallback path enables these (see docstring).
        _normalize_exocyclic_aromatic_imine(mol)
        mol = fix_charge_based_on_valence(mol)
    _assign_formal_charges_from_valence(mol)

    assign_stereochemistry_from_3d(mol)

    if keep_hydrogens:
        arr_indices = list(range(len(atom_array)))
    else:
        # Deuterium is atomic number 1 to RDKit; treat it as hydrogen here too.
        heavy_arr_indices = [
            i for i, e in enumerate(atom_array.element) if str(e) not in ("H", "D")
        ]
        mol = _remove_hs_tolerant(mol)
        arr_indices = heavy_arr_indices

    _apply_atom_array_annotations(mol, atom_array, arr_indices)
    mol = _strip_metals(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{res_name}: failed to build RDKit Mol")
    return mol


def ligand_atom_array_to_rdkit_mol(
    ligand_info: NonStandardResidueInfo,
    *,
    keep_hydrogens: bool = False,
) -> Chem.Mol:
    """Build an RDKit Mol from a detected ligand's AtomArray.

    Thin wrapper over :func:`rdkit_mol_from_ligand_atom_array`.
    """
    mol = rdkit_mol_from_ligand_atom_array(
        ligand_info.atom_array,
        res_name=ligand_info.res_name,
        keep_hydrogens=keep_hydrogens,
    )
    if ligand_info.skip_protonation and any(
        source_subtype(atom) == "co2"
        and atom.GetDegree() == 1
        and all(
            bond.GetBondType() == Chem.BondType.SINGLE
            for bond in atom.GetNeighbors()[0].GetBonds()
        )
        for atom in mol.GetAtoms()
    ):
        # Tripos delocalized COO bonds become singles during normalization.
        # Reuse the shared repair on the generated, finite mol2 geometry.
        mol = correct_carboxylate_bond_orders(mol)
    return mol


def transfer_tetrahedral_stereochemistry(
    mol: Chem.Mol,
    reference: Chem.Mol,
    atom_mapping: Mapping[int, int],
    *,
    replaced_atoms: Collection[int] = (),
) -> int:
    """Fill undefined tetrahedral centers using an explicit structural correspondence.

    ``atom_mapping`` maps reference indices to molecule indices, including any
    explicitly chosen replacement for a displaced neighbor. The caller owns that
    chemical correspondence. Reference indices in ``replaced_atoms`` explicitly
    identify displaced neighbors, whose replacements may have different elements.
    Other mapped atoms must have matching elements/isotopes;
    a center is transferred only if all neighbors, bond orders and hydrogen counts
    match. Existing chirality and coordinates are unchanged. Neighbor-order parity,
    rather than the reference's R/S label, preserves handedness when an attachment
    changes CIP priorities.

    Returns:
        Number of centers assigned.

    Raises:
        ValueError: If the mapping repeats a destination, has invalid indices, or
            maps different elements/isotopes.
    """
    if (
        not set(replaced_atoms) <= atom_mapping.keys()
        or len(set(atom_mapping.values())) != len(atom_mapping)
        or any(
            not 0 <= source < reference.GetNumAtoms()
            or not 0 <= target < mol.GetNumAtoms()
            for source, target in atom_mapping.items()
        )
    ):
        raise ValueError("Stereo correspondence requires valid, distinct atom indices")
    for source, target in atom_mapping.items():
        if source in replaced_atoms:
            continue
        left, right = reference.GetAtomWithIdx(source), mol.GetAtomWithIdx(target)
        if (left.GetAtomicNum(), left.GetIsotope()) != (
            right.GetAtomicNum(),
            right.GetIsotope(),
        ):
            raise ValueError(
                f"Stereo correspondence maps different elements/isotopes: {source} -> {target}"
            )
    pending = []
    for source, target in atom_mapping.items():
        if source in replaced_atoms:
            continue
        left, right = reference.GetAtomWithIdx(source), mol.GetAtomWithIdx(target)
        if left.GetChiralTag() not in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            continue
        if right.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        neighbors = [atom_mapping.get(atom.GetIdx()) for atom in left.GetNeighbors()]
        target_neighbors = [atom.GetIdx() for atom in right.GetNeighbors()]
        if (
            len(neighbors) not in (3, 4)
            or set(neighbors) != set(target_neighbors)
            or left.GetTotalNumHs() != right.GetTotalNumHs()
            or any(
                bond.GetBondType()
                != mol.GetBondBetweenAtoms(target, neighbor).GetBondType()
                for bond, neighbor in zip(left.GetBonds(), neighbors, strict=True)
            )
        ):
            continue
        order = [target_neighbors.index(neighbor) for neighbor in neighbors]
        invert = sum(a > b for i, a in enumerate(order) for b in order[i + 1 :]) % 2
        pending.append((right, left.GetChiralTag(), invert))
    for atom, tag, invert in pending:
        atom.SetChiralTag(tag)
        if invert:
            atom.InvertChirality()
    if pending:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return sum(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom, _, _ in pending
    )


# ---------------------------------------------------------------------------
# Template and stereochemistry helpers
#
# These read a component template into RDKit and infer stereo from geometry.
# They exist only on an AtomWorks branch that was never merged, so TMol keeps
# them here rather than pinning to that branch.
# ---------------------------------------------------------------------------


def _double_bond_ends(mol: Mol):
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            yield bond, ((i, j), (j, i))


def _impute_double_bond_substituents(mol: Mol, coords, finite):
    """Coordinates with an unresolved second substituent on a resolved double bond
    end placed opposite the resolved one, in the bond's plane."""
    coords = coords.copy()
    for _, ends in _double_bond_ends(mol):
        for end, other in ends:
            subs = [n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors()]
            subs = [x for x in subs if x != other]
            resolved = [x for x in subs if finite[x]]
            if (
                len(subs) != 2
                or len(resolved) != 1
                or not (finite[end] and finite[other])
            ):
                continue
            axis = coords[other] - coords[end]
            axis = axis / np.linalg.norm(axis)
            v = coords[resolved[0]] - coords[end]
            missing = subs[0] if subs[1] == resolved[0] else subs[1]
            coords[missing] = coords[end] + 2 * (v @ axis) * axis - v
    return coords


def _clear_undetermined_double_bond_stereo(mol: Mol, finite) -> None:
    """Drop cis/trans that rests on an unresolved atom, with its bond directions."""
    for bond, ((i, j), _) in _double_bond_ends(mol):
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if finite[[i, j, *bond.GetStereoAtoms()]].all():
            continue
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        for end in (i, j):
            for adjacent in mol.GetAtomWithIdx(end).GetBonds():
                adjacent.SetBondDir(Chem.BondDir.NONE)


def assign_stereochemistry_from_3d(mol: Mol) -> None:
    """Assign geometry-derived stereo while preserving unresolved coordinates.

    Three resolved neighbors determine a tetrahedral center's orientation even
    when its fourth neighbor is unresolved. Use a temporary conformer for that
    inference; centers with insufficient geometry remain unspecified. A double
    bond's cis/trans is read from one resolved substituent on each end.
    """
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        imputed = _impute_double_bond_substituents(mol, coords, finite)
        conf.SetPositions(imputed)
    Chem.AssignStereochemistryFrom3D(mol)
    if not finite.all():
        conf.SetPositions(coords)
        _clear_undetermined_double_bond_stereo(mol, np.isfinite(imputed).all(axis=1))
    # Record chiral H as a count without adding atoms or changing coordinates.
    # An implicit H on a ring root can otherwise reverse SMILES stereochemistry.
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and (
            hydrogens := atom.GetNumImplicitHs()
        ):
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + hydrogens)
            atom.SetNoImplicit(True)
    coords = mol.GetConformer().GetPositions()
    finite = np.isfinite(coords).all(axis=1)
    if finite.all():
        return
    probe = None
    for atom in mol.GetAtoms():
        center = atom.GetIdx()
        neighbors = np.array([n.GetIdx() for n in atom.GetNeighbors()], dtype=int)
        known = finite[neighbors]
        if finite[center] and known.all():
            continue
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if finite[center] and len(neighbors) == 4 and known.sum() == 3:
            if probe is None:
                probe = Chem.Mol(mol)
            conf = probe.GetConformer()
            conf.SetPositions(coords)
            missing = int(neighbors[~known][0])
            conf.SetAtomPosition(
                missing, 2 * coords[center] - coords[neighbors[known]].mean(axis=0)
            )
            Chem.AssignStereochemistryFrom3D(probe)
            atom.SetChiralTag(probe.GetAtomWithIdx(center).GetChiralTag())
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def ccd_template_to_rdkit(
    atom_array: AtomArray,
    *,
    hydrogen_policy: Literal["infer", "remove", "keep"] = "keep",
    **atom_array_to_rdkit_kwargs,
) -> Mol:
    """Convert a complete component template, without consulting the dictionary.

    Resolved geometry determines chirality first. Undefined tetrahedral centers
    use the template's declared R/S labels, which are verified against its graph.
    Coordinates, including NaNs, remain unchanged. The caller must supply the
    complete component: isolated-component R/S labels cannot be applied directly
    to a residue with covalent attachments or missing chemical atoms.

    Raises:
        ValueError: If the template is empty or an unresolved declared R/S center
            is incompatible with its graph.
    """
    if len(atom_array) == 0:
        raise ValueError("A CCD template must contain atoms")
    ccd_code = str(atom_array.res_name[0])
    mol = atom_array_to_rdkit(
        atom_array,
        set_coord=True,  # ... coordinate needed for stereochemistry assignment
        hydrogen_policy=hydrogen_policy,  # ... hydrogens needed for stereochemistry assignment
        **atom_array_to_rdkit_kwargs,
    )

    try:
        assign_stereochemistry_from_3d(mol)
    except (ValueError, RuntimeError):
        logger.warning(
            f"Failed to assign geometry-derived stereochemistry to {ccd_code}."
        )
    if "stereo" in atom_array.get_annotation_categories():
        declared = dict(zip(atom_array.atom_name, atom_array.stereo, strict=True))
        pending = [
            (atom, declared[atom.GetProp("atom_name")])
            for atom in mol.GetAtoms()
            if atom.HasProp("atom_name")
            and declared.get(atom.GetProp("atom_name")) in ("R", "S")
            and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        ]
        if pending:
            for atom, _ in pending:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            for atom, expected in pending:
                if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") != expected:
                    atom.InvertChirality()
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            # A declared configuration that cannot be honoured is an error for
            # caller-supplied chemistry, which is actionable. The bundled dictionary
            # declares R/S on atoms RDKit can never label -- porphyrin and corrin ring
            # nitrogens -- and refusing those would make chlorophyll and cobalamin
            # unbuildable, so they are downgraded to a warning.
            is_registered = ccd_code.upper() in get_custom_ccd_entries()
            unperceived = []
            for atom, expected in pending:
                if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") == expected:
                    continue
                if atom.HasProp("_CIPCode") or is_registered:
                    raise ValueError(
                        f"Cannot assign declared {expected} configuration to {ccd_code}.{atom.GetProp('atom_name')}"
                    )
                # Never leave the speculative tag behind as invented chirality.
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                unperceived.append(atom.GetProp("atom_name"))
            if unperceived:
                logger.warning(
                    "%s declares stereochemistry on %s, which RDKit does not perceive as stereocenters; "
                    "leaving them unspecified.",
                    ccd_code,
                    ", ".join(unperceived),
                )
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    return mol
