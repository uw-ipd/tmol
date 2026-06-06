"""Detection of non-standard residues in biotite AtomArrays.

Identifies residues that are not represented in tmol's ChemicalDatabase
and classifies them using Biotite's built-in Chemical Component Dictionary
(CCD) as either true ligands (non-polymer) or modified amino acids /
nucleotides (polymer-linked).
"""

import functools
import logging
from pathlib import Path
from typing import Optional

import attr
import biotite.structure as struc
import biotite.structure.info as struc_info
import biotite.structure.info.ccd as ccd
import numpy as np
from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import RWMol, rdDetermineBonds

from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.mol2_names import apply_disambiguated_mol2_names

logger = logging.getLogger(__name__)

AA_LIKE_CHEM_TYPES = frozenset(
    {
        "D-PEPTIDE LINKING",
        "D-PEPTIDE NH3 AMINO TERMINUS",
        "L-PEPTIDE LINKING",
        "L-PEPTIDE NH3 AMINO TERMINUS",
        "PEPTIDE LINKING",
        "PEPTIDE-LIKE",
    }
)

NA_LIKE_CHEM_TYPES = frozenset(
    {
        "DNA LINKING",
        "DNA OH 3 PRIME TERMINUS",
        "DNA OH 5 PRIME TERMINUS",
        "L-DNA LINKING",
        "L-RNA LINKING",
        "RNA LINKING",
        "RNA OH 3 PRIME TERMINUS",
        "RNA OH 5 PRIME TERMINUS",
    }
)

SKIP_RESIDUES = frozenset({"HOH", "WAT", "DOD", "VRT"})


@attr.s(auto_attribs=True, frozen=True)
class NonStandardResidueInfo:
    """Detected non-standard residue requiring preparation.

    Any residue not in tmol's standard database is represented here,
    regardless of whether it is a true ligand, modified amino acid,
    or modified nucleotide.

    Attributes:
        res_name: Three-letter residue code (e.g. "ATP", "NAG").
        ccd_type: CCD chemical component type string, or "UNKNOWN" if the
            residue is not in the CCD.  Informational only.
        atom_names: Atom names for one representative instance.
        elements: Element symbols for each atom.
        coords: Cartesian coordinates of shape (n_atoms, 3).
        atom_array: The sub-AtomArray (with bonds if available).
        ccd_smiles: Canonical SMILES from the CCD, or None if unavailable.
        partial_charges: Optional ``{atom_name: charge}`` map. When
            provided, ``prepare_single_ligand`` uses these directly instead of
            recomputing MMFF94 charges.
        skip_protonation: If True, Dimorphite-DL protonation is skipped and
            explicit hydrogens from the input (mol2/CIF) are preserved.
        source_path: Optional path to the originating mol2/CIF file.
    """

    res_name: str
    ccd_type: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray = attr.ib(eq=False, hash=False)
    atom_array: struc.AtomArray = attr.ib(eq=False, hash=False)
    ccd_smiles: Optional[str] = None
    covalently_linked: bool = False
    partial_charges: Optional[dict[str, float]] = None
    skip_protonation: bool = False
    source_path: Optional[Path] = None


LigandInfo = NonStandardResidueInfo


@functools.cache
def _chem_comp_type_dict() -> dict[str, str]:
    """Build a dict mapping CCD component IDs to their chemical type."""
    ccd_data = ccd.get_ccd()
    ids = np.char.upper(ccd_data["chem_comp"]["id"].as_array())
    types = np.char.upper(ccd_data["chem_comp"]["type"].as_array())
    return dict(zip(ids, types))


def get_chem_comp_type(res_name: str) -> Optional[str]:
    """Look up the CCD chemical component type for a residue name.

    Args:
        res_name: Three-letter residue code.

    Returns:
        The CCD type string (e.g. "NON-POLYMER", "L-PEPTIDE LINKING"),
        or None if the code is not found in the CCD.
    """
    return _chem_comp_type_dict().get(res_name.upper())


_METAL_SYMBOLS = frozenset(
    {
        "Fe",
        "Zn",
        "Cu",
        "Mn",
        "Co",
        "Ni",
        "Mg",
        "Ca",
        "Na",
        "K",
        "Cr",
        "Mo",
        "W",
        "V",
        "Pt",
        "Pd",
        "Ru",
        "Rh",
        "Ir",
        "Os",
    }
)


def _extract_authoritative_partial_charges(
    atom_array: struc.AtomArray,
) -> Optional[dict[str, float]]:
    """Extract authoritative per-atom partial charges from annotations.

    The ligand pipeline should only skip MMFF when we have a complete charge
    map keyed by atom name. We therefore require:
    - one value per atom,
    - finite numeric values, and
    - unique atom names.

    The `partial_charge` annotation is preferred. A generic `charge`
    annotation is accepted only when it does not look like integer formal
    charges.
    """
    atom_names = [str(name).strip() for name in atom_array.atom_name]
    if len(set(atom_names)) != len(atom_names):
        return None

    candidates = [
        ("partial_charge", True),
        ("tmol_partial_charge", True),
        ("charge", False),
    ]

    for field_name, is_explicit_partial in candidates:
        if not hasattr(atom_array, field_name):
            continue

        raw = np.asarray(getattr(atom_array, field_name))
        if raw.ndim != 1 or raw.shape[0] != len(atom_array):
            continue

        try:
            vals = raw.astype(np.float64)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(vals).all():
            continue

        # Heuristic guardrail: integer-valued "charge" from biotite is often
        # formal charge (e.g., +1/-1/0), not authoritative partial charges.
        if not is_explicit_partial and np.all(np.isclose(vals, np.round(vals))):
            continue

        by_name = {name: float(q) for name, q in zip(atom_names, vals)}
        if len(by_name) != len(atom_names):
            continue

        return by_name

    return None


def _strip_metals(mol: Chem.Mol) -> Chem.Mol:
    """Remove metal atoms from an RDKit Mol.

    OpenBabel downstream cannot parse CCD coordination-bond SMILES, and
    metals are dropped during ligand preparation anyway.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in _METAL_SYMBOLS]
    if metals:
        em = RWMol(mol)
        for idx in sorted(metals, reverse=True):
            em.RemoveAtom(idx)
        return em.GetMol()
    return mol


def _rdkit_bond_to_biotite_type(bond: Chem.Bond) -> int:
    """Map an RDKit bond to a Biotite ``BondType`` integer."""
    if bond.GetIsAromatic():
        return int(struc.BondType.AROMATIC)
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE:
        return int(struc.BondType.SINGLE)
    if btype == Chem.BondType.DOUBLE:
        return int(struc.BondType.DOUBLE)
    if btype == Chem.BondType.TRIPLE:
        return int(struc.BondType.TRIPLE)
    if btype == Chem.BondType.QUADRUPLE:
        return int(struc.BondType.QUADRUPLE)
    return int(struc.BondType.SINGLE)


def _infer_res_name_from_mol2(mol: Chem.Mol, fallback: str) -> str:
    """Best-effort residue-name extraction from a Mol2-loaded RDKit Mol."""
    for atom in mol.GetAtoms():
        if atom.HasProp("_TriposSubstName"):
            subst = atom.GetProp("_TriposSubstName").strip()
            if subst:
                return subst
    return fallback


def _source_subtype_from_mol2_atom_type(mol2_type: str) -> str:
    """Return the subtype suffix from Tripos atom type (e.g. ``C.ar`` -> ``ar``)."""
    if not mol2_type:
        return "?"
    parts = mol2_type.split(".")
    if len(parts) < 2:
        return "?"
    return parts[1] or "?"


def _mol2_charge_model(mol2_path: Path) -> str:
    """Return the Tripos charge model line from a mol2 file (e.g. ``GASTEIGER``)."""
    in_molecule = False
    lines_after_molecule = 0
    with mol2_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>MOLECULE"):
                in_molecule = True
                lines_after_molecule = 0
                continue
            if not in_molecule:
                continue
            if stripped.startswith("@<TRIPOS>"):
                break
            if not stripped:
                continue
            lines_after_molecule += 1
            # MOLECULE block: name, counts, comment, type, charge_type, ...
            if lines_after_molecule == 5:
                return stripped.upper()
    return ""


def _mol2_partial_charges_are_authoritative(mol2_path: Path) -> bool:
    """True only when mol2 partial charges are a trusted force-field model."""
    model = _mol2_charge_model(mol2_path)
    if not model:
        return False
    if model == "GASTEIGER":
        return False
    # PLI fixtures and legacy mol2s use Gasteiger; MMFF94/AM1-BCC/etc. are OK.
    return True


def _charges_are_plausible(charges: dict[str, float]) -> bool:
    """True if ``charges`` look like real partial charges.

    Guards against mol2 files whose charge column holds garbage (e.g. a
    ``USER_CHARGES``/``INVALID_CHARGES`` block with coordinate-like values):
    partial charges have small magnitudes and a near-integer net charge.
    """
    if not charges:
        return False
    if any(abs(q) > 2.0 for q in charges.values()):
        return False
    net = sum(charges.values())
    return abs(net - round(net)) < 0.1


def nonstandard_residue_info_from_mol2(
    mol2_path: str | Path,
    res_name: str | None = None,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand Mol2 file.

    This path preserves Tripos aromatic flags, atom-type subtypes, and
    per-atom partial charges when present, avoiding lossy rdkit<->biotite
    round-trips.
    """
    from tmol.ligand.atom_typing import sanitize_tolerant

    path = Path(mol2_path)
    mol = Chem.MolFromMol2File(
        str(path),
        sanitize=False,
        removeHs=False,
        cleanupSubstructures=False,
    )
    if mol is None:
        raise ValueError(f"Could not parse Mol2 file: {path}")
    sanitize_tolerant(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Mol2 file has no 3D coordinates: {path}")
    disambiguated_names = apply_disambiguated_mol2_names(mol)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    atom_names: list[str] = []
    elements: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    aromatic_flags = np.zeros(n_atoms, dtype=bool)
    source_subtypes: list[str] = []
    has_full_partial_charges = True
    partial_charges: dict[str, float] = {}

    for i, atom in enumerate(mol.GetAtoms()):
        name = disambiguated_names[i]
        atom_names.append(name)
        elements.append(atom.GetSymbol())
        p = conf.GetAtomPosition(i)
        coords[i] = [float(p.x), float(p.y), float(p.z)]
        mol2_type = (
            atom.GetProp("_TriposAtomType") if atom.HasProp("_TriposAtomType") else ""
        )
        subtype = _source_subtype_from_mol2_atom_type(mol2_type)
        source_subtypes.append(subtype)
        aromatic_flags[i] = subtype == "ar" or atom.GetIsAromatic()
        if atom.HasProp("_TriposPartialCharge"):
            partial_charges[name] = float(atom.GetProp("_TriposPartialCharge"))
        else:
            has_full_partial_charges = False

    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.atom_name = np.array(atom_names, dtype="U16")
    atom_array.element = np.array(elements, dtype="U4")
    inferred_res_name = _infer_res_name_from_mol2(
        mol, fallback="LG1" if res_name is None else res_name
    )
    atom_array.res_name = np.array([inferred_res_name] * n_atoms, dtype="U8")
    atom_array.chain_id = np.array(["A"] * n_atoms, dtype="U4")
    atom_array.res_id = np.array([1] * n_atoms, dtype=np.int32)
    atom_array.hetero = np.array([True] * n_atoms, dtype=bool)
    atom_array.set_annotation("tmol_aromatic", aromatic_flags.astype(bool))
    atom_array.set_annotation(
        "tmol_source_subtype", np.array(source_subtypes, dtype="U8")
    )

    bond_array = np.array(
        [
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                _rdkit_bond_to_biotite_type(bond),
            )
            for bond in mol.GetBonds()
        ],
        dtype=np.int32,
    )
    atom_array.bonds = struc.BondList(n_atoms, bond_array)

    use_input_charges = (
        has_full_partial_charges
        and _mol2_partial_charges_are_authoritative(path)
        and _charges_are_plausible(partial_charges)
    )
    authoritative_q = partial_charges if use_input_charges else None
    return NonStandardResidueInfo(
        res_name=inferred_res_name,
        ccd_type="UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(elements),
        coords=coords,
        atom_array=atom_array,
        ccd_smiles=None,
        covalently_linked=False,
        partial_charges=authoritative_q,
        skip_protonation=authoritative_q is not None,
        source_path=path,
    )


def _cif_value_order_to_biotite_bond_type(value_order: str, aromatic_flag: str) -> int:
    """Map CIF ``chem_comp_bond`` order/aromatic fields to ``BondType``."""
    order = str(value_order).strip().upper()
    is_aromatic = str(aromatic_flag).strip().upper() == "Y"

    if is_aromatic:
        if order == "SING":
            return int(struc.BondType.AROMATIC_SINGLE)
        if order == "DOUB":
            return int(struc.BondType.AROMATIC_DOUBLE)
        if order == "TRIP":
            return int(struc.BondType.AROMATIC_TRIPLE)
        return int(struc.BondType.AROMATIC)

    if order == "SING":
        return int(struc.BondType.SINGLE)
    if order == "DOUB":
        return int(struc.BondType.DOUBLE)
    if order == "TRIP":
        return int(struc.BondType.TRIPLE)
    if order == "AROM":
        return int(struc.BondType.AROMATIC)
    return int(struc.BondType.SINGLE)


def _partial_charges_from_atom_site(
    atom_site,
    atom_names: list[str],
) -> Optional[dict[str, float]]:
    if "partial_charge" not in atom_site:
        return None
    try:
        site_names = [str(v).strip() for v in atom_site["label_atom_id"].as_array()]
        vals = np.asarray(atom_site["partial_charge"].as_array(float), dtype=np.float64)
        if not np.isfinite(vals).all():
            return None
        charge_by_name = {
            name: float(q) for name, q in zip(site_names, vals, strict=False)
        }
        out = {
            name: charge_by_name[name] for name in atom_names if name in charge_by_name
        }
        if len(out) != len(atom_names):
            return None
        return out
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _apply_cif_atom_array_annotations(arr: struc.AtomArray, atom_site) -> None:
    """Stamp per-atom CIF fields onto ``arr``, keyed by ``label_atom_id``."""
    site_names = [str(v).strip() for v in atom_site["label_atom_id"].as_array()]
    arr_names = [str(v).strip() for v in arr.atom_name]
    site_by_name = {name: i for i, name in enumerate(site_names)}

    if "tmol_aromatic" in atom_site:
        aromatic_vals = atom_site["tmol_aromatic"].as_array()
        aromatic = np.zeros(len(arr), dtype=bool)
        for i, name in enumerate(arr_names):
            j = site_by_name.get(name)
            if j is not None:
                aromatic[i] = str(aromatic_vals[j]).strip().upper() == "Y"
        arr.set_annotation("tmol_aromatic", aromatic)

    if "tmol_source_subtype" in atom_site:
        subtype_vals = atom_site["tmol_source_subtype"].as_array()
        subtypes = np.array(["?"] * len(arr), dtype="U8")
        for i, name in enumerate(arr_names):
            j = site_by_name.get(name)
            if j is not None:
                subtypes[i] = str(subtype_vals[j])
        arr.set_annotation("tmol_source_subtype", subtypes)


def _attach_chem_comp_bonds(arr: struc.AtomArray, cif, atom_names: list[str]) -> None:
    if "chem_comp_bond" not in cif.block:
        return
    bond_site = cif.block["chem_comp_bond"]
    atom_id_1 = [str(v) for v in bond_site["atom_id_1"].as_array()]
    atom_id_2 = [str(v) for v in bond_site["atom_id_2"].as_array()]
    value_order = [str(v) for v in bond_site["value_order"].as_array()]
    if "pdbx_aromatic_flag" in bond_site:
        aromatic_flags = [str(v) for v in bond_site["pdbx_aromatic_flag"].as_array()]
    else:
        aromatic_flags = ["N"] * len(value_order)
    name_to_idx = {name: i for i, name in enumerate(atom_names)}
    bonds = []
    for a1, a2, order, aromatic_flag in zip(
        atom_id_1, atom_id_2, value_order, aromatic_flags, strict=False
    ):
        if a1 not in name_to_idx or a2 not in name_to_idx:
            continue
        bonds.append(
            (
                name_to_idx[a1],
                name_to_idx[a2],
                _cif_value_order_to_biotite_bond_type(order, aromatic_flag),
            )
        )
    if bonds:
        arr.bonds = struc.BondList(len(arr), np.asarray(bonds, dtype=np.int32))


def _resolve_cif_res_name(atom_site, arr: struc.AtomArray, res_name: str | None) -> str:
    if res_name is not None:
        return res_name
    if "label_comp_id" in atom_site:
        comp_ids = [str(v).strip() for v in atom_site["label_comp_id"].as_array()]
        nonempty = [c for c in comp_ids if c and c != "?"]
        if nonempty:
            return nonempty[0]
    return str(arr.res_name[0]).strip() if len(arr) else "LG1"


def nonstandard_residue_info_from_cif(
    cif_path: str | Path,
    res_name: str | None = None,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand CIF file.

    Expects a self-contained CIF with ``_chem_comp_bond`` chemistry and optional
    per-atom ``partial_charge``, ``tmol_aromatic``, and ``tmol_source_subtype``
    fields (as produced by :func:`tmol.ligand.cif_normalization.render_cif_from_mol2`).
    """
    import biotite.structure.io.pdbx as pdbx

    path = Path(cif_path)
    cif = pdbx.CIFFile.read(str(path))
    arr = pdbx.get_structure(cif, model=1, include_bonds=True, extra_fields=["charge"])
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]

    atom_site = cif.block["atom_site"]
    res_name = _resolve_cif_res_name(atom_site, arr, res_name)
    arr.res_name = np.array([res_name] * len(arr), dtype=arr.res_name.dtype)

    atom_names = [str(v).strip() for v in arr.atom_name]
    partial_charges = _partial_charges_from_atom_site(atom_site, atom_names)
    _apply_cif_atom_array_annotations(arr, atom_site)
    _attach_chem_comp_bonds(arr, cif, atom_names)

    return NonStandardResidueInfo(
        res_name=res_name,
        ccd_type=get_chem_comp_type(res_name) or "UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(str(e) for e in arr.element),
        coords=arr.coord.copy(),
        atom_array=arr,
        ccd_smiles=_atom_array_to_smiles(arr, source=str(path), res_name=res_name),
        covalently_linked=False,
        partial_charges=partial_charges,
        skip_protonation=partial_charges is not None,
        source_path=path,
    )


def _atom_array_to_smiles(
    atom_array: struc.AtomArray,
    *,
    source: str | None = None,
    res_name: str | None = None,
) -> Optional[str]:
    """Convert an AtomArray to a canonical SMILES string via RDKit.

    Uses biotite.interface.rdkit.to_mol() for arrays with bonds.
    Falls back to rdDetermineBonds for arrays without bonds.
    Metal atoms are stripped since OpenBabel cannot handle coordination
    bond SMILES downstream.
    """
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0

    context = ", ".join(
        x
        for x in (
            f"res={res_name}" if res_name else "",
            f"source={source}" if source else "",
        )
        if x
    )

    try:
        if has_bonds:
            mol = to_mol(atom_array)
        else:
            if len(atom_array) == 0:
                return None
            rwmol = Chem.RWMol()
            conf = Chem.Conformer(len(atom_array))
            for i, (elem, coord) in enumerate(
                zip(atom_array.element, atom_array.coord)
            ):
                rwmol.AddAtom(Chem.Atom(elem.strip().capitalize()))
                conf.SetAtomPosition(
                    i, (float(coord[0]), float(coord[1]), float(coord[2]))
                )
            rwmol.AddConformer(conf, assignId=True)
            if rwmol.GetNumAtoms() > 1:
                rdDetermineBonds.DetermineBonds(rwmol)
            mol = rwmol.GetMol()
    except Exception:
        logger.debug(
            "Failed to convert AtomArray to RDKit Mol for SMILES (%s)",
            context or "unknown",
            exc_info=True,
        )
        return None

    try:
        mol = Chem.RemoveHs(mol)
        mol = _strip_metals(mol)
        smi = Chem.MolToSmiles(mol)
        return smi if smi else None
    except Exception:
        logger.debug(
            "Failed to finalize SMILES conversion after RDKit Mol construction (%s)",
            context or "unknown",
            exc_info=True,
        )
        return None


def _get_ccd_smiles(res_name: str) -> Optional[str]:
    """Look up canonical SMILES for a residue from the CCD.

    Uses biotite.structure.info.residue() to get the full CCD AtomArray
    (with bonds) and converts to SMILES via RDKit.
    Returns None if the component is not in the CCD or conversion fails.
    """
    try:
        ccd_array = struc_info.residue(res_name)
    except KeyError:
        return None
    if ccd_array is None:
        return None
    return _atom_array_to_smiles(ccd_array)


def detect_nonstandard_residues(
    atom_array: struc.AtomArray,
    canonical_ordering: CanonicalOrdering,
) -> list[NonStandardResidueInfo]:
    """Detect residues in an AtomArray that are not in tmol's database.

    Any residue whose 3-letter code is not in the canonical ordering
    is returned for preparation, regardless of whether it is a ligand,
    modified amino acid, or modified nucleotide.

    Args:
        atom_array: Biotite AtomArray from a CIF or PDB file.
        canonical_ordering: The current tmol CanonicalOrdering, which
            defines known residue types.

    Returns:
        A list of NonStandardResidueInfo objects, one per unique unknown
        residue name.
    """
    known_names = set(canonical_ordering.restype_io_equiv_classes)
    seen: set[str] = set()
    results: list[NonStandardResidueInfo] = []

    covalently_linked_names = _residue_names_with_cross_residue_bonds(atom_array)

    residue_starts = struc.get_residue_starts(atom_array)

    for start in residue_starts:
        res_name = atom_array.res_name[start].strip()

        if res_name in known_names or res_name in SKIP_RESIDUES or res_name in seen:
            continue
        seen.add(res_name)

        mask = atom_array.res_name == atom_array.res_name[start]
        if hasattr(atom_array, "res_id"):
            mask &= atom_array.res_id == atom_array.res_id[start]
        if hasattr(atom_array, "chain_id"):
            mask &= atom_array.chain_id == atom_array.chain_id[start]

        sub = atom_array[mask]
        ccd_type = get_chem_comp_type(res_name) or "UNKNOWN"
        ccd_smiles = _get_ccd_smiles(res_name)

        logger.info(
            "Detected non-standard residue %s (CCD type: %s, %d atoms)",
            res_name,
            ccd_type,
            len(sub),
        )
        partial_charges = _extract_authoritative_partial_charges(sub)
        skip_protonation = partial_charges is not None
        if partial_charges is not None:
            logger.info(
                "Using %d authoritative partial charges for %s",
                len(partial_charges),
                res_name,
            )

        results.append(
            NonStandardResidueInfo(
                res_name=res_name,
                ccd_type=ccd_type,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
                atom_array=sub,
                ccd_smiles=ccd_smiles,
                covalently_linked=res_name in covalently_linked_names,
                partial_charges=partial_charges,
                skip_protonation=skip_protonation,
            )
        )

    return results


def _residue_names_with_cross_residue_bonds(
    atom_array: struc.AtomArray,
    spatial_cutoff: float = 1.8,
) -> frozenset[str]:
    """Return the set of res_names that have at least one bond to a different residue.

    A "different residue" is identified by (chain_id, res_id, res_name) — so
    distinct instances of the same residue name (e.g. two NAGs in a glycan
    chain) also count as cross-residue bonds. Used to flag ligands that are
    covalently attached to a polymer or to other ligand instances.

    Detection runs two passes:
    1. Explicit bonds in ``atom_array.bonds`` (if present).
    2. Heavy-atom spatial proximity within ``spatial_cutoff`` Å. This
       catches covalent attachments missing from the bond table — e.g.
       CCD NON-POLYMER caps like ACE in a polypeptide, or files lacking
       ``_struct_conn`` records.
    """
    chain_ids = atom_array.chain_id if hasattr(atom_array, "chain_id") else None
    res_ids = atom_array.res_id
    res_names = atom_array.res_name

    linked: set[str] = set()

    def _record(a: int, b: int) -> None:
        """Record a covalent connection spanning different residues.

        Args:
            a: Atom index for the first endpoint.
            b: Atom index for the second endpoint.
        """
        same_chain = chain_ids is None or chain_ids[a] == chain_ids[b]
        if same_chain and res_ids[a] == res_ids[b] and res_names[a] == res_names[b]:
            return
        linked.add(res_names[a].strip())
        linked.add(res_names[b].strip())

    if atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0:
        for a, b, _ in atom_array.bonds.as_array():
            _record(int(a), int(b))

    if len(atom_array) > 1:
        heavy_mask = np.char.strip(atom_array.element.astype(str)) != "H"
        if heavy_mask.any():
            from scipy.spatial import cKDTree

            heavy_indices = np.nonzero(heavy_mask)[0]
            tree = cKDTree(atom_array.coord[heavy_mask])
            for i, j in tree.query_pairs(spatial_cutoff, output_type="ndarray"):
                _record(int(heavy_indices[i]), int(heavy_indices[j]))

    return frozenset(linked)
