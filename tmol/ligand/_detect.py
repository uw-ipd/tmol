"""Detection of non-standard residues in biotite AtomArrays.

Identifies residues that are not represented in tmol's ChemicalDatabase
and classifies them as either true ligands (non-polymer) or modified amino
acids / nucleotides (polymer-linked). The input file decides: mmCIF gives a
polymer residue a label_seq_id and a non-polymer one a "."; failing that the
file's own _chem_comp.type says. Nothing is read from the Chemical Component
Dictionary, so a residue under a code no dictionary defines is classified the
same as one under its deposited code.
"""

import logging
from pathlib import Path
from typing import Optional

import attr
import biotite.structure as struc
import numpy as np
from rdkit import Chem
from rdkit.Chem import RWMol

from tmol.io import CanonicalOrdering
from tmol.ligand._mol2_names import apply_disambiguated_mol2_names

logger = logging.getLogger(__name__)

SKIP_RESIDUES = frozenset({"HOH", "WAT", "DOD", "VRT"})
_HALOGEN_ELEMENTS = frozenset({"F", "Cl", "Br", "I", "At", "Ts"})
_SOURCE_FORMAL_CHARGE_ANNOTATION = "tmol_source_formal_charge"
_FORMAL_CHARGE_SPECIFIED_ANNOTATION = "tmol_formal_charge_specified"
# Valence a delocalized-oxyacid or amidine center fills, and the charge its
# singly-bonded, unprotonated neighbors carry.
_DELOCALIZED_VALENCE = {"P": 5, "S": 6, "C": 4, "N": 4}
_DELOCALIZED_ANION = {"O": -1, "S": -1, "N": 0}
# Bonds a neutral atom of each element carries, for setting a charge from what
# the rewritten bonds actually use and for asking what a neighbour has room for.
_NEUTRAL_VALENCE = {"N": 3, "O": 2, "P": 5, "S": 6, "C": 4}


@attr.s(auto_attribs=True, frozen=True)
class NonStandardResidueInfo:
    """Detected non-standard residue requiring preparation.

    Any residue not in tmol's standard database is represented here,
    regardless of whether it is a true ligand, modified amino acid,
    or modified nucleotide.

    Attributes:
        res_name: Three-letter residue code (e.g. "ATP", "NAG").
        component_type: Chemical component type the input file declares, or
            "UNKNOWN" where it declares none.
        in_polymer_entity: Whether the input file places this residue in a
            polymer entity (mmCIF label_seq_id). ``None`` where the source
            cannot say, as a bare SMILES cannot.
        chemistry_problem: Why this residue's chemistry cannot be established,
            or None. A residue carrying one is refused rather than prepared
            from atoms that may not be the whole molecule.
        atom_names: Atom names for one representative instance.
        elements: Element symbols for each atom.
        coords: Cartesian coordinates of shape (n_atoms, 3).
        atom_array: The sub-AtomArray (with bonds if available).
        connection_atom_names: Names of the atoms bonded to a neighbouring
            residue, which is where this residue's polymer connections attach.
            ``None`` where the source cannot say, as a bare SMILES cannot.
        partial_charges: Authoritative ``{atom_name: charge}`` map (OpenBabel
            MMFF94 charges). Set only on the mol2 / SMILES-via-mol2 reader path,
            where ``prepare_single_ligand`` consumes them directly. ``None`` for
            raw CIF/atom-array detections (the unified path re-derives charges
            from the SMILES).
        skip_protonation: If True, Dimorphite-DL protonation is skipped and
            explicit hydrogens from the input (mol2) are preserved. Paired with
            ``partial_charges`` on the mol2 / SMILES-via-mol2 path.
        original_single_bonds: Optional set of ``frozenset({name_a, name_b})``
            pairs that the source mol2 records as literal single (order ``'1'``)
            bonds, keyed by disambiguated atom name. ``build_chi_topology`` uses
            these to honor the mol2 bond order (Rosetta-faithful) instead of
            RDKit's post-kekulization order. Only set on the mol2 / SMILES-via-
            mol2 paths; ``None`` elsewhere.
    """

    res_name: str
    component_type: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray = attr.ib(eq=False, hash=False)
    atom_array: struc.AtomArray = attr.ib(eq=False, hash=False)
    covalently_linked: bool = False
    partial_charges: Optional[dict[str, float]] = None
    skip_protonation: bool = False
    original_single_bonds: Optional[frozenset[frozenset[str]]] = None
    # source atom-array index per mol2 heavy atom (SMILES-via-mol2 path only)
    source_atom_order: Optional[tuple[int, ...]] = None
    # atoms bonded to a neighbouring residue; empty when the residue stands
    #    alone, None when the source cannot say (a bare SMILES)
    connection_atom_names: Optional[frozenset[str]] = None
    # whether the input file places this residue in a polymer entity
    in_polymer_entity: Optional[bool] = None
    # why this residue's chemistry cannot be trusted, or None
    chemistry_problem: Optional[str] = None
    # {own connection atom: {(partner residue name, partner atom name)}}
    connection_partners: Optional[dict] = attr.ib(default=None, eq=False, hash=False)


def _formal_charge(residue: struc.AtomArray, index: int = 0) -> int:
    """Formal charge of one atom, or zero when the array declares none.

    ``charge`` is an optional Biotite annotation: an array built by hand, or
    read by a path that never saw a formal-charge column, simply does not have
    it. Treat that as uncharged rather than raising, which is what the callers
    below already do for the two tmol-specific charge annotations.
    """
    if "charge" not in residue.get_annotation_categories():
        return 0
    return int(residue.charge[index])


def _formal_charge_is_specified(residue: struc.AtomArray) -> bool:
    """Whether a monatomic residue carries an authoritative formal charge."""
    if _FORMAL_CHARGE_SPECIFIED_ANNOTATION in residue.get_annotation_categories():
        return bool(residue.get_annotation(_FORMAL_CHARGE_SPECIFIED_ANNOTATION)[0])
    if _SOURCE_FORMAL_CHARGE_ANNOTATION in residue.get_annotation_categories():
        source_text = str(
            residue.get_annotation(_SOURCE_FORMAL_CHARGE_ANNOTATION)[0]
        ).strip()
        if source_text not in ("", ".", "?"):
            return True
    return _formal_charge(residue) != 0


def _isolated_atom_charge_problem(
    atom_array: struc.AtomArray,
    residue: struc.AtomArray,
    *,
    covalently_linked: bool,
) -> Optional[str]:
    """Explain why an isolated atom's charge cannot be safely regenerated."""
    if covalently_linked or len(residue) != 1:
        return None

    res_name = str(residue.res_name[0]).strip()
    atom_name = str(residue.atom_name[0]).strip()
    element = str(residue.element[0]).strip().capitalize()
    if not _formal_charge_is_specified(residue):
        return (
            f"{res_name}: formal charge is unspecified for isolated {element} atom "
            f"{atom_name}; no bond or hydrogen valence can establish its charge. "
            "Supply _atom_site.pdbx_formal_charge or _chem_comp_atom.charge, enable "
            "use_ccd for a CCD component, or load its prepared .tmol parameters"
        )

    if element not in _HALOGEN_ELEMENTS or _formal_charge(residue) != 0:
        return None

    source_text = (
        str(residue.get_annotation(_SOURCE_FORMAL_CHARGE_ANNOTATION)[0]).strip()
        if _SOURCE_FORMAL_CHARGE_ANNOTATION in residue.get_annotation_categories()
        else ""
    )
    try:
        source_charge = int(source_text)
    except ValueError:
        source_charge = _formal_charge(residue)

    template = getattr(atom_array, "_custom_ccd_registry", {}).get(res_name.upper())
    ccd_charge = None
    if template is not None:
        selected = np.flatnonzero(template.atom_name.astype(str) == atom_name)
        if len(selected) == 1:
            ccd_charge = int(template.charge[selected[0]])

    disagreement = (
        f", while the CCD component definition declares {ccd_charge}"
        if ccd_charge is not None and ccd_charge != source_charge
        else ""
    )
    return (
        f"{res_name}: source explicitly declares formal charge {source_charge} "
        f"for isolated {element}{disagreement}. Neutral monatomic {element} "
        "cannot be safely regenerated as an aqueous ligand because closed-shell "
        "preparation may create HCl-like hydrogen-halide chemistry. Correct the "
        "source formal charge, supply explicit HCl/hydrogen-halide chemistry, "
        "provide prepared parameters with params_files, or skip the residue with "
        "strict_ligands=False."
    )


def get_chem_comp_type(
    res_name: str, chem_comp_types: Optional[dict] = None
) -> Optional[str]:
    """The chemical component type the input file declares for a residue.

    Args:
        res_name: Three-letter residue code.
        chem_comp_types: Types declared by the input file.

    Returns:
        The type string (e.g. "NON-POLYMER", "L-PEPTIDE LINKING"), or None
        where the file declares none.
    """
    if not chem_comp_types:
        return None
    declared = chem_comp_types.get(res_name.upper())
    return declared.upper() if declared else None


def chem_comp_types_from_cif(cif_path) -> dict:
    """Read the _chem_comp.type of every component declared by a CIF file.

    A file that does not number its residues along a polymer sequence can still
    declare their types, which is what says whether they are polymer-linking.
    """
    from atomworks.io.utils.io_utils import read_any

    cif = read_any(cif_path)
    types = {}
    for block in cif.values():
        if "chem_comp" not in block:
            continue
        category = block["chem_comp"]
        if "id" not in category or "type" not in category:
            continue
        ids = category["id"].as_array()
        values = category["type"].as_array()
        for comp_id, comp_type in zip(ids, values):
            comp_id = str(comp_id).strip().upper()
            comp_type = str(comp_type).strip().upper()
            if comp_id and comp_type and comp_type != "?":
                types[comp_id] = comp_type
    return types


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


def _strip_metals(mol: Chem.Mol) -> Chem.Mol:
    """Remove metal atoms from an RDKit Mol.

    OpenBabel downstream cannot parse coordination-bond SMILES, and metals
    are dropped during ligand preparation anyway.
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
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
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
    return int(struc.BondType.ANY)


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


def _mol2_charge_model_from_text(mol2_text: str) -> str:
    """Return the Tripos charge model from mol2 text (e.g. ``GASTEIGER``)."""
    in_molecule = False
    lines_after_molecule = 0
    for line in mol2_text.splitlines():
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
        # TRIPOS MOLECULE block order: mol_name, counts, mol_type,
        # charge_type, [status_bits], [comment]. The charge type is the
        # 4th non-blank line (optional status/comment follow it).
        if lines_after_molecule == 4:
            return stripped.upper()
    return ""


def _mol2_single_bond_ids(mol2_text: str) -> frozenset[frozenset[int]]:
    """Literal single bonds, expressed as 1-based atom-row indices.

    Tripos IDs need not be consecutive or ordered. Preserve literal singles
    before aromaticity perception for Rosetta-compatible chi classification.
    """
    atom_ids = {}
    pairs = []
    section = ""
    for line in mol2_text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0].startswith("@<TRIPOS>"):
            section = fields[0]
        elif section == "@<TRIPOS>ATOM":
            atom_ids[int(fields[0])] = len(atom_ids) + 1
        elif section == "@<TRIPOS>BOND" and len(fields) >= 4 and fields[3] == "1":
            try:
                pairs.append((int(fields[1]), int(fields[2])))
            except ValueError:
                continue
    if atom_ids:
        missing = sorted({a for pair in pairs for a in pair} - set(atom_ids))
        if missing:
            raise ValueError(
                f"MOL2 bond records reference undeclared atom IDs {missing}; Tripos "
                "IDs need not be consecutive, so an unmatched ID is not an index"
            )
    # A fragment carrying no ATOM section declares nothing to resolve against,
    # so its own numbering is all it can mean.
    return frozenset(
        frozenset((atom_ids.get(a, a), atom_ids.get(b, b))) for a, b in pairs
    )


def _charge_model_is_authoritative(model: str) -> bool:
    """Recognized prepared charges; unknown and population-analysis models regenerate."""
    return model.upper().removesuffix("_CHARGES") in {
        "MMFF94",
        "MMFF94S",
        "AM1BCC",
        "AM1-BCC",
        "AMBER",
        "EEM",
        "USER",
    }


def _apply_mol2_metadata(mol, text):
    """Preserve source properties and explicit formal charges across either reader.

    Returns the charges the file declared and, separately, the charges oxyacid
    localization had to synthesize. Keeping them apart is what lets the caller
    still tell a rewritten molecule from a merely perceived one: folding the
    synthesized ones into the declared map makes the comparison below true by
    construction.
    """
    indices = {}
    declared_charges = {}
    delocalized_bonds = set()
    coordinates = []
    section = ""
    atom_id = None
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0].startswith("@<TRIPOS>"):
            section = fields[0]
        elif section == "@<TRIPOS>ATOM":
            if int(fields[0]) in indices:
                raise ValueError(f"Duplicate MOL2 atom ID {fields[0]}")
            atom = mol.GetAtomWithIdx(len(indices))
            indices[int(fields[0])] = len(indices)
            coordinates.append([float(v) for v in fields[2:5]])
            atom.SetProp("_TriposAtomName", fields[1])
            atom.SetProp("_TriposAtomType", fields[5])
            if len(fields) >= 9:
                atom.SetDoubleProp("_TriposPartialCharge", float(fields[8]))
        elif section == "@<TRIPOS>BOND":
            if fields[3] not in ("1", "2", "3", "4", "ar", "am"):
                raise ValueError(
                    f"Undefined MOL2 bond order {fields[3]!r} between atom IDs "
                    f"{fields[1]} and {fields[2]}; supply explicit chemical bond orders"
                )
            if fields[3] == "ar":
                delocalized_bonds.add(
                    frozenset((indices[int(fields[1])], indices[int(fields[2])]))
                )
        elif section == "@<TRIPOS>UNITY_ATOM_ATTR":
            if fields[0] == "charge":
                index, charge = indices[atom_id], int(fields[1])
                mol.GetAtomWithIdx(index).SetFormalCharge(charge)
                declared_charges[index] = charge
            elif fields[0].isdigit():
                atom_id = int(fields[0])

    mol.GetConformer().SetPositions(np.asarray(coordinates, dtype=np.float64))
    synthesized_charges: dict = {}
    _infer_oxyacid_bonds(mol, declared_charges, delocalized_bonds, synthesized_charges)
    return declared_charges, synthesized_charges


def _delocalized_neighbors(mol, center, delocalized_bonds):
    """Acyclic neighbors sharing a source ``ar`` bond with ``center`` only."""
    ring_info = mol.GetRingInfo()
    if ring_info.NumAtomRings(center.GetIdx()):
        return []
    neighbors = []
    for atom in center.GetNeighbors():
        pair = frozenset((center.GetIdx(), atom.GetIdx()))
        if pair not in delocalized_bonds:
            continue
        if ring_info.NumAtomRings(atom.GetIdx()):
            return []
        if any(
            frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            in delocalized_bonds
            and center.GetIdx() not in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in atom.GetBonds()
        ):
            return []
        neighbors.append(atom)
    return neighbors


def _localize(center, neighbors, n_double, conformer, charge, charges, synthesized):
    """Write one X=Y plus single bonds, charging the unprotonated remainder.

    ``charges`` is the declared-charge map, read and written. A center the file
    charged keeps that charge -- zeroing a declared quaternary nitrogen turns a
    +1 amidinium into a -1 and fails sanitization -- and every charge written
    here is recorded, so the net the caller compares against still describes the
    molecule this leaves behind. Without that, localizing an undeclared
    phosphonate moves the net away from the declared sum and sends every such
    ligand down the OpenBabel fallback.
    """
    if conformer is not None:
        origin = np.asarray(conformer.GetAtomPosition(center.GetIdx()))
        neighbors = sorted(
            neighbors,
            key=lambda a: float(
                np.linalg.norm(
                    np.asarray(conformer.GetAtomPosition(a.GetIdx())) - origin
                )
            ),
        )
    center.SetIsAromatic(False)

    def protonated(atom):
        """Whether the file wrote a hydrogen of this neighbour's own.

        Only an explicit one counts. ``GetTotalNumHs`` also reports the implicit
        hydrogens RDKit fills in from the delocalized bonds still on the atom --
        the very bonds this routine is replacing -- so believing it makes a
        heavy-atom-only file look protonated and charges it as if it were.
        """
        return any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())

    def terminal(atom):
        """Whether this neighbour hangs off the center by its only heavy bond."""
        return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1) <= 1

    def demanded_charge(atom, order):
        """The charge this neighbour's bonds demand at that order to the center.

        Counts what the file drew -- explicit hydrogens included, since they are
        bonds. Exact for a protonated neighbour; for a heavy-atom-only one the
        missing hydrogens make it a lower bound, which is all the caller asks of
        it there.
        """
        neutral = _NEUTRAL_VALENCE.get(atom.GetSymbol())
        if neutral is None:
            return 0
        used = order + sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
            if bond.GetOtherAtomIdx(atom.GetIdx()) != center.GetIdx()
            and bond.GetBondType() != Chem.BondType.AROMATIC
        )
        return int(used - neutral)

    def wants_double(atom):
        """Whether the double bond belongs on this neighbour.

        A neighbour the file charged wants it exactly when the double bond is
        what that charge describes: a guanidinium's +1 nitrogen is the one
        holding the double bond, while a carboxylate's -1 oxygen is the one
        that is not. An uncharged neighbour wants it when taking it leaves it
        neutral -- a carbonyl oxygen or an amidine's N-H would be, a hydroxyl
        and a phosphodiester's bridging oxygen would have to become cations.
        """
        declared = charges.get(atom.GetIdx())
        if declared is not None:
            return declared == demanded_charge(atom, 2)
        return demanded_charge(atom, 2) <= 0

    # Fall back by degrees. A guanidinium the file left uncharged has no
    # neutral option at all -- every nitrogen is the cation in some Kekule form
    # -- so there the geometry decides among the undeclared, as it did before
    # any of this was filtered. Falling back further, to a neighbour whose
    # declared charge contradicts the bond it is being handed, leaves a
    # molecule that will not sanitize: that input describes no arrangement, and
    # failing there sends it to the fallback reader instead of quietly
    # inventing a carbanion that balances the books.
    undeclared = [atom for atom in neighbors if atom.GetIdx() not in charges]
    eligible = [a for a in neighbors if wants_double(a)] or undeclared or neighbors
    n_double = min(n_double, len(eligible))
    doubled = set(atom.GetIdx() for atom in eligible[:n_double])

    for atom in neighbors:
        bond = center.GetOwningMol().GetBondBetweenAtoms(center.GetIdx(), atom.GetIdx())
        bond.SetIsAromatic(False)
        atom.SetIsAromatic(False)
        double = atom.GetIdx() in doubled
        bond.SetBondType(Chem.BondType.DOUBLE if double else Chem.BondType.SINGLE)
        if atom.GetIdx() in charges:
            # The file said what this one is; recording it again as synthesized
            # would double it in the net the caller checks against.
            continue
        if protonated(atom):
            # Its hydrogens are on the page, so its bonds settle its charge:
            # neutral for a hydroxyl or an amide nitrogen, +1 for the nitrogen
            # a guanidinium's double bond lands on.
            assigned = demanded_charge(atom, 2 if double else 1)
        elif double or not terminal(atom):
            # A bridging neighbour is an ordinary two-bonded atom once the bond
            # orders are explicit. Whatever charge aromatic perception guessed
            # for it -- RDKit warns it is guessing when the file has no explicit
            # hydrogens -- describes a molecule that no longer exists.
            assigned = 0
        else:
            assigned = charge
        atom.SetFormalCharge(assigned)
        synthesized[atom.GetIdx()] = assigned

    if center.GetIdx() not in charges:
        # The center is left holding whatever its bonds now demand. Forcing 0
        # gives a four-bonded nitro nitrogen no valence for its fourth bond, and
        # the molecule stops sanitizing; P, S and C come out at 0 either way.
        used = (
            sum(bond.GetBondTypeAsDouble() for bond in center.GetBonds())
            + center.GetTotalNumHs()
        )
        neutral = _NEUTRAL_VALENCE.get(center.GetSymbol())
        assigned = int(used - neutral) if neutral is not None else 0
        center.SetFormalCharge(assigned)
        if assigned:
            synthesized[center.GetIdx()] = assigned


def _infer_oxyacid_bonds(mol, charges, delocalized_bonds, synthesized=None):
    """Localize acyclic delocalized bonds the source writes as Tripos ``ar``.

    Tripos ``ar`` off a ring means delocalization, not aromaticity: a
    phosphonate, sulfonate, carboxylate or amidine drawn this way has no
    Kekule structure and cannot be sanitized. Rewrite each such center as one
    double bond plus charged single bonds. Explicit oxygen charges decide when
    the file supplies them; otherwise the double bond is the shortest bond and
    the count comes from the center's valence, which keeps an ester or other
    substituent on the center accounted for. A neighbor holding a hydrogen is a
    real hydroxyl and keeps its bond and charge.
    """
    # Ring membership decides what may be rewritten, and RingInfo is not
    # populated until something perceives rings; the mol here is unsanitized.
    if synthesized is None:
        # A caller that does not ask for the synthesized charges still needs
        # somewhere for them to go.
        synthesized = {}
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    for center in mol.GetAtoms():
        neighbors = _delocalized_neighbors(mol, center, delocalized_bonds)
        if len(neighbors) < 2 or len({a.GetSymbol() for a in neighbors}) != 1:
            continue
        anion = _DELOCALIZED_ANION.get(neighbors[0].GetSymbol())
        if anion is None:
            continue

        # Charges the file declares settle the arrangement when they describe a
        # complete one. Most mol2 files in practice declare none at all, so the
        # geometry-and-atom-type path below is the ordinary one, and a partial
        # or unrepresentable declaration -- one oxygen of a mesylate marked, or
        # an amidinium's +1, which no arrangement of one double bond and charged
        # singles describes -- falls through to it rather than abandoning the
        # centre unlocalized.
        # Only an anionic center can be read off the declared charges: where the
        # anion is zero -- nitrogen -- "neutral" and "negative" name the same
        # atoms, so the partition below cannot describe anything and the
        # geometry path, which reads declared charges directly, is the one that
        # can.
        declared = [a for a in neighbors if a.GetIdx() in charges] if anion else []
        if declared:
            neutral = [a for a in neighbors if a.GetFormalCharge() == 0]
            negative = [a for a in neighbors if a.GetFormalCharge() == anion]
            if len(neutral) == 1 and len(negative) + 1 == len(neighbors):
                _localize(
                    center, neutral + negative, 1, None, anion, charges, synthesized
                )
                continue

        valence = _DELOCALIZED_VALENCE.get(center.GetSymbol())
        if valence is None:
            continue
        spent = sum(
            bond.GetBondTypeAsDouble()
            for bond in center.GetBonds()
            if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            not in delocalized_bonds
        )
        n_double = int(valence - spent - len(neighbors))
        if not 0 < n_double <= len(neighbors):
            continue
        conformer = mol.GetConformer() if mol.GetNumConformers() else None
        _localize(center, neighbors, n_double, conformer, anion, charges, synthesized)
    mol.UpdatePropertyCache(strict=False)


def nonstandard_residue_info_from_mol2(
    mol2_path: str | Path,
    res_name: str | None = None,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand Mol2 file.

    Low-level reader retained for the DUD-80 SMILES->params parity harness
    (it reads both the OpenBabel-generated and Rosetta ground-truth mol2
    files). Preserves Tripos aromatic flags, atom-type subtypes, and per-atom
    partial charges, avoiding lossy rdkit<->biotite round-trips.
    """
    try:
        return nonstandard_residue_info_from_mol2_block(
            Path(mol2_path).read_text(), res_name=res_name
        )
    except ValueError as error:
        raise ValueError(f"{mol2_path}: {error}") from error


def nonstandard_residue_info_from_mol2_block(
    mol2_block: str,
    res_name: str | None = None,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from an in-memory mol2 *string*.

    In-memory analogue of :func:`nonstandard_residue_info_from_mol2` — parses a
    TRIPOS mol2 block directly, with no temp-file write/read. Preferred for
    high-throughput SMILES batches (see
    :func:`nonstandard_residue_info_from_smiles_via_mol2`).
    """
    from tmol.ligand._openbabel_compat import (
        OpenBabelUnavailableError,
        obabel_read_mol2_block,
    )

    if mol2_block.count("@<TRIPOS>MOLECULE") != 1:
        raise ValueError("Expected exactly one MOL2 molecule record")
    mol = Chem.MolFromMol2Block(
        mol2_block,
        sanitize=False,
        removeHs=False,
        cleanupSubstructures=False,
    )
    if mol is None:
        try:
            mol = obabel_read_mol2_block(mol2_block)
        except OpenBabelUnavailableError:
            mol = None
        if mol is None:
            raise ValueError(
                "Could not parse in-memory Mol2 block (RDKit MolFromMol2Block "
                "failed; OpenBabel fallback also failed or is not installed)"
            )
        logger.info("Used OpenBabel fallback to parse in-memory mol2 block")
    declared_charges, synthesized_charges = _apply_mol2_metadata(mol, mol2_block)
    probe = Chem.Mol(mol)
    invalid = bool(Chem.SanitizeMol(probe, catchErrors=True))
    charge_model = _mol2_charge_model_from_text(mol2_block)
    # Tripos ``ar`` bonds carry no kekulization, so a reader that picks the
    # wrong Kekule structure can return a sanitizable but different molecule --
    # a pyrrole-type ring nitrogen comes back protonated and cationic. The mol2
    # records every nonzero formal charge, so a net charge that disagrees with
    # what was declared means the chemistry was rewritten, not just perceived.
    # Localization legitimately adds charges the file never wrote, so the net to
    # compare against is what the file declared plus what localization put
    # there. Anything else is the reader having changed the molecule.
    declared_net = sum(declared_charges.values())
    expected_net = declared_net + sum(synthesized_charges.values())
    rewritten = not invalid and Chem.GetFormalCharge(probe) != expected_net
    if invalid or rewritten:
        try:
            alternate = obabel_read_mol2_block(mol2_block)
        except OpenBabelUnavailableError:
            alternate = None
        if alternate is not None and [
            a.GetAtomicNum() for a in alternate.GetAtoms()
        ] == [a.GetAtomicNum() for a in mol.GetAtoms()]:
            _, alternate_synthesized = _apply_mol2_metadata(alternate, mol2_block)
            from atomworks.io.tools.rdkit import fix_charge_based_on_valence

            fix_charge_based_on_valence(alternate)
            preserves_charges = all(
                alternate.GetAtomWithIdx(i).GetFormalCharge() == charge
                for i, charge in declared_charges.items()
            )
            if (
                preserves_charges
                and not Chem.SanitizeMol(alternate, catchErrors=True)
                and not any(
                    atom.GetNumRadicalElectrons() for atom in alternate.GetAtoms()
                )
                # When the primary reader is unusable the alternate is the only
                # candidate; require the declared net charge only when we are
                # rejecting a primary that parsed but rewrote the chemistry.
                and (
                    invalid
                    or Chem.GetFormalCharge(alternate)
                    == declared_net + sum(alternate_synthesized.values())
                )
            ):
                mol = alternate
                invalid = False
    if invalid:
        raise ValueError(
            "MOL2 atom and bond types do not define a sanitizable chemical graph; "
            "supply corrected aromatic bond orders and protonation"
        )
    return _nonstandard_residue_info_from_mol2_mol(
        mol,
        charge_model,
        res_name,
        source="<mol2 block>",
        single_bond_ids=_mol2_single_bond_ids(mol2_block),
    )


def _nonstandard_residue_info_from_mol2_mol(
    mol: Chem.Mol,
    charge_model: str,
    res_name: str | None,
    *,
    source: str,
    single_bond_ids: frozenset[frozenset[int]] = frozenset(),
) -> NonStandardResidueInfo:
    """Build ``NonStandardResidueInfo`` from an already-parsed mol2 RDKit mol.

    Shared core of the file and in-memory-block mol2 entry points. ``charge_model``
    is the TRIPOS ``charge_type`` (already parsed from the source) used to decide
    whether the per-atom charges are authoritative; ``source`` labels the input in
    error messages. ``single_bond_ids`` is the set of 1-based atom-row pairs
    the source records as literal single bonds (see :func:`_mol2_single_bond_ids`);
    these are mapped to disambiguated atom names and stored as
    ``original_single_bonds`` so the CHI classifier can honor the mol2 bond order.
    """
    from tmol.ligand._atom_typing import sanitize_tolerant

    sanitize_tolerant(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Mol2 input has no 3D coordinates: {source}")
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
    atom_array.atom_name = np.array(atom_names, dtype=str)
    atom_array.element = np.array(elements, dtype="U4")
    inferred_res_name = (
        _infer_res_name_from_mol2(mol, fallback="L_1") if res_name is None else res_name
    )
    atom_array.res_name = np.array([inferred_res_name] * n_atoms, dtype=str)
    atom_array.chain_id = np.array(["A"] * n_atoms, dtype="U4")
    atom_array.res_id = np.array([1] * n_atoms, dtype=np.int32)
    atom_array.hetero = np.array([True] * n_atoms, dtype=bool)
    atom_array.set_annotation(
        "charge",
        np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.int32),
    )
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

    complete = Chem.Mol(mol)
    for atom in complete.GetAtoms():
        atom.SetNoImplicit(False)
    complete.UpdatePropertyCache(strict=False)
    missing_hydrogens = any(
        atom.GetNumImplicitHs() or atom.GetNumRadicalElectrons()
        for atom in complete.GetAtoms()
    )
    use_input_charges = (
        has_full_partial_charges
        and not missing_hydrogens
        and _charge_model_is_authoritative(charge_model)
        and np.isfinite(list(partial_charges.values())).all()
    )
    authoritative_q = partial_charges if use_input_charges else None

    # Map source atom rows to the names retained through preparation.
    original_single_bonds: frozenset[frozenset[str]] = frozenset(
        frozenset((disambiguated_names[i - 1], disambiguated_names[j - 1]))
        for i, j in (tuple(pair) for pair in single_bond_ids)
        if 1 <= i <= n_atoms and 1 <= j <= n_atoms
    )

    return NonStandardResidueInfo(
        res_name=inferred_res_name,
        component_type="UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(elements),
        coords=coords,
        atom_array=atom_array,
        covalently_linked=False,
        partial_charges=authoritative_q,
        skip_protonation=authoritative_q is not None,
        original_single_bonds=original_single_bonds or None,
        source_atom_order=tuple(
            i for i, element in enumerate(elements) if element != "H"
        ),
    )


def _normalize_radical_oxygens(smiles: str) -> str:
    """Restore the formal charge on bare radical oxygens (e.g. DUD ``[O]``).

    Some source databases write anionic oxygens (carboxylate, sulfonate,
    phosphate) as a bare ``[O]`` — a neutral, singly-bonded oxygen with no
    hydrogen, which RDKit reads as a radical. Dimorphite cannot act on it
    (its acid rules need an explicit ``-O-H``), and downstream 3D tools fill
    the open valence inconsistently (OpenBabel via a PDB hop adds an H ->
    neutral acid; the direct SMILES read keeps a symmetric carboxylate). Convert
    each such oxygen to the proper anion ``[O-]`` so the protonation state is
    well-defined end to end. Returns the input unchanged if nothing matches or
    the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    from atomworks.io.tools.protonation import normalize_radical_oxygens

    fixed = normalize_radical_oxygens(mol)
    return smiles if fixed is mol else Chem.MolToSmiles(fixed)


def _dimorphite_protonate_smiles(
    smiles: str, ph: float = 7.4, precision: float = 0.1
) -> str:
    """Return the SMILES pKa-protonated at ``ph`` via Dimorphite-DL.

    Takes the first protonation variant (matching the reference ligand-prep
    protocol). Falls back to the input SMILES if RDKit cannot parse it or
    Dimorphite produces no variant.
    """
    from tmol.ligand._dimorphite_dl import protonate_mol_variants

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        variants = protonate_mol_variants(
            mol,
            min_ph=ph,
            max_ph=ph,
            pka_precision=precision,
            max_variants=128,
            silent=True,
        )
    except Exception:
        logger.warning(
            "Dimorphite protonation failed for SMILES %r; using input", smiles
        )
        return smiles
    if not variants:
        return smiles
    return Chem.MolToSmiles(variants[0])


def _has_free_lone_pair(atom) -> bool:
    """Whether an atom has a lone pair left to give a metal.

    A cation has none, and neither does a trigonal nitrogen: a pyrrole or amide
    N-H keeps its pair in the pi system.
    """
    if atom.GetFormalCharge() > 0:
        return False
    if atom.GetSymbol() == "N":
        sigma = atom.GetDegree() + atom.GetTotalNumHs()
        return not (
            sigma == 3 and atom.GetHybridization() == Chem.HybridizationType.SP2
        )
    return True


def _deprotonated_for_coordination(smiles: str, map_numbers) -> str:
    """The SMILES with each mapped metal donor given a free lone pair.

    A donor without one loses a hydrogen and gains a negative charge; any
    other donor is left as protonated.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() not in map_numbers:
            continue
        while atom.GetTotalNumHs() > 0 and not _has_free_lone_pair(atom):
            atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
            atom.SetNoImplicit(True)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
            mol.UpdatePropertyCache(strict=False)
            Chem.SetHybridization(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def nonstandard_residue_info_from_smiles_via_mol2(
    smiles: str,
    res_name: str | None = None,
    *,
    ph: float = 7.4,
    protonate: bool = True,
    seed: int | None = None,
    coordinating: frozenset = frozenset(),
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a SMILES via the mol2 route.

    Implements the canonical ligand-prep protocol end to end:

    0. normalize bare radical oxygens (``[O]`` -> ``[O-]``) so source
       carboxylate/sulfonate notation has a well-defined charge,
    1. optionally pKa-protonate the SMILES with Dimorphite-DL (``protonate``),
    2. generate a 3D mol2 with MMFF94 partial charges via OpenBabel (kept
       in memory as a string — no temp file), then
    3. read that mol2 with :func:`nonstandard_residue_info_from_mol2_block`.

    This never builds a biotite atom-array from an RDKit embedding and never
    recomputes MMFF on a reconstructed graph — the OpenBabel MMFF94 charges
    flow through untouched (``skip_protonation`` / authoritative charges are
    set by the mol2 reader), so fused-ring aromatics keep correct charges.

    Args:
        smiles: Ligand SMILES string.
        res_name: Residue name (default inferred / ``"L_1"``).
        ph: Target pH for the Dimorphite protonation step.
        protonate: When ``True`` (default) run Dimorphite on ``smiles`` first;
            set ``False`` to pin an already-protonated SMILES verbatim.
        seed: Fixed RNG seed for reproducible 3D coordinates; ``None`` is random.
        coordinating: Atom map numbers of metal donors, each deprotonated
            after protonation if it has no free lone pair.

    Raises:
        OpenBabelUnavailableError: If the ``openbabel`` package is missing
            (this path requires it for the SMILES -> mol2 conversion).
        ValueError: If OpenBabel cannot build a charged mol2 for ``smiles``.
    """
    from tmol.ligand._openbabel_compat import (
        _build_charged_3d_mol2_mol,
        source_atom_order_from_mapped_smiles,
    )

    smiles = _normalize_radical_oxygens(smiles)
    prep_smiles = _dimorphite_protonate_smiles(smiles, ph) if protonate else smiles
    if coordinating:
        prep_smiles = _deprotonated_for_coordination(prep_smiles, coordinating)
    source_order = source_atom_order_from_mapped_smiles(prep_smiles)
    molecule = _build_charged_3d_mol2_mol(prep_smiles, seed=seed)
    info = nonstandard_residue_info_from_mol2_block(
        molecule.write("mol2"), res_name=res_name
    )
    # Mol2 preserves partial charges, but its atom types can lose formal charge
    # on fully deprotonated species. Keep the already protonated atom state.
    info.atom_array.set_annotation(
        "charge", np.array([a.formalcharge for a in molecule.atoms], dtype=np.int8)
    )
    return attr.evolve(
        info,
        source_atom_order=source_order,
        # The mol2 text rounds charges to four decimals. Keep the computed
        # values so rounding error cannot accumulate over larger ligands.
        partial_charges=dict(
            zip(info.atom_array.atom_name, (a.partialcharge for a in molecule.atoms))
        ),
    )


def _component_types_from_annotations(atom_array, explicit=None):
    """Consume per-atom chemistry supplied by AtomWorks or another reader.

    An explicit mapping overrides annotations. A component name must otherwise
    identify one type; conflicting annotations cannot define a reusable tmol
    residue type and are rejected before parameter generation.
    """
    result = dict(explicit or {})
    if "chem_comp_type" not in atom_array.get_annotation_categories():
        return result
    names = atom_array.res_name.astype(str)
    values = atom_array.get_annotation("chem_comp_type").astype(str)
    # Most annotations repeat across every atom of a residue. Collapse runs
    # before sorting, while still detecting contradictory values within a run.
    starts = np.r_[True, (names[1:] != names[:-1]) | (values[1:] != values[:-1])]
    pairs = (
        np.unique(np.stack((names[starts], values[starts]), axis=1), axis=0)
        if len(names)
        else []
    )
    declared = {}
    for name, value in pairs:
        name, value = name.strip().upper(), value.strip().upper()
        if not value or value in {".", "?", "UNKNOWN"} or name in result:
            continue
        previous = declared.setdefault(name, value)
        if previous != value:
            raise ValueError(
                f"Conflicting chem_comp_type annotations for {name}: "
                f"{previous!r} and {value!r}"
            )
    declared.update(result)
    return declared


def detect_nonstandard_residues(
    atom_array: struc.AtomArray,
    canonical_ordering: CanonicalOrdering,
    chem_comp_types: Optional[dict] = None,
) -> list[NonStandardResidueInfo]:
    """Detect residues in an AtomArray that are not in tmol's database.

    Any residue whose 3-letter code is not in the canonical ordering
    is returned for preparation, regardless of whether it is a ligand,
    modified amino acid, or modified nucleotide.

    Args:
        atom_array: Biotite AtomArray from a CIF or PDB file.
        canonical_ordering: The current tmol CanonicalOrdering, which
            defines known residue types.
        chem_comp_types: ``{comp_id: type}`` declared by the input file,
            consulted where it does not number residues along a sequence.

    Returns:
        A list of NonStandardResidueInfo objects, one per unique unknown
        residue name.
    """
    chem_comp_types = _component_types_from_annotations(atom_array, chem_comp_types)
    known_names = set(canonical_ordering.restype_io_equiv_classes) | SKIP_RESIDUES
    residue_starts = struc.get_residue_starts(atom_array)
    unknown_starts = [
        start
        for start in residue_starts
        if atom_array.res_name[start].strip() not in known_names
    ]
    if not unknown_starts:
        return []
    heavy_counts = np.add.reduceat(
        ~np.isin(atom_array.element, ("H", "D")), residue_starts
    )
    seen: set[str] = set()
    results: list[NonStandardResidueInfo] = []
    polymer_names = polymer_entity_residues(atom_array)

    cross_residue_atoms, cross_residue_partners = _cross_residue_bond_atoms(atom_array)
    covalently_linked_names = frozenset(
        res_name for _chain, _res_id, res_name in cross_residue_atoms
    )
    # collect by name, since all copies of a residue share one residue type.
    #    A copy at the end of a chain is only bonded on one side.
    connection_atoms_by_name: dict[str, set[str]] = {}
    for (_chain, _res_id, res_name), atoms in cross_residue_atoms.items():
        connection_atoms_by_name.setdefault(res_name, set()).update(atoms)
    partners_by_name: dict[str, dict[str, set[tuple]]] = {}
    for (_chain, _res_id, res_name), by_atom in cross_residue_partners.items():
        for atom, far_side in by_atom.items():
            partners_by_name.setdefault(res_name, {}).setdefault(atom, set()).update(
                far_side
            )

    for start in unknown_starts:
        res_name = atom_array.res_name[start].strip()

        if res_name in seen:
            continue
        seen.add(res_name)

        connections = connection_atoms_by_name.get(res_name, ())
        sub = _representative_instance(
            atom_array, residue_starts, start, connections, heavy_counts
        )
        component_type = get_chem_comp_type(res_name, chem_comp_types) or "UNKNOWN"
        covalently_linked = res_name in covalently_linked_names

        logger.info(
            "Detected non-standard residue %s (declared type: %s, polymer "
            "entity: %s, %d atoms)",
            res_name,
            component_type,
            "unstated" if polymer_names is None else res_name in polymer_names,
            len(sub),
        )

        # Chemistry (bond orders) and charges are intentionally *not* trusted
        # from the input here: the unified path re-derives a SMILES from the
        # atoms and generates OpenBabel MMFF94 charges. Detection only needs
        # connectivity, atom names, and coordinates.
        results.append(
            NonStandardResidueInfo(
                res_name=res_name,
                component_type=component_type,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
                atom_array=sub,
                covalently_linked=covalently_linked,
                connection_atom_names=frozenset(
                    connection_atoms_by_name.get(res_name, ())
                ),
                in_polymer_entity=(
                    None if polymer_names is None else res_name in polymer_names
                ),
                connection_partners={
                    atom: frozenset(far_side)
                    for atom, far_side in partners_by_name.get(res_name, {}).items()
                },
                chemistry_problem=_isolated_atom_charge_problem(
                    atom_array, sub, covalently_linked=covalently_linked
                ),
            )
        )

    return results


def _representative_instance(
    atom_array, residue_starts, start, connection_atoms=(), heavy_counts=None
):
    """The copy of this residue to describe the type from.

    Prefer copies carrying all connection atoms, then the fuller heavy-atom
    inventory, then resolved coordinates. A fully observed internal sugar has
    lost its anomeric leaving oxygen; using it ahead of a fuller terminal copy
    would omit an atom the shared base type must describe.
    """
    wanted = set(connection_atoms)

    ends = np.append(residue_starts[1:], atom_array.array_length())
    if heavy_counts is None:
        heavy_counts = np.add.reduceat(
            ~np.isin(atom_array.element, ("H", "D")), residue_starts
        )
    copies = np.flatnonzero(
        atom_array.res_name[residue_starts] == atom_array.res_name[start]
    )
    copies = copies[np.argsort(-heavy_counts[copies], kind="stable")]
    fallback, best_count = None, -1
    for i in copies:
        if fallback is not None and heavy_counts[i] < best_count:
            break
        candidate = atom_array[residue_starts[i] : ends[i]]
        if not wanted <= set(candidate.atom_name):
            continue
        if np.isfinite(candidate.coord).all():
            return candidate.copy()
        if fallback is None:
            fallback, best_count = candidate, heavy_counts[i]
    if fallback is not None:
        return fallback.copy()
    # Slicing an AtomArray yields a view; the caller owns what it gets back.
    return atom_array[start : ends[np.searchsorted(residue_starts, start)]].copy()


def with_resolved_coordinates(atom_array, res_name: str, use_ccd: bool):
    """Resolve scratch geometry for parameter generation from declared chemistry.

    Input-scoped AtomWorks templates take precedence over the bundled CCD.
    Partially observed components require three anchors for alignment. A wholly
    unresolved component returns None because template geometry cannot place it
    in the pose. Supplied coordinates and the caller's NaNs remain unchanged.
    """
    unresolved = np.isnan(atom_array.coord).any(axis=-1)
    if not unresolved.any():
        return atom_array
    if unresolved.all() or not use_ccd:
        return None
    template = getattr(atom_array, "_custom_ccd_registry", {}).get(res_name.upper())
    from atomworks.io.utils.ccd import atom_array_from_ccd_code, custom_ccd_residues

    try:
        with custom_ccd_residues({res_name: template} if template is not None else {}):
            component = atom_array_from_ccd_code(res_name, ccd_mirror_path=None)
        indices = {str(name): i for i, name in enumerate(component.atom_name)}
        ideal = np.full_like(atom_array.coord, np.nan)
        matched = np.array([str(name) in indices for name in atom_array.atom_name])
        ideal[matched] = component.coord[
            [indices[str(name)] for name in atom_array.atom_name[matched]]
        ]
    except (ValueError, KeyError):
        return None
    if not np.isfinite(ideal[unresolved]).all():
        return None
    anchors = ~unresolved & np.isfinite(ideal).all(axis=1)
    if np.count_nonzero(anchors) < 3:
        return None
    try:
        _, transform = struc.superimpose(atom_array.coord[anchors], ideal[anchors])
        ideal = transform.apply(ideal)
    except ValueError:
        return None
    filled = atom_array.copy()
    filled.coord[unresolved] = ideal[unresolved]
    return filled


def is_polymer_linking_component_type(component_type: Optional[str]) -> bool:
    """Whether a declared chemical-component type denotes a polymer residue.

    Returns True for modified amino acids, nucleotides, and saccharides
    (e.g. "L-PEPTIDE LINKING", "DNA LINKING", "D-SACCHARIDE"), which are
    expected to be covalently attached to a chain. Returns False for
    "NON-POLYMER" small-molecule ligands and where nothing is declared.
    """
    if not component_type:
        return False
    return "LINKING" in component_type or "SACCHARIDE" in component_type


def polymer_entity_residues(atom_array: struc.AtomArray) -> Optional[frozenset[str]]:
    """Names of the residues the input file places in a polymer entity.

    An mmCIF says which of its entities are polymers, and that is what
    separates a chain from a ligand, a glycan or solvent -- including for a
    terminal cap, which is typed NON-POLYMER despite being part of the chain.
    :func:`tmol.io.atom_array_from_cif` records it per atom.

    Failing that, ``label_seq_id`` is read as a proxy: a polymer entity numbers
    its residues along its sequence. It is only a proxy -- a generated file may
    number a ligand that belongs to no sequence -- so it is used only where the
    entities themselves are not recorded. Returns None when neither is.
    """
    categories = atom_array.get_annotation_categories()
    names = atom_array.res_name.astype(str)
    for annotation in ("tmol_polymer_entity", "is_polymer"):
        if annotation in categories:
            flag = atom_array.get_annotation(annotation).astype(bool)
            return frozenset(str(n).strip() for n in names[flag])
    if "label_seq_id" not in categories:
        return None
    seq = atom_array.get_annotation("label_seq_id").astype(str)
    numbered = np.char.strip(seq) != "."
    if not numbered.any():
        return None
    return frozenset(str(n).strip() for n in names[numbered])


def _cross_residue_bond_atoms(
    atom_array: struc.AtomArray,
) -> tuple[dict[tuple, frozenset[str]], dict[tuple, dict[str, frozenset[tuple]]]]:
    """Atoms of each residue instance that bond to a different residue.

    Returns the atoms alongside, for each of them, the (residue name, atom
    name) pairs on the far side of the bond. Which residue an attachment lands
    on is what separates a chain link from a conjugation.

    Keyed by ``(chain_id, res_id, res_name)``. These atoms are where the
    residue's polymer connections attach, which is what says where its backbone
    ends; classifying a backbone from atom names instead mistakes a residue
    linked through a sidechain carbon for one linked through its carbonyl.

    A "different residue" is identified by (chain_id, res_id, res_name) — so
    distinct instances of the same residue name (e.g. two NAGs in a glycan
    chain) also count as cross-residue bonds. Used to flag ligands that are
    covalently attached to a polymer or to other ligand instances.

    Use the supplied bond graph, independent of coordinates and component type.
    Readers own polymer/link inference. Preparation must not turn close contacts
    into covalent links or generate unused conjugation types from those contacts.
    Supply explicit connectivity before preparation when a file omits link data.
    """
    chain_ids = atom_array.chain_id if hasattr(atom_array, "chain_id") else None
    res_ids = atom_array.res_id
    res_names = atom_array.res_name

    linked: dict[tuple, set[str]] = {}
    partners: dict[tuple, dict[str, set[tuple[str, str]]]] = {}
    atom_names = atom_array.atom_name

    def _key(idx: int) -> tuple:
        chain = chain_ids[idx] if chain_ids is not None else None
        return (chain, res_ids[idx], res_names[idx].strip())

    def _record(idx: int, partner: Optional[int] = None) -> None:
        name = str(atom_names[idx]).strip()
        linked.setdefault(_key(idx), set()).add(name)
        if partner is None:
            return
        partners.setdefault(_key(idx), {}).setdefault(name, set()).add(
            (res_names[partner].strip(), str(atom_names[partner]).strip())
        )

    def _spans_residues(a: int, b: int) -> bool:
        """Whether atoms ``a`` and ``b`` belong to different residues."""
        same_chain = chain_ids is None or chain_ids[a] == chain_ids[b]
        if (
            same_chain
            and res_ids[a] == res_ids[b]
            and res_names[a] == res_names[b]
            and atom_array.ins_code[a] == atom_array.ins_code[b]
        ):
            return False
        return True

    if atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0:
        for a, b, _ in atom_array.bonds.as_array():
            a, b = int(a), int(b)
            if _spans_residues(a, b):
                _record(a, b)
                _record(b, a)

    return (
        {key: frozenset(names) for key, names in linked.items()},
        {
            key: {atom: frozenset(seen) for atom, seen in by_atom.items()}
            for key, by_atom in partners.items()
        },
    )


def _residue_names_with_cross_residue_bonds(
    atom_array: struc.AtomArray,
) -> frozenset[str]:
    """Names of the residues that have at least one bond to a different residue."""
    return frozenset(
        res_name
        for _chain, _res_id, res_name in _cross_residue_bond_atoms(atom_array)[0]
    )
