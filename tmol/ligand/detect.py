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
from tmol.ligand.cif_normalization import (
    infer_paired_mol2_path,
    repaired_cif_path_from_mol2,
)
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
        original_single_bonds: Optional set of ``frozenset({name_a, name_b})``
            pairs that the source mol2 records as literal single (order ``'1'``)
            bonds, keyed by disambiguated atom name. ``build_chi_topology`` uses
            these to honor the mol2 bond order (Rosetta-faithful) instead of
            RDKit's post-kekulization order. Only set on the mol2 / SMILES-via-
            mol2 paths; ``None`` elsewhere.
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
    original_single_bonds: Optional[frozenset[frozenset[str]]] = None


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


def _mol2_charge_model(mol2_path: Path) -> str:
    """Return the Tripos charge model line from a mol2 file (e.g. ``GASTEIGER``)."""
    return _mol2_charge_model_from_text(mol2_path.read_text())


def _mol2_single_bond_ids(mol2_text: str) -> frozenset[frozenset[int]]:
    """Return the set of single (order ``'1'``) bonds from a mol2 BOND section.

    Each bond is returned as a ``frozenset`` of its two 1-based TRIPOS atom
    ids. Only bonds whose TRIPOS ``bond_type`` is literally ``'1'`` are
    included — aromatic (``ar``), amide (``am``), double (``2``), etc. are
    excluded.

    RDKit's sanitize/kekulize step promotes some mol2 single bonds (e.g. a
    ``C.ar``-``N.pl3`` written as ``1``) to AROMATIC/DOUBLE, which then makes
    :func:`tmol.ligand.chi_topology.build_chi_topology` skip them as
    ``border > 1``. Rosetta's ``mol2genparams`` reads the literal mol2 order
    instead, so honoring the source ``'1'`` here restores parity. Returns an
    empty set if no BOND section is present.
    """
    single_bonds: set[frozenset[int]] = set()
    in_bond = False
    for line in mol2_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            in_bond = stripped.startswith("@<TRIPOS>BOND")
            continue
        if not in_bond or not stripped:
            continue
        tokens = stripped.split()
        # TRIPOS BOND line: bond_id atom1 atom2 bond_type [status_bits]
        if len(tokens) < 4 or tokens[3] != "1":
            continue
        try:
            a1, a2 = int(tokens[1]), int(tokens[2])
        except ValueError:
            continue
        single_bonds.add(frozenset((a1, a2)))
    return frozenset(single_bonds)


def _charge_model_is_authoritative(model: str) -> bool:
    """True only when a Tripos charge model is a trusted force-field model.

    PLI fixtures and legacy mol2s use Gasteiger; MMFF94/AM1-BCC/etc. are OK.
    """
    if not model:
        return False
    if model == "GASTEIGER":
        return False
    return True


def _mol2_partial_charges_are_authoritative(mol2_path: Path) -> bool:
    """True only when mol2 partial charges are a trusted force-field model."""
    return _charge_model_is_authoritative(_mol2_charge_model(mol2_path))


def nonstandard_residue_info_from_mol2(
    mol2_path: str | Path,
    res_name: str | None = None,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand Mol2 file.

    This path preserves Tripos aromatic flags, atom-type subtypes, and
    per-atom partial charges when present, avoiding lossy rdkit<->biotite
    round-trips.
    """
    from tmol.ligand.openbabel_compat import (
        OpenBabelUnavailableError,
        obabel_read_mol2,
    )

    path = Path(mol2_path)
    mol2_text = path.read_text()
    mol = Chem.MolFromMol2File(
        str(path),
        sanitize=False,
        removeHs=False,
        cleanupSubstructures=False,
    )
    if mol is None:
        try:
            mol = obabel_read_mol2(path)
        except OpenBabelUnavailableError:
            mol = None
        if mol is None:
            raise ValueError(
                f"Could not parse Mol2 file: {path} "
                "(RDKit MolFromMol2File failed; OpenBabel fallback also "
                "failed or is not installed)"
            )
        logger.info("Used OpenBabel fallback to parse mol2 %s", path)
    return _nonstandard_residue_info_from_mol2_mol(
        mol,
        _mol2_charge_model_from_text(mol2_text),
        res_name,
        source=str(path),
        single_bond_ids=_mol2_single_bond_ids(mol2_text),
    )


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
    from tmol.ligand.openbabel_compat import (
        OpenBabelUnavailableError,
        obabel_read_mol2_block,
    )

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
    return _nonstandard_residue_info_from_mol2_mol(
        mol,
        _mol2_charge_model_from_text(mol2_block),
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
    error messages. ``single_bond_ids`` is the set of 1-based TRIPOS atom-id pairs
    the source records as literal single bonds (see :func:`_mol2_single_bond_ids`);
    these are mapped to disambiguated atom names and stored as
    ``original_single_bonds`` so the CHI classifier can honor the mol2 bond order.
    """
    from tmol.ligand.atom_typing import sanitize_tolerant

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

    use_input_charges = has_full_partial_charges and _charge_model_is_authoritative(
        charge_model
    )
    authoritative_q = partial_charges if use_input_charges else None

    # Map literal mol2 single bonds (1-based TRIPOS ids) to disambiguated atom
    # names. RDKit index i corresponds to TRIPOS atom id i+1 (the reader keeps
    # mol2 atom order), the same correspondence used for names/coords above.
    original_single_bonds: frozenset[frozenset[str]] = frozenset(
        frozenset((disambiguated_names[i - 1], disambiguated_names[j - 1]))
        for i, j in (tuple(pair) for pair in single_bond_ids)
        if 1 <= i <= n_atoms and 1 <= j <= n_atoms
    )

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
        original_single_bonds=original_single_bonds or None,
    )


def nonstandard_residue_info_from_pdb(
    pdb_path: str | Path,
    res_name: str | None = None,
    *,
    perceive_bond_orders: bool = True,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand PDB file via OpenBabel.

    PDB does not carry bond orders. OpenBabel's ``ConnectTheDots`` +
    ``PerceiveBondOrders`` are used to derive chemistry from geometry,
    matching the standard ligand-prep practice in Rosetta/AMBER. The
    original PDB atom names are preserved.

    Args:
        pdb_path: Path to a PDB file containing a single ligand. ATOM
            and HETATM records are read; other records are ignored.
        res_name: Override for the 3-letter residue name. Falls back to
            the PDB residue name if present, else ``"LG1"``.
        perceive_bond_orders: If True (default), OB infers bond orders
            from geometry. If False, all bonds are emitted as single.

    Returns:
        A :class:`NonStandardResidueInfo` ready for
        :func:`prepare_single_ligand`.
    """
    from tmol.ligand.atom_typing import sanitize_tolerant
    from tmol.ligand.openbabel_compat import (
        OpenBabelUnavailableError,
        _import_openbabel,
        _obmol_to_rdkit_mol,
    )

    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB file not found: {path}")

    try:
        openbabel, pybel = _import_openbabel()
    except OpenBabelUnavailableError as exc:
        raise OpenBabelUnavailableError(
            "prepare_ligand_from_pdb requires the optional 'openbabel' "
            "Python package because PDB files lack explicit bond orders "
            "and need geometry-based bond perception. "
            "Install with `pip install openbabel-wheel`."
        ) from exc

    try:
        pymol = next(pybel.readfile("pdb", str(path)))
    except StopIteration:
        raise ValueError(f"No molecules in PDB file: {path}")
    obmol = pymol.OBMol
    if perceive_bond_orders:
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()

    residue = obmol.GetResidue(0) if obmol.NumResidues() > 0 else None

    n_atoms = obmol.NumAtoms()
    atom_names: list[str] = []
    elements: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i, atom in enumerate(openbabel.OBMolAtomIter(obmol)):
        name = ""
        if residue is not None:
            name = (residue.GetAtomID(atom) or "").strip()
        symbol = openbabel.GetSymbol(atom.GetAtomicNum()) or "X"
        if not name:
            name = f"{symbol}{i + 1}"
        atom_names.append(name)
        elements.append(symbol)
        coords[i] = (atom.GetX(), atom.GetY(), atom.GetZ())

    mol = _obmol_to_rdkit_mol(obmol, sanitize=False)
    if mol is None:
        raise ValueError(f"Could not convert OpenBabel-parsed PDB to RDKit Mol: {path}")
    sanitize_tolerant(mol)
    if mol.GetNumAtoms() != n_atoms:
        raise ValueError(
            f"Atom count mismatch reading PDB {path}: "
            f"OBMol={n_atoms} RDKit={mol.GetNumAtoms()}"
        )

    inferred_res_name = res_name
    if inferred_res_name is None and residue is not None:
        candidate = (residue.GetName() or "").strip()
        if candidate:
            inferred_res_name = candidate
    if inferred_res_name is None:
        inferred_res_name = "LG1"

    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.atom_name = np.array(atom_names, dtype="U16")
    atom_array.element = np.array(elements, dtype="U4")
    atom_array.res_name = np.array([inferred_res_name] * n_atoms, dtype="U8")
    atom_array.chain_id = np.array(["A"] * n_atoms, dtype="U4")
    atom_array.res_id = np.array([1] * n_atoms, dtype=np.int32)
    atom_array.hetero = np.array([True] * n_atoms, dtype=bool)

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

    return NonStandardResidueInfo(
        res_name=inferred_res_name,
        ccd_type="UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(elements),
        coords=coords,
        atom_array=atom_array,
        ccd_smiles=None,
        covalently_linked=False,
        partial_charges=None,
        skip_protonation=False,
    )


def nonstandard_residue_info_from_smiles(
    smiles: str,
    res_name: str | None = None,
    *,
    add_hydrogens: bool = True,
    seed: int = 0xC0FFEE,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a SMILES string.

    RDKit parses the SMILES, adds explicit hydrogens, and embeds a 3D
    conformer (UFF-optimized). If RDKit's SMILES parser returns ``None``
    on input that OpenBabel can read, the parse step falls back to OB.

    Atom names are synthesized as ``<element><1-based-index>`` since
    SMILES carries no atom labels.

    Args:
        smiles: SMILES string for the ligand.
        res_name: Three-letter residue name (default ``"LG1"``).
        add_hydrogens: If True (default), explicit Hs are added before
            embedding so 3D geometry is chemically sensible.
        seed: RNG seed for RDKit ``EmbedMolecule`` (deterministic by
            default).
    """
    from rdkit.Chem import AllChem as _AllChem

    from tmol.ligand.atom_typing import sanitize_tolerant
    from tmol.ligand.openbabel_compat import (
        OpenBabelUnavailableError,
        obabel_read_smiles,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = obabel_read_smiles(smiles)
        except OpenBabelUnavailableError:
            mol = None
        if mol is None:
            raise ValueError(
                f"Could not parse SMILES: {smiles!r} "
                "(RDKit MolFromSmiles failed; OpenBabel fallback also "
                "failed or is not installed)"
            )
        logger.info("Used OpenBabel fallback to parse SMILES %r", smiles)

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    if mol.GetNumConformers() == 0:
        embed_result = _AllChem.EmbedMolecule(mol, randomSeed=seed)
        if embed_result < 0:
            # RDKit embedding failed; try OB-based 3D generation.
            try:
                mol = obabel_read_smiles(smiles, generate_3d=True, minimize=True)
            except OpenBabelUnavailableError:
                mol = None
            if mol is None:
                raise ValueError(
                    f"Could not embed 3D coordinates for SMILES {smiles!r}"
                )
        else:
            try:
                _AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                logger.debug(
                    "UFF optimization failed for SMILES %r; using raw embed",
                    smiles,
                    exc_info=True,
                )

    sanitize_tolerant(mol)

    if mol.GetNumConformers() == 0:
        raise ValueError(f"3D generation produced no conformer for SMILES {smiles!r}")

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    inferred_res_name = res_name or "LG1"

    atom_names: list[str] = []
    elements: list[str] = []
    elem_counts: dict[str, int] = {}
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        elem_counts[symbol] = elem_counts.get(symbol, 0) + 1
        atom_names.append(f"{symbol}{elem_counts[symbol]}")
        elements.append(symbol)
        p = conf.GetAtomPosition(i)
        coords[i] = (float(p.x), float(p.y), float(p.z))

    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.atom_name = np.array(atom_names, dtype="U16")
    atom_array.element = np.array(elements, dtype="U4")
    atom_array.res_name = np.array([inferred_res_name] * n_atoms, dtype="U8")
    atom_array.chain_id = np.array(["A"] * n_atoms, dtype="U4")
    atom_array.res_id = np.array([1] * n_atoms, dtype=np.int32)
    atom_array.hetero = np.array([True] * n_atoms, dtype=bool)

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

    return NonStandardResidueInfo(
        res_name=inferred_res_name,
        ccd_type="UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(elements),
        coords=coords,
        atom_array=atom_array,
        ccd_smiles=smiles,
        covalently_linked=False,
        partial_charges=None,
        skip_protonation=False,
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
    changed = False
    for atom in mol.GetAtoms():
        if (
            atom.GetSymbol() == "O"
            and atom.GetDegree() == 1
            and atom.GetTotalNumHs() == 0
            and atom.GetFormalCharge() == 0
            and atom.GetNumRadicalElectrons() > 0
        ):
            atom.SetFormalCharge(-1)
            atom.SetNumRadicalElectrons(0)
            changed = True
    if not changed:
        return smiles
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        logger.warning(
            "Radical-oxygen normalization failed to sanitize SMILES %r; " "using input",
            smiles,
        )
        return smiles
    return Chem.MolToSmiles(mol)


def _dimorphite_protonate_smiles(
    smiles: str, ph: float = 7.4, precision: float = 0.1
) -> str:
    """Return the SMILES pKa-protonated at ``ph`` via Dimorphite-DL.

    Takes the first protonation variant (matching the reference ligand-prep
    protocol). Falls back to the input SMILES if RDKit cannot parse it or
    Dimorphite produces no variant.
    """
    from tmol.ligand.dimorphite_dl import protonate_mol_variants

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


def nonstandard_residue_info_from_smiles_via_mol2(
    smiles: str,
    res_name: str | None = None,
    *,
    ph: float = 7.4,
    protonate: bool = True,
    conformer_search: bool = True,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a SMILES via the mol2 route.

    Implements the canonical ligand-prep protocol end to end:

    0. normalize bare radical oxygens (``[O]`` -> ``[O-]``) so source
       carboxylate/sulfonate notation has a well-defined charge,
    1. optionally pKa-protonate the SMILES with Dimorphite-DL (``protonate``),
    2. generate a 3D mol2 with MMFF94 partial charges via OpenBabel (kept
       in memory as a string — no temp file), then
    3. read that mol2 with :func:`nonstandard_residue_info_from_mol2_block`.

    Unlike :func:`nonstandard_residue_info_from_smiles`, this never builds a
    biotite atom-array from an RDKit embedding and never recomputes MMFF on a
    reconstructed graph — the OpenBabel MMFF94 charges flow through untouched
    (``skip_protonation`` / authoritative charges are set by the mol2 reader),
    so fused-ring aromatics keep correct charges.

    Args:
        smiles: Ligand SMILES string.
        res_name: Three-letter residue name (default inferred / ``"LG1"``).
        ph: Target pH for the Dimorphite protonation step.
        protonate: When ``True`` (default) run Dimorphite on ``smiles`` first;
            set ``False`` to pin an already-protonated SMILES verbatim.
        conformer_search: When ``True`` (default) run a rotor conformer search
            during the 3D mol2 generation (matching the reference pipeline);
            set ``False`` for faster single-conformer generation.

    Raises:
        OpenBabelUnavailableError: If the ``openbabel`` package is missing
            (this path requires it for the SMILES -> mol2 conversion).
        ValueError: If OpenBabel cannot build a charged mol2 for ``smiles``.
    """
    from tmol.ligand.openbabel_compat import obabel_smiles_to_mol2_block

    smiles = _normalize_radical_oxygens(smiles)
    prep_smiles = _dimorphite_protonate_smiles(smiles, ph) if protonate else smiles
    mol2_block = obabel_smiles_to_mol2_block(
        prep_smiles, conformer_search=conformer_search
    )
    return nonstandard_residue_info_from_mol2_block(mol2_block, res_name=res_name)


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


def _cif_read_path_with_optional_mol2_repair(
    path: Path,
    *,
    paired_mol2_path: str | Path | None,
    res_name: str | None,
    repair_invalid_bonds: bool,
) -> tuple[Path, Path | None]:
    """Return the CIF path to read and an optional temp file to delete."""
    cif_path_to_read = path
    temporary_regenerated_path: Path | None = None
    if not repair_invalid_bonds:
        return cif_path_to_read, temporary_regenerated_path

    inferred_mol2 = (
        Path(paired_mol2_path)
        if paired_mol2_path is not None
        else infer_paired_mol2_path(path)
    )
    if inferred_mol2 is None or not inferred_mol2.is_file():
        return cif_path_to_read, temporary_regenerated_path

    regen_res_name = res_name or "LG1"
    cif_path_to_read, audit, regenerated = repaired_cif_path_from_mol2(
        path,
        inferred_mol2,
        res_name=regen_res_name,
    )
    if regenerated:
        temporary_regenerated_path = cif_path_to_read
        logger.warning(
            "Regenerated CIF bonds from paired MOL2 before loading %s; "
            "missing=%d extra=%d note=%s",
            path,
            len(audit.missing_in_cif),
            len(audit.extra_in_cif),
            audit.note or "none",
        )
    return cif_path_to_read, temporary_regenerated_path


def _partial_charges_from_atom_site(
    atom_site,
    atom_names: list[str],
) -> Optional[dict[str, float]]:
    """Extract a per-atom partial-charge map from a CIF ``atom_site`` category.

    Returns ``None`` unless the ``partial_charge`` column is present, finite, and
    has exactly one value per atom name.
    """
    if "partial_charge" not in atom_site:
        return None
    try:
        vals = atom_site["partial_charge"].as_array(float)
        vals = np.asarray(vals, dtype=np.float64)
        if vals.shape[0] == len(atom_names) and np.isfinite(vals).all():
            return {name: float(q) for name, q in zip(atom_names, vals, strict=False)}
    except (TypeError, ValueError):
        return None
    return None


def _apply_cif_atom_array_annotations(arr: struc.AtomArray, atom_site) -> None:
    """Copy tmol custom CIF columns (``tmol_aromatic``, ``tmol_source_subtype``)
    onto an AtomArray as annotations when present."""
    if "tmol_aromatic" in atom_site:
        aromatic_vals = atom_site["tmol_aromatic"].as_array()
        arr.set_annotation(
            "tmol_aromatic",
            np.array(
                [str(v).strip().upper() == "Y" for v in aromatic_vals], dtype=bool
            ),
        )
    if "tmol_source_subtype" in atom_site:
        arr.set_annotation(
            "tmol_source_subtype",
            np.array([str(v) for v in atom_site["tmol_source_subtype"].as_array()]),
        )


def _attach_chem_comp_bonds(arr: struc.AtomArray, cif, atom_names: list[str]) -> None:
    """Rebuild ``arr.bonds`` from the CIF ``chem_comp_bond`` table.

    Bonds referencing atom names not present in ``atom_names`` are skipped; if no
    valid bonds remain the existing bond table is left unchanged.
    """
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
    """Resolve the residue name, preferring an explicit override, then the CIF
    ``label_comp_id`` column, then the AtomArray, falling back to ``"LG1"``."""
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
    *,
    paired_mol2_path: str | Path | None = None,
    repair_invalid_bonds: bool = True,
) -> NonStandardResidueInfo:
    """Construct ``NonStandardResidueInfo`` from a ligand CIF file.

    Preserves custom per-atom fields (``partial_charge``, ``tmol_aromatic``,
    ``tmol_source_subtype``) and rebuilds the bond table directly from
    ``_chem_comp_bond`` when present.
    """
    import biotite.structure.io.pdbx as pdbx

    path = Path(cif_path)
    cif_path_to_read, temporary_regenerated_path = (
        _cif_read_path_with_optional_mol2_repair(
            path,
            paired_mol2_path=paired_mol2_path,
            res_name=res_name,
            repair_invalid_bonds=repair_invalid_bonds,
        )
    )
    try:
        cif = pdbx.CIFFile.read(str(cif_path_to_read))
        arr = pdbx.get_structure(
            cif, model=1, include_bonds=True, extra_fields=["charge"]
        )
        if isinstance(arr, struc.AtomArrayStack):
            arr = arr[0]

        atom_site = cif.block["atom_site"]
        atom_names = [str(v) for v in atom_site["label_atom_id"].as_array()]
        res_name = _resolve_cif_res_name(atom_site, arr, res_name)
        arr.res_name = np.array([res_name] * len(arr), dtype=arr.res_name.dtype)

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
            ccd_smiles=_atom_array_to_smiles(
                arr,
                source=str(path),
                res_name=res_name,
            ),
            covalently_linked=False,
            partial_charges=partial_charges,
            skip_protonation=partial_charges is not None,
        )
    finally:
        if temporary_regenerated_path is not None:
            try:
                temporary_regenerated_path.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Failed to clean temporary regenerated CIF %s",
                    temporary_regenerated_path,
                    exc_info=True,
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
