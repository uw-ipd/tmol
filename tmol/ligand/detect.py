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
import biotite.structure.info.ccd as ccd
import numpy as np
from rdkit import Chem
from rdkit.Chem import RWMol

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
    ccd_type: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray = attr.ib(eq=False, hash=False)
    atom_array: struc.AtomArray = attr.ib(eq=False, hash=False)
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

    Low-level reader retained for the DUD-80 SMILES->params parity harness
    (it reads both the OpenBabel-generated and Rosetta ground-truth mol2
    files). Preserves Tripos aromatic flags, atom-type subtypes, and per-atom
    partial charges, avoiding lossy rdkit<->biotite round-trips.
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
        covalently_linked=False,
        partial_charges=authoritative_q,
        skip_protonation=authoritative_q is not None,
        original_single_bonds=original_single_bonds or None,
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

    This never builds a biotite atom-array from an RDKit embedding and never
    recomputes MMFF on a reconstructed graph — the OpenBabel MMFF94 charges
    flow through untouched (``skip_protonation`` / authoritative charges are
    set by the mol2 reader), so fused-ring aromatics keep correct charges.

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

        logger.info(
            "Detected non-standard residue %s (CCD type: %s, %d atoms)",
            res_name,
            ccd_type,
            len(sub),
        )

        # Chemistry (bond orders) and charges are intentionally *not* trusted
        # from the input here: the unified path re-derives a SMILES from the
        # atoms and generates OpenBabel MMFF94 charges. Detection only needs
        # connectivity, atom names, and coordinates.
        results.append(
            NonStandardResidueInfo(
                res_name=res_name,
                ccd_type=ccd_type,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
                atom_array=sub,
                covalently_linked=res_name in covalently_linked_names,
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
