"""RDKit construction utilities vendored from atomworks.

This module is a faithful, self-contained port of the relevant
``atom_array_to_rdkit`` stack from atomworks
(``src/atomworks/io/tools/rdkit.py``), used by tmol to convert a biotite
``AtomArray`` ligand into an RDKit molecule (and thence a SMILES string).

Provenance:
    Source: atomworks (https://github.com/baker-laboratory/atomworks), uw-ipd.
    The transition-metal-complex bond-perception fallback lives in the
    sibling ``xyz2mol_tm`` package (vendored verbatim from
    https://github.com/jensengroup/xyz2mol_tm, MIT licensed).

Differences from the atomworks original (intentional, to drop heavy deps):
    - The Chemical Component Dictionary source for :func:`ccd_code_to_rdkit`
      is biotite's bundled CCD (``biotite.structure.info``) rather than an
      external CCD mirror.
    - The ``timer.timeout`` decorator is replaced by a small signal-based
      timeout context manager.
    - atomworks-internal helpers (``exists``, ``not_isin``,
      ``ta.remove_hydrogens``) and constants are inlined here.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import signal
from functools import cache, wraps
from pathlib import Path
from typing import Callable, Final, Generator, Literal

import numpy as np
from biotite import structure as struc
from biotite.structure import AtomArray
from biotite.structure.info import residue as _biotite_residue
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.MolStandardize import rdMolStandardize

from tmol.ligand.external.xyz2mol_tm import get_tmc_mol

Mol = Chem.Mol

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants (inlined from atomworks.constants)
# ----------------------------------------------------------------------------
# fmt: off
METAL_ELEMENTS: Final[frozenset[str]] = frozenset(map(str.upper, [
    "Li", "Na", "K", "Rb", "Cs", "Fr",                          # alkali metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",                         # alkaline earth metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # 3d transition metals
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",  # 4d transition metals
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",        # 5d transition metals
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",                   # post-transition metals
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # lanthanides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",   # actinides
]))
# fmt: on

HYDROGEN_LIKE_SYMBOLS: Final[tuple[str, ...]] = ("H", "H2", "D", "T")

BIOTITE_DEFAULT_ANNOTATIONS: Final[tuple[str, ...]] = (
    "chain_id",
    "res_id",
    "res_name",
    "atom_name",
    "hetero",
    "element",
)

PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL: Final[dict[str, str]] = {
    "D": "H",
    "T": "H",
}


# ----------------------------------------------------------------------------
# RDKit <-> biotite bond-type maps (from atomworks.io.tools.rdkit)
# ----------------------------------------------------------------------------
BIOTITE_BOND_TYPE_TO_RDKIT: Final[
    dict[struc.bonds.BondType, tuple[Chem.BondType, bool]]
] = {
    struc.bonds.BondType.ANY: (Chem.BondType.UNSPECIFIED, False),
    struc.bonds.BondType.SINGLE: (Chem.BondType.SINGLE, False),
    struc.bonds.BondType.DOUBLE: (Chem.BondType.DOUBLE, False),
    struc.bonds.BondType.TRIPLE: (Chem.BondType.TRIPLE, False),
    struc.bonds.BondType.QUADRUPLE: (Chem.BondType.QUADRUPLE, False),
    struc.bonds.BondType.COORDINATION: (Chem.BondType.DATIVE, False),
    # NOTE: kekulized aromatics are mapped to single/double/triple (not
    #       Chem.BondType.AROMATIC) so the kekulized PDB bond order is preserved.
    struc.bonds.BondType.AROMATIC_SINGLE: (Chem.BondType.SINGLE, True),
    struc.bonds.BondType.AROMATIC_DOUBLE: (Chem.BondType.DOUBLE, True),
    struc.bonds.BondType.AROMATIC_TRIPLE: (Chem.BondType.TRIPLE, True),
    # Generic aromatic bonds (no kekulized order, e.g. a CIF ``_chem_comp_bond``
    # with ``pdbx_aromatic_flag = Y``) are added as RDKit aromatic bonds; the
    # atom aromatic flags below + RDKit sanitization recover the Kekulé form.
    struc.bonds.BondType.AROMATIC: (Chem.BondType.AROMATIC, True),
}


# ----------------------------------------------------------------------------
# Small helpers (inlined from atomworks.common / atom_array transforms)
# ----------------------------------------------------------------------------
def exists(x) -> bool:
    """Return True if ``x`` is not None."""
    return x is not None


def not_isin(element, test_elements) -> np.ndarray:
    """Boolean mask of ``element`` entries that are NOT in ``test_elements``."""
    return ~np.isin(element, test_elements)


def _remove_hydrogens(atom_array: AtomArray) -> AtomArray:
    """Drop hydrogen-like atoms from an AtomArray (atomworks ``ta.remove_hydrogens``)."""
    return atom_array[not_isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)]


@contextlib.contextmanager
def _suppress_stderr_fd() -> Generator[None, None, None]:
    """Suppress stderr at the file-descriptor level.

    Silences YAeHMOP warnings emitted directly to fd 2 by ``rdDetermineBonds``.
    """
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stderr)


class _TimeoutError(Exception):
    """Raised when a signal-based timeout fires."""


@contextlib.contextmanager
def _signal_timeout(seconds: int) -> Generator[None, None, None]:
    """Signal-based timeout (main thread only); replaces atomworks ``timer.timeout``.

    On non-main threads (where SIGALRM is unavailable) this is a no-op.
    """
    try:
        previous = signal.getsignal(signal.SIGALRM)
    except (ValueError, AttributeError):
        # Not in the main thread / platform without SIGALRM: run without a timeout.
        yield
        return

    def _handler(signum, frame):
        raise _TimeoutError(f"Operation timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@cache
def element_to_atomic_number(element: str) -> int:
    """Convert an element symbol (e.g. ``'C'``, ``'D'``) to its atomic number."""
    element = PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL.get(element, element)
    return Chem.GetPeriodicTable().GetAtomicNumber(element.capitalize())


def preserve_annotations(func: Callable[..., Mol]) -> Callable[..., Mol]:
    """Decorator that copies the ``_annotations`` attribute across a Mol-returning call."""

    @wraps(func)
    def wrapped(*args, **kwargs) -> Mol:
        if "mol" in kwargs:
            mol = kwargs["mol"]
        else:
            mol = next(arg for arg in args if isinstance(arg, Mol))

        if hasattr(mol, "_annotations"):
            annotations = mol._annotations
            new_mol = func(*args, **kwargs)
            new_mol._annotations = annotations
        else:
            new_mol = func(*args, **kwargs)
        return new_mol

    return wrapped


# ----------------------------------------------------------------------------
# ChEMBL-style normalization (from atomworks.io.tools.rdkit)
# ----------------------------------------------------------------------------
class ChEMBLNormalizer:
    """Normalize an RDKit molecule like the ChEMBL structure pipeline does.

    Useful for rescuing molecules that fail RDKit sanitization on their own.

    Reference:
        ChEMBL Structure Pipeline standardizer.
    """

    def __init__(self) -> None:
        with open(Path(__file__).parent / "chembl_transformations.smirks") as f:
            self._normalization_transforms = f.read()
        self._normalizer_params = rdMolStandardize.CleanupParameters()
        self._normalizer = rdMolStandardize.NormalizerFromData(
            paramData=self._normalization_transforms, params=self._normalizer_params
        )

    def normalize_in_place(self, mol: Mol) -> Mol:
        self._normalizer.normalizeInPlace(mol)
        return mol


@cache
def get_valence_checker() -> rdMolStandardize.RDKitValidation:
    """Cached RDKit valence checker."""
    return rdMolStandardize.RDKitValidation()


@cache
def get_chembl_normalizer() -> ChEMBLNormalizer:
    """Cached ChEMBL normalizer."""
    return ChEMBLNormalizer()


def _has_correct_valence(mol: Mol) -> bool:
    """Return True if every atom in ``mol`` has a valid valence."""
    mol.UpdatePropertyCache(strict=False)
    return len(get_valence_checker().validate(mol)) == 0


def _calc_formal_charge_from_valence(rdatom: Chem.Atom) -> int:
    """Infer an atom's formal charge from its current valence state."""
    num_valence_electrons = Chem.GetPeriodicTable().GetDefaultValence(
        rdatom.GetSymbol()
    )
    num_electrons_in_bonds = rdatom.GetTotalValence()
    num_radicals = rdatom.GetNumRadicalElectrons()
    return (num_electrons_in_bonds + num_radicals) - num_valence_electrons


def fix_charge_based_on_valence(mol: Mol) -> Mol:
    """Try to repair formal charges so they are consistent with atom valences."""
    previous_mol = copy.deepcopy(mol)

    if not _has_correct_valence(mol):
        for rdatom in mol.GetAtoms():
            rdatom.SetFormalCharge(_calc_formal_charge_from_valence(rdatom))

    return mol if _has_correct_valence(mol) else previous_mol


def change_metal_bonds_to_dative(
    mol: Mol, *, qualifying_bond_types: set[Chem.BondType] = {Chem.BondType.SINGLE}
) -> Mol:
    """Convert qualifying metal bonds into dative (coordination) bonds."""
    if not isinstance(mol, Chem.RWMol):
        mol = Chem.RWMol(mol)

    metal_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in METAL_ELEMENTS
    ]

    for metal_idx in metal_indices:
        for bond in mol.GetAtomWithIdx(metal_idx).GetBonds():
            if bond.GetBondType() in qualifying_bond_types:
                other_idx = bond.GetOtherAtomIdx(metal_idx)
                mol.RemoveBond(metal_idx, other_idx)
                mol.AddBond(other_idx, metal_idx, Chem.BondType.DATIVE)

    return mol


@preserve_annotations
def fix_mol(
    mol: Mol,
    *,
    attempt_fix_by_calculating_charge_from_valence: bool = True,
    attempt_fix_by_normalizing_like_chembl: bool = True,
    attempt_fix_by_normalizing_like_rdkit: bool = True,
    in_place: bool = True,
    raise_on_failure: bool = True,
) -> Mol:
    """Attempt to make a corrupted RDKit molecule sanitizable without altering its graph.

    Infers aromaticity, valences, implicit hydrogens, and formal charges. Does not
    change the heavy atoms or bonds.
    """
    if not in_place:
        mol = copy.deepcopy(mol)

    sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)

    if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
        return mol

    logger.warning(
        f"Molecule failed sanitization: {sanitize_result}. "
        "Attempting to fix by inferring valences and aromaticity."
    )

    mol.UpdatePropertyCache(strict=False)

    if attempt_fix_by_normalizing_like_chembl:
        get_chembl_normalizer().normalize_in_place(mol)
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_by_normalizing_like_rdkit:
        rdMolStandardize.NormalizeInPlace(mol)
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_by_calculating_charge_from_valence:
        fix_charge_based_on_valence(mol)
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
        logger.warning(
            f"Could not fix molecule, final sanitization result: {sanitize_result}"
        )
        if raise_on_failure:
            raise Chem.MolSanitizeException(
                f"Molecule failed sanitization: {sanitize_result}"
            )

    return mol


def add_hydrogens(mol: Mol, add_coords: bool = True) -> Mol:
    """Add explicit hydrogens to an RDKit molecule."""
    return Chem.AddHs(mol, addCoords=add_coords)


# ----------------------------------------------------------------------------
# AtomArray -> RDKit (the main entry point, from atomworks.io.tools.rdkit)
# ----------------------------------------------------------------------------
def atom_array_to_rdkit(
    atom_array: AtomArray,
    *,
    set_coord: bool | None = None,
    hydrogen_policy: Literal["infer", "remove", "keep"] = "keep",
    annotations_to_keep: tuple[str, ...] = BIOTITE_DEFAULT_ANNOTATIONS,
    sanitize: bool = True,
    attempt_fixing_corrupted_molecules: bool = True,
    assume_metal_bonds_are_dative: bool = False,
    infer_bonds: bool = False,
    system_charge: int | None = None,
    timeout_seconds: int = 1,
) -> Mol:
    """Generate an RDKit molecule from a biotite ``AtomArray``.

    Args:
        atom_array: The biotite ``AtomArray`` to convert.
        set_coord: Whether to set atomic coordinates. If None, coordinates are
            only set when they are not NaN.
        hydrogen_policy: ``"infer"``, ``"remove"``, or ``"keep"`` hydrogens.
        annotations_to_keep: Atom annotations to preserve from the ``AtomArray``.
        sanitize: Whether to sanitize the molecule.
        attempt_fixing_corrupted_molecules: Run :func:`fix_mol` when needed.
        assume_metal_bonds_are_dative: Treat all metal bonds as coordination bonds.
        infer_bonds: If True, infer bonds from 3D coordinates (ignoring existing
            bonds); otherwise use ``atom_array.bonds``.
        system_charge: Overall charge for bond-order determination when inferring
            bonds. Defaults to the summed formal charges of the ``AtomArray``.
        timeout_seconds: Timeout for the transition-metal ``xyz2mol_tm`` fallback.

    Returns:
        The RDKit molecule built from the ``AtomArray``.
    """
    atom_array = atom_array.copy()
    mol = Chem.RWMol()

    rdkit_atom_ids = []

    if hydrogen_policy in ("infer", "remove"):
        atom_array = _remove_hydrogens(atom_array)
    elif hydrogen_policy == "keep":
        pass
    else:
        raise ValueError(
            f"Invalid hydrogen policy: {hydrogen_policy}. Must be 'infer', 'remove', or 'keep'."
        )

    has_charge = "charge" in atom_array.get_annotation_categories()
    for atom_id, atom in enumerate(atom_array):
        atomic_number = element_to_atomic_number(atom.element)

        rdatom = Chem.Atom(atomic_number)
        if has_charge:
            rdatom.SetFormalCharge(int(atom.charge))

        rdatom.SetIntProp("rdkit_atom_id", atom_id)
        rdatom.SetProp("atom_name", atom.atom_name)
        rdkit_atom_ids.append(atom_id)
        mol.AddAtom(rdatom)

    set_coord = set_coord or not np.any(np.isnan(atom_array.coord))
    if set_coord:
        conf_id = mol.AddConformer(Chem.Conformer(len(atom_array)), assignId=True)
        for atom_id, atom_coord in enumerate(atom_array.coord):
            mol.GetConformer(conf_id).SetAtomPosition(atom_id, atom_coord.tolist())

    _should_be_aromatic = set()
    if infer_bonds:
        assert mol.GetNumAtoms() > 0, "Cannot infer bonds for empty molecule"
        assert (
            has_charge or system_charge is not None
        ), "System charge must be provided when inferring bonds if atom_array has no 'charge' annotation."

        system_charge = (
            system_charge
            if system_charge is not None
            else int(np.nansum(atom_array.charge))
        )

        try:
            # (Fast) Try standard rdDetermineBonds first.
            with _suppress_stderr_fd():
                rdDetermineBonds.DetermineBonds(
                    mol, useHueckel=True, charge=system_charge, maxIterations=10_000
                )
        except Exception as err_rdkit:
            # (Slow) Transition-metal complexes (TMC): fall back to xyz2mol_tm.
            try:

                def _get_tmc_mol_with_timeout(
                    mol: Mol, overall_charge: int, with_stereo: bool
                ) -> Mol:
                    with _signal_timeout(timeout_seconds), _suppress_stderr_fd():
                        return get_tmc_mol(
                            mol,
                            overall_charge=overall_charge,
                            with_stereo=with_stereo,
                        )

                mol = _get_tmc_mol_with_timeout(
                    mol, overall_charge=system_charge, with_stereo=True
                )

                # xyz2mol_tm preserves rdkit_atom_id but reorders atoms; restore order.
                rdkit_id_to_current_idx = {}
                for current_idx, atom in enumerate(mol.GetAtoms()):
                    rdkit_id = atom.GetIntProp("rdkit_atom_id")
                    rdkit_id_to_current_idx[rdkit_id] = current_idx

                new_order = [
                    rdkit_id_to_current_idx[i]
                    for i in range(len(rdkit_id_to_current_idx))
                ]
                mol = Chem.RenumberAtoms(mol, new_order)

            except Exception as err_xyz2mol:
                raise RuntimeError(
                    f"Bond inference failed with both methods:\n"
                    f"  rdDetermineBonds: {err_rdkit}\n"
                    f"  xyz2mol_tm: {err_xyz2mol}"
                ) from err_xyz2mol

    elif exists(atom_array.bonds):
        for bond in atom_array.bonds.as_array():
            atom1, atom2, bond_type = list(map(int, bond))

            if bond_type == struc.bonds.BondType.ANY:
                logger.warning("Encountered BondType.ANY. Interpreting as single bond.")

            bond_order, bond_is_aromatic = BIOTITE_BOND_TYPE_TO_RDKIT[bond_type]
            mol.AddBond(atom1, atom2, order=bond_order)

            if bond_is_aromatic and not attempt_fixing_corrupted_molecules:
                mol.GetAtomWithIdx(atom1).SetIsAromatic(True)
                mol.GetAtomWithIdx(atom2).SetIsAromatic(True)

            _should_be_aromatic.union({atom1, atom2})

    if mol.GetNumConformers() > 0:
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except (ValueError, RuntimeError):
            logger.warning("Failed to assign stereochemistry to molecule.")
            pass

    if assume_metal_bonds_are_dative:
        mol = change_metal_bonds_to_dative(mol)

    if attempt_fixing_corrupted_molecules:
        mol = fix_mol(
            mol,
            attempt_fix_by_normalizing_like_chembl=True,
            attempt_fix_by_normalizing_like_rdkit=True,
            attempt_fix_by_calculating_charge_from_valence=True,
            in_place=True,
            raise_on_failure=False,
        )

    if sanitize or attempt_fixing_corrupted_molecules:
        try:
            Chem.SanitizeMol(mol)
        except (
            Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException,
            Chem.MolSanitizeException,
        ) as e:
            logger.warning(
                f"Sanitization failed: {type(e).__name__}: {e}. "
                "Molecule will be used without sanitization. "
                "This may affect aromaticity perception and formal charges."
            )

        for atom_idx in _should_be_aromatic:
            assert mol.GetAtomWithIdx(
                atom_idx
            ).GetIsAromatic(), (
                f"Atom {atom_idx} is not aromatic but was labelled as aromatic."
            )

    mol = mol.GetMol() if isinstance(mol, Chem.RWMol) else mol

    mol._annotations = {"rdkit_atom_id": np.array(rdkit_atom_ids)}
    for annotation in annotations_to_keep:
        if annotation in atom_array.get_annotation_categories():
            mol._annotations[annotation] = atom_array._annot[annotation]

    if hydrogen_policy == "infer":
        mol = add_hydrogens(mol, add_coords=set_coord)

    return mol


def ccd_code_to_rdkit(
    ccd_code: str,
    *,
    hydrogen_policy: Literal["infer", "remove", "keep"] = "keep",
    **atom_array_to_rdkit_kwargs,
) -> Mol:
    """Convert a CCD residue code (e.g. ``'ALA'``, ``'9RH'``) to an RDKit molecule.

    Uses biotite's bundled Chemical Component Dictionary as the template source.

    Args:
        ccd_code: The CCD three-letter (or longer) component code.
        hydrogen_policy: Whether to keep/remove/infer hydrogens.
        **atom_array_to_rdkit_kwargs: Forwarded to :func:`atom_array_to_rdkit`.

    Returns:
        The RDKit molecule for the requested component.

    Raises:
        KeyError: If ``ccd_code`` is not present in biotite's CCD.
    """
    atom_array = _biotite_residue(ccd_code)

    mol = atom_array_to_rdkit(
        atom_array,
        set_coord=True,  # coordinates needed for stereochemistry assignment
        hydrogen_policy=hydrogen_policy,
        **atom_array_to_rdkit_kwargs,
    )

    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except (ValueError, RuntimeError):
        logger.warning(
            f"Failed to assign stereochemistry to {ccd_code}. Returning unstereochem molecule."
        )
        pass

    return mol
