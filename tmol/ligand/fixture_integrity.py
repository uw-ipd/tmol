"""Fixture-integrity gate for mol2/``.params`` reference pairs.

Before a mol2-strict comparison runs, the prepared ligand's mol2 and its
reference ``.params`` must describe the *same molecule*. A silent mismatch
(for example the legacy ``ref1.mol2`` / ``ref1.params`` fixtures, which are
different molecules) would otherwise produce a misleading pass/fail. This
module reads a mol2 and checks it against a parsed reference, raising a clear
:class:`FixtureMismatch` that lists every failed check.

Note on atom names: ``mol2genparams --rename_atoms`` keeps heavy-atom names but
re-derives hydrogen names by attachment (``H1`` -> ``HC*/HN*/HO*/HS*``), so a
valid pair has identical *heavy*-atom names while hydrogen names legitimately
differ. The name-set check therefore compares heavy-atom names; element
composition and counts corroborate the pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tmol.ligand.params_reference import ReferenceParams, parse_reference_params


class FixtureMismatch(ValueError):
    """Raised when a mol2 and its reference ``.params`` are not the same molecule."""


@dataclass(frozen=True)
class Mol2Summary:
    """Structural summary of a TRIPOS mol2 file (no chemistry perception)."""

    title: str
    atom_names: tuple[str, ...]
    heavy_names: frozenset[str]
    element_counts: dict[str, int]
    n_bonds: int
    charge_type: str

    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the mol2."""
        return len(self.atom_names)

    @property
    def has_hydrogen(self) -> bool:
        """Whether the mol2 contains any hydrogen atoms."""
        return self.element_counts.get("H", 0) > 0


def _element_of(sybyl_type: str) -> str:
    """Return the element symbol from a SYBYL atom type (``C.3`` -> ``C``)."""
    return sybyl_type.split(".", 1)[0]


def read_mol2_summary(path: str | Path) -> Mol2Summary:
    """Read structural counts and names from a single-molecule mol2 file.

    Args:
        path: Path to a TRIPOS ``.mol2`` file.

    Returns:
        A :class:`Mol2Summary`.
    """
    title = ""
    charge_type = ""
    atom_names: list[str] = []
    heavy_names: set[str] = set()
    element_counts: dict[str, int] = {}
    n_bonds = 0
    section = ""
    molecule_line = 0

    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                section = stripped
                molecule_line = 0
                continue
            if section == "@<TRIPOS>MOLECULE":
                molecule_line += 1
                # Block order: 1=name, 2=counts, 3=mol_type, 4=charge_type.
                if molecule_line == 1 and not title:
                    title = stripped
                elif molecule_line == 4 and not charge_type:
                    charge_type = stripped
                continue
            if section == "@<TRIPOS>ATOM":
                parts = stripped.split()
                if len(parts) < 6:
                    continue
                name = parts[1]
                element = _element_of(parts[5])
                atom_names.append(name)
                element_counts[element] = element_counts.get(element, 0) + 1
                if element != "H":
                    heavy_names.add(name)
            elif section == "@<TRIPOS>BOND":
                if stripped:
                    n_bonds += 1

    return Mol2Summary(
        title=title,
        atom_names=tuple(atom_names),
        heavy_names=frozenset(heavy_names),
        element_counts=element_counts,
        n_bonds=n_bonds,
        charge_type=charge_type,
    )


# Charge-model aliases -> the TRIPOS charge-type tag they require.
_CHARGE_MODEL_TAGS: dict[str, frozenset[str]] = {
    "mmff94": frozenset({"MMFF94_CHARGES"}),
    "user": frozenset({"USER_CHARGES"}),
    "input": frozenset({"USER_CHARGES"}),
    "gasteiger": frozenset({"GASTEIGER"}),
}
_NO_CHARGE_TAGS = frozenset({"", "NO_CHARGES"})


def _charge_model_failure(
    expected_charge_model: str, mol2_charge_type: str
) -> str | None:
    """Return a failure message if the mol2 charge model does not satisfy expected.

    ``'auto'`` (or ``'any'``) accepts any usable charge type (charges present,
    not ``NO_CHARGES``). A specific alias requires the matching TRIPOS tag. An
    unrecognized expected model can never be satisfied and fails.
    """
    model = expected_charge_model.strip().lower()
    if model in {"auto", "any"}:
        if mol2_charge_type.upper() in _NO_CHARGE_TAGS:
            return (
                f"charge model: expected usable charges (mode '{model}') "
                f"but mol2 declares '{mol2_charge_type or 'NONE'}'"
            )
        return None
    accepted = _CHARGE_MODEL_TAGS.get(model)
    if accepted is None or mol2_charge_type.upper() not in accepted:
        wanted = sorted(accepted) if accepted else f"<unknown model '{model}'>"
        return (
            f"charge model: expected {wanted} for mode "
            f"'{expected_charge_model}' but mol2 declares "
            f"'{mol2_charge_type or 'NONE'}'"
        )
    return None


def require_paired_fixture(
    mol2_path: str | Path,
    reference: ReferenceParams | str | Path,
    *,
    expected_charge_model: str = "auto",
) -> Mol2Summary:
    """Verify a mol2 and its reference ``.params`` describe the same molecule.

    Args:
        mol2_path: Path to the ligand mol2.
        reference: A parsed :class:`ReferenceParams` or a path to a ``.params``.
        expected_charge_model: The charge model the mol2 must declare.
            ``'auto'`` (default) requires usable charges (a charge-type tag that
            is not ``NO_CHARGES``); an explicit model such as ``'mmff94'``
            requires the matching TRIPOS tag (``MMFF94_CHARGES``). An
            unrecognized model can never be satisfied and fails. The gate also
            confirms the reference carries a charge for every atom.

    Returns:
        The :class:`Mol2Summary` on success.

    Raises:
        FixtureMismatch: If any pairing check fails; the message lists them.
    """
    if not isinstance(reference, ReferenceParams):
        reference = parse_reference_params(reference)

    mol2 = read_mol2_summary(mol2_path)
    ref_heavy = reference.heavy_atom_names()
    failures: list[str] = []

    if len(mol2.heavy_names) != len(ref_heavy):
        failures.append(
            f"heavy-atom count: mol2={len(mol2.heavy_names)} "
            f"params={len(ref_heavy)}"
        )
    if mol2.heavy_names != ref_heavy:
        only_mol2 = sorted(mol2.heavy_names - ref_heavy)
        only_params = sorted(ref_heavy - mol2.heavy_names)
        failures.append(
            f"heavy-atom name set: only_in_mol2={only_mol2} "
            f"only_in_params={only_params}"
        )
    if mol2.title != reference.name:
        failures.append(f"residue name: mol2='{mol2.title}' params='{reference.name}'")
    if mol2.n_atoms != len(reference.atoms):
        failures.append(
            f"total atom count: mol2={mol2.n_atoms} params={len(reference.atoms)}"
        )
    if mol2.n_bonds != len(reference.bond_types):
        failures.append(
            f"bond count: mol2={mol2.n_bonds} params={len(reference.bond_types)}"
        )
    if mol2.has_hydrogen != reference.has_hydrogen:
        failures.append(
            f"hydrogen presence: mol2={mol2.has_hydrogen} "
            f"params={reference.has_hydrogen}"
        )
    charge_failure = _charge_model_failure(expected_charge_model, mol2.charge_type)
    if charge_failure is not None:
        failures.append(charge_failure)
    if len(reference.charges) != len(reference.atoms):
        failures.append(
            f"charge coverage: reference has {len(reference.charges)} charges "
            f"for {len(reference.atoms)} atoms"
        )

    if failures:
        raise FixtureMismatch(
            "Fixture mismatch between "
            f"{Path(mol2_path).name} and params '{reference.name}': "
            + "; ".join(failures)
        )
    return mol2
