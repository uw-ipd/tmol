"""Database-backed ligand chemistry lookup tables.

These helpers derive atom-class sets and hbond metadata from the default
chemical database so atom-type updates can be handled centrally in YAML.
"""

from typing import Any

from tmol.database import ParameterDatabase

_LEGACY_POLAR_CLASSES = frozenset(
    {
        "Oat",
        "Oad",
        "Ohx",
        "Oal",
        "OG23",
        "OG2",
        "OG3",
        "Nam",
        "Ngu1",
        "Ngu2",
        "NG22",
        "SG5",
        "PG3",
        "PG5",
    }
)


def _default_atom_types() -> list[Any]:
    """Return atom-type entries from the default parameter database.

    Returns:
        List of atom-type records loaded from ``ParameterDatabase``.
    """
    return ParameterDatabase.get_default().chemical.atom_types


def get_hbond_properties() -> dict[str, dict[str, Any]]:
    """Build hydrogen-bond related properties by atom-type name.

    Returns:
        Mapping from atom-type name to a compact property dictionary.
    """
    props: dict[str, dict] = {}
    for at in _default_atom_types():
        entry = {}
        if at.is_acceptor:
            entry["is_acceptor"] = True
        if at.is_donor:
            entry["is_donor"] = True
        if at.is_polarh:
            entry["is_polarh"] = True
        if at.is_hydroxyl:
            entry["is_hydroxyl"] = True
        if at.acceptor_hybridization is not None:
            entry["acceptor_hybridization"] = str(at.acceptor_hybridization).upper()
        if entry:
            props[at.name] = entry
    return props


def get_polar_classes() -> frozenset[str]:
    """Return configured legacy polar classes available in the database.

    Returns:
        Set of polar atom-class names that exist in the current database.
    """
    available = {at.name for at in _default_atom_types()}
    return frozenset(name for name in _LEGACY_POLAR_CLASSES if name in available)


def get_sp2_atom_types() -> frozenset[str]:
    """Collect atom-type names treated as sp2-like for typing helpers.

    Returns:
        Set of atom-type names considered sp2-like.
    """
    sp2_types = set()
    for at in _default_atom_types():
        if at.name.startswith(("CD", "CR")):
            sp2_types.add(at.name)
        if at.element in ("O", "N") and str(at.acceptor_hybridization) in (
            "sp2",
            "ring",
        ):
            sp2_types.add(at.name)
    return frozenset(sp2_types)
