"""Private copy of the canonical-AA tuple used by the backbone-torsion
scoring subpackage.

Ideally the rama/omega modules would ``from tmol_jax.chemical.constants
import canonical_aa_3letter``. At the time this port landed, however,
``tmol_jax.chemical`` had a circular-import chain triggered whenever any
``tmol_jax.chemical.*`` module is imported (``chemical.__init__`` pulls
in ``restypes``, which pulls ``database.chemical`` -> ``PatchedChemical
Database`` -> back into ``restypes`` before its classes are fully
defined). That breaks ``import tmol_jax.score.backbone_torsion`` any
time it goes through the chemical package.

The tuple is tiny (20 entries) and static — keeping a private copy
removes the chemical-package dependency entirely. If upstream resolves
the circular import, callers can drop this module and re-import from
``tmol_jax.chemical.constants`` without touching the public API.
"""

from __future__ import annotations

# 20 canonical AAs, in the same insertion order as
# ``tmol_jax.chemical.constants.one2three``.
canonical_aa_3letter: tuple = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)


__all__ = ["canonical_aa_3letter"]
