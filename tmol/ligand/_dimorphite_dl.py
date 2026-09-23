"""Dimorphite-DL with TMol's rule inventory.

The engine is AtomWorks' vendored copy; only the rules are TMol's.
``site_substructures.smarts`` beside this module extends the upstream inventory
with cases it does not encode -- enamines, nitrosamines, anilines, phosphate and
phosphorothioate esters, phosphoramides, and tertiary amides and sulfonamides.

Every TMol caller goes through this module, so the rule set cannot be forgotten
at a call site.
"""

import os

from atomworks.protonation.external.dimorphite_dl.dimorphite_dl import (  # noqa: F401
    ArgParseFuncs,
    LoadSMIFile,
    MyParser,
    ProtectUnprotectFuncs,
    Protonate,
    ProtSubstructFuncs as _ProtSubstructFuncs,
    TestFuncs,
    UtilFuncs,
    main,
    print_header,
)
from atomworks.protonation.external.dimorphite_dl.dimorphite_dl import (
    protonate_mol_variants as _protonate_mol_variants,
)

__all__ = [
    "ArgParseFuncs",
    "LoadSMIFile",
    "MyParser",
    "ProtSubstructFuncs",
    "ProtectUnprotectFuncs",
    "Protonate",
    "TestFuncs",
    "UtilFuncs",
    "main",
    "print_header",
    "protonate_mol_variants",
]


class ProtSubstructFuncs(_ProtSubstructFuncs):
    """Reads the rule inventory beside this module rather than the engine's own."""

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "site_substructures.smarts"
        )
        with open(path) as handle:
            return [
                line for line in handle if line.strip() and not line.startswith("#")
            ]


def protonate_mol_variants(mol, **kwargs):
    """Enumerate protonation variants using TMol's rules.

    Thin wrapper over the shared engine that supplies :class:`ProtSubstructFuncs`
    unless a caller names its own provider.
    """
    kwargs.setdefault("rule_provider", ProtSubstructFuncs)
    return _protonate_mol_variants(mol, **kwargs)
