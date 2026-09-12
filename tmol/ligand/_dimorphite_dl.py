"""Shared AtomWorks protonation engine with tmol's fixed scientific rule inventory."""

from pathlib import Path

from atomworks.external.dimorphite_dl import dimorphite_dl as _shared
from atomworks.external.dimorphite_dl.dimorphite_dl import (  # noqa: F401
    ArgParseFuncs,
    LoadSMIFile,
    MyParser,
    ProtectUnprotectFuncs,
    TestFuncs,
    UtilFuncs,
    print_header,
)


class ProtSubstructFuncs(_shared.ProtSubstructFuncs):
    """Select tmol's rule ordering and pKa values without changing AtomWorks defaults."""

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        with Path(__file__).with_name("site_substructures.smarts").open() as source:
            return [
                line for line in source if line.strip() and not line.startswith("#")
            ]


class Protonate(_shared.Protonate):
    """Enumerate SMILES using tmol's explicit rule provider."""

    def __init__(self, args):
        super().__init__({**args, "rule_provider": ProtSubstructFuncs})


def main(params=None):
    return _shared.main({**(params or {}), "rule_provider": ProtSubstructFuncs})


def protonate_mol_variants(
    mol,
    min_ph=6.4,
    max_ph=8.4,
    pka_precision=1.0,
    max_variants=128,
    silent=True,
):
    """Enumerate mapped molecular states using tmol's explicit rule provider."""
    return _shared.protonate_mol_variants(
        mol,
        min_ph=min_ph,
        max_ph=max_ph,
        pka_precision=pka_precision,
        max_variants=max_variants,
        silent=silent,
        rule_provider=ProtSubstructFuncs,
    )


if __name__ == "__main__":
    main()
