"""Contracts for the Dimorphite engine and rule inventory TMol takes from AtomWorks."""

import pytest
from atomworks.experimental.protonation.external.dimorphite_dl.dimorphite_dl import (
    ProtSubstructFuncs,
    protonate_mol_variants,
)
from rdkit import Chem


class OverlappingSiteRules(ProtSubstructFuncs):
    """Rules whose ionization sites do not overlap but whose contexts do."""

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        return [
            "High_acid [O:1]-[C:2] 0 1.0 0.0",
            "Low_amine [C:1]-[C:2]-[N:3] 2 14.0 0.0",
        ]


def canonical_states(products: list[Chem.Mol]) -> list[str]:
    """Return canonical isomeric SMILES in enumeration order."""
    return [
        Chem.MolToSmiles(product, isomericSmiles=True, canonical=True)
        for product in products
    ]


def test_rule_priority_blocks_only_previously_tagged_ionization_sites():
    """Allow a lower-priority site whose SMARTS context overlaps a tagged group."""
    molecule = Chem.MolFromSmiles("OCCN")
    rules = OverlappingSiteRules.load_protonation_substructs_calc_state_for_ph(
        7.4, 7.4, 0.0
    )
    sites, _ = ProtSubstructFuncs.get_prot_sites_and_target_states_from_mol(
        molecule, rules
    )
    products = protonate_mol_variants(
        molecule,
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0.0,
        rule_provider=OverlappingSiteRules,
    )

    assert [(site[1], site[2]) for site in sites] == [
        ("DEPROTONATED", "High_acid"),
        ("PROTONATED", "Low_amine"),
    ]
    assert canonical_states(products) == ["[NH3+]CC[O-]"]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CN(C)C=C", "C=CN(C)C"),
        ("CN(C)N=O", "CN(C)N=O"),
        ("COP(=O)(S)OC", "COP(=O)([S-])OC"),
        ("[O-]S([O-])(=O)=O", "O=S(=O)([O-])[O-]"),
        ("OP(O)(O)=O", "O=P([O-])([O-])O"),
    ],
)
def test_rules_cover_frank_protonation_cases(smiles, expected):
    """Pin the cases Frank added to the rule inventory TMol protonates with."""
    products = protonate_mol_variants(
        Chem.MolFromSmiles(smiles),
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0.1,
    )
    assert canonical_states(products) == [expected]
