"""Contracts for tmol's Dimorphite rule inventory, run on AtomWorks' engine."""

from pathlib import Path

import pytest
from rdkit import Chem

from atomworks.protonation.external.dimorphite_dl.dimorphite_dl import (
    ProtSubstructFuncs as AtomWorksProtSubstructFuncs,
)
from atomworks.protonation.external.dimorphite_dl.dimorphite_dl import (
    protonate_mol_variants as atomworks_protonate_mol_variants,
)
from tmol.ligand._dimorphite_dl import ProtSubstructFuncs, protonate_mol_variants


class OverlappingSiteRules(ProtSubstructFuncs):
    """Rules whose ionization sites do not overlap but whose contexts do."""

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        return [
            "High_acid [O:1]-[C:2] 0 1.0 0.0",
            "Low_amine [C:1]-[C:2]-[N:3] 2 14.0 0.0",
        ]


class AtomWorksTmolRules(AtomWorksProtSubstructFuncs):
    """Load tmol's scientific inventory through the public AtomWorks engine."""

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        path = Path(__file__).parents[2] / "ligand" / "site_substructures.smarts"
        return [
            line
            for line in path.read_text().splitlines(keepends=True)
            if line.strip() and not line.startswith("#")
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
def test_tmol_specific_rules_cover_frank_protonation_cases(smiles, expected):
    """Pin cases added by Frank that AtomWorks' default SMARTS do not encode."""
    products = protonate_mol_variants(
        Chem.MolFromSmiles(smiles),
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0.1,
    )
    assert canonical_states(products) == [expected]


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)O",
        "CN(C)C=C",
        "CN(C)N=O",
        "COP(=O)(S)OC",
        "N[C@@H](COP(=O)(O)O)C(=O)O",
    ],
)
def test_tmol_rules_are_supplied_without_the_caller_asking(smiles):
    """Calling tmol's wrapper must use tmol's inventory, not the engine's default.

    The engine is shared, so the only thing that can go wrong is the rule set
    silently reverting to AtomWorks' -- which no call site would notice.
    """
    molecule = Chem.MolFromSmiles(smiles)
    kwargs = {
        "min_ph": 7.4,
        "max_ph": 7.4,
        "pka_precision": 0.1,
    }
    expected = atomworks_protonate_mol_variants(
        molecule,
        **kwargs,
        rule_provider=AtomWorksTmolRules,
    )
    observed = protonate_mol_variants(molecule, **kwargs)
    assert canonical_states(observed) == canonical_states(expected)


def test_the_shared_engine_carries_the_rule_tmol_needs():
    """A case Frank added: the engine's own defaults once left this neutral.

    AtomWorks now ships the full rule inventory, so the two agree. Asserting
    that agreement is what keeps the shared engine honest -- a rule dropped
    upstream would show up here.
    """
    molecule = Chem.MolFromSmiles("COP(=O)(S)OC")
    kwargs = {"min_ph": 7.4, "max_ph": 7.4, "pka_precision": 0.1}

    with_tmol_rules = canonical_states(protonate_mol_variants(molecule, **kwargs))
    with_engine_rules = canonical_states(
        atomworks_protonate_mol_variants(
            molecule, **kwargs, rule_provider=AtomWorksProtSubstructFuncs
        )
    )

    assert with_tmol_rules == ["COP(=O)([S-])OC"]
    assert with_engine_rules == with_tmol_rules
