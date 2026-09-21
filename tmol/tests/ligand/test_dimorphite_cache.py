"""Rule queries are bounded and cannot be poisoned through public results."""

import pytest
from rdkit import Chem

from tmol.ligand._dimorphite_dl import ProtSubstructFuncs, protonate_mol_variants


@pytest.fixture(autouse=True)
def clear_rule_cache():
    ProtSubstructFuncs._compiled_substructures.cache_clear()
    yield
    ProtSubstructFuncs._compiled_substructures.cache_clear()


def test_rule_loader_returns_owned_queries_and_nested_states():
    load = ProtSubstructFuncs.load_protonation_substructs_calc_state_for_ph
    first = load(2.0, 2.0, 0.1)
    name = first[0]["name"]
    states = [state.copy() for state in first[0]["prot_states_for_pH"]]
    first[0]["name"] = "poisoned"
    first[0]["mol"].SetProp("poisoned", "yes")
    first[0]["mol"].GetAtomWithIdx(0).SetAtomMapNum(999)
    first[0]["prot_states_for_pH"][0][1] = "poisoned"
    first.pop()
    second = load(2.0, 2.0, 0.1)
    assert second[0]["name"] == name
    assert second[0]["prot_states_for_pH"] == states
    assert not second[0]["mol"].HasProp("poisoned")
    assert second[0]["mol"].GetAtomWithIdx(0).GetAtomMapNum() != 999
    assert len(second) == len(first) + 1
    products = protonate_mol_variants(
        Chem.MolFromSmiles("CC(=O)O"), min_ph=2.0, max_ph=2.0, pka_precision=0.1
    )
    assert [Chem.GetFormalCharge(mol) for mol in products] == [0]


def test_ph_sweep_compiles_one_rule_set(monkeypatch):
    calls = []
    original = ProtSubstructFuncs.load_substructre_smarts_file

    def read_rules():
        calls.append(True)
        return original()

    monkeypatch.setattr(ProtSubstructFuncs, "load_substructre_smarts_file", read_rules)
    retained_queries = None
    for value in range(500):
        subs = ProtSubstructFuncs._substructures_for_ph(value / 30, value / 30, 0.1)
        queries = tuple(id(sub["mol"]) for sub in subs)
        if retained_queries is None:
            retained_queries = queries
        assert queries == retained_queries
    assert len(calls) == 1
    assert ProtSubstructFuncs._compiled_substructures.cache_info().currsize == 1


@pytest.mark.parametrize("ph,charge", [(2.0, 0), (12.0, -1), (2.0, 0), (7.4, -1)])
def test_interleaved_ph_requests_get_current_states(ph, charge):
    # Populate different states first so a cache keyed only by rule identity
    # cannot accidentally apply its first pH to the molecule.
    mol = Chem.MolFromSmiles("CC(=O)O")
    protonate_mol_variants(mol, min_ph=0, max_ph=14)
    products = protonate_mol_variants(mol, min_ph=ph, max_ph=ph, pka_precision=0.1)
    assert [Chem.GetFormalCharge(product) for product in products] == [charge]
