import numpy
import torch

from tmol.database.scoring._genbonded import GenBondedDatabase
from tmol.score.genbonded import GenBondedEnergyTerm


def test_genbonded_parameter_lookups_are_reused(
    fresh_default_packed_block_types, default_database, monkeypatch
):
    calls = {"torsion": 0, "improper": 0}
    original_torsion = GenBondedDatabase.find_torsion_params
    original_improper = GenBondedDatabase.find_improper_params

    def counted_torsion(database, *args):
        calls["torsion"] += 1
        return original_torsion(database, *args)

    def counted_improper(database, *args):
        calls["improper"] += 1
        return original_improper(database, *args)

    monkeypatch.setattr(GenBondedDatabase, "find_torsion_params", counted_torsion)
    monkeypatch.setattr(GenBondedDatabase, "find_improper_params", counted_improper)

    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    blocks = fresh_default_packed_block_types.active_block_types
    torsion_block, torsions = next(
        (block, subgraphs)
        for block in blocks
        if (subgraphs := term.find_torsion_subgraphs(block.bond_indices))
    )
    improper_block, impropers = next(
        (block, subgraphs)
        for block in blocks
        if (subgraphs := term.find_improper_subgraphs(block.bond_indices))
    )

    first_torsions, first_torsion_params = term.resolve_torsion_params(
        torsion_block, torsions
    )
    first_impropers, first_improper_params = term.resolve_improper_params(
        improper_block, impropers
    )
    first_call_counts = calls.copy()

    second_torsions, second_torsion_params = term.resolve_torsion_params(
        torsion_block, torsions
    )
    second_impropers, second_improper_params = term.resolve_improper_params(
        improper_block, impropers
    )

    assert first_call_counts["torsion"] > 0
    assert first_call_counts["improper"] > 0
    assert calls == first_call_counts
    assert second_torsions == first_torsions
    assert second_impropers == first_impropers
    numpy.testing.assert_array_equal(second_torsion_params, first_torsion_params)
    numpy.testing.assert_array_equal(second_improper_params, first_improper_params)
