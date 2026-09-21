"""Declared library references must resolve before preparing native tables."""

import attr
import pandas
import pytest

from tmol.score.dunbrack import DunbrackParamResolver
from tmol.tests.score.dunbrack.test_empty_libraries import select_libraries


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing", "MISSING_TABLE"),
        ("missing_empty", "MISSING_TABLE"),
        ("duplicate_residue", "lookup names must be unique"),
        ("conflicting_residue", "lookup names must be unique"),
        ("duplicate_table", "table names must be unique"),
        ("cross_family_table", "table names must be unique"),
    ],
)
def test_bad_lookup_is_rejected_before_tensor_preparation(
    default_database, torch_device, monkeypatch, defect, message
):
    full = default_database.scoring.dun
    database = select_libraries(full, "rotameric")
    row = database.dun_lookup[0]
    if defect.startswith("missing"):
        database = attr.evolve(
            database,
            dun_lookup=(attr.evolve(row, dun_table_name="MISSING_TABLE"),),
            rotameric_libraries=(
                () if defect == "missing_empty" else database.rotameric_libraries
            ),
        )
    elif defect.endswith("residue"):
        other = row
        if defect == "conflicting_residue":
            semi = select_libraries(full, "semirotameric")
            database = attr.evolve(
                database, semi_rotameric_libraries=semi.semi_rotameric_libraries
            )
            other = attr.evolve(row, dun_table_name=semi.dun_lookup[0].dun_table_name)
        database = attr.evolve(database, dun_lookup=(*database.dun_lookup, other))
    elif defect == "duplicate_table":
        database = attr.evolve(
            database, rotameric_libraries=database.rotameric_libraries * 2
        )
    else:
        other = attr.evolve(
            full.semi_rotameric_libraries[0],
            table_name=database.rotameric_libraries[0].table_name,
        )
        database = attr.evolve(database, semi_rotameric_libraries=(other,))

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("derived tensor preparation reached before lookup validation")

    monkeypatch.setattr(
        DunbrackParamResolver, "_create_nchi_for_table_set", unexpected_allocation
    )
    with pytest.raises(ValueError, match=message):
        DunbrackParamResolver.from_database(database, torch_device)


def test_reordered_aliases_keep_family_local_indices(default_database, torch_device):
    full = default_database.scoring.dun
    leu = select_libraries(full, "rotameric")
    phe = select_libraries(full, "semirotameric")
    rows = (
        attr.evolve(phe.dun_lookup[0], residue_name="PHE_ALIAS"),
        attr.evolve(leu.dun_lookup[0], residue_name="LEU_ALIAS"),
        attr.evolve(phe.dun_lookup[0], residue_name="PHE_ALIAS_2"),
    )
    database = attr.evolve(
        leu, dun_lookup=rows, semi_rotameric_libraries=phe.semi_rotameric_libraries
    )
    resolver = DunbrackParamResolver.from_database(database, torch_device)
    for name, values in (
        ("all_table_indices", [1, 0, 1]),
        ("rotameric_table_indices", [-1, 0, -1]),
        ("semirotameric_table_indices", [0, -1, 0]),
    ):
        expected = pandas.DataFrame(
            {"dun_table_name": values},
            index=pandas.Index([r.residue_name for r in rows], name="residue_name"),
        )
        pandas.testing.assert_frame_equal(getattr(resolver, name), expected)
