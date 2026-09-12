"""Used chemical types must have real, finite scoring parameters."""

import attr
import numpy
import pytest

from tmol.pose import PackedBlockTypes
from tmol.score import AtomTypeDependentTerm
from tmol.score.hbond import HBondEnergyTerm
from tmol.score.ljlk import LJLKEnergyTerm, LJLKParamResolver
from tmol.score.lk_ball import LKBallEnergyTerm


@pytest.fixture
def ala(fresh_default_restype_set):
    return next(
        rt for rt in fresh_default_restype_set.residue_types if rt.name == "ALA"
    )


def change_ljlk(database, transform):
    return attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            ljlk=attr.evolve(
                database.scoring.ljlk,
                atom_type_parameters=tuple(
                    transform(database.scoring.ljlk.atom_type_parameters)
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "term_class",
    [AtomTypeDependentTerm, HBondEnergyTerm, LJLKEnergyTerm, LKBallEnergyTerm],
)
def test_used_unregistered_chemical_type_rejected(
    default_database, ala, torch_device, term_class
):
    database = attr.evolve(
        default_database,
        chemical=attr.evolve(
            default_database.chemical,
            atom_types=tuple(
                a for a in default_database.chemical.atom_types if a.name != "CH3"
            ),
        ),
    )
    term = term_class(database, torch_device)
    with pytest.raises(ValueError, match="ALA.*CB.*CH3"):
        term.setup_block_type(ala)


@pytest.mark.parametrize("term_class", [LJLKEnergyTerm, LKBallEnergyTerm])
@pytest.mark.parametrize(
    "defect,field",
    [("missing", "lj_radius"), ("inf", "lj_radius"), ("float32_overflow", "lj_radius")]
    + [
        ("nan", f)
        for f in ("lj_radius", "lj_wdepth", "lk_dgfree", "lk_lambda", "lk_volume")
    ],
)
def test_used_incomplete_ljlk_type_rejected(
    default_database, ala, torch_device, term_class, defect, field
):
    def transform(rows):
        for row in rows:
            if row.name != "CH3":
                yield row
            elif defect != "missing":
                value = {
                    "nan": float("nan"),
                    "inf": float("inf"),
                    "float32_overflow": 1e100,
                }[defect]
                yield attr.evolve(row, **{field: value})

    term = term_class(change_ljlk(default_database, transform), torch_device)
    with pytest.raises(ValueError, match=f"ALA.*CB.*CH3.*{field}"):
        term.setup_block_type(ala)


@pytest.mark.parametrize("term_class", [LJLKEnergyTerm, LKBallEnergyTerm])
def test_empty_ljlk_table_reports_used_atoms(
    default_database, ala, torch_device, term_class
):
    term = term_class(change_ljlk(default_database, lambda rows: ()), torch_device)
    with pytest.raises(ValueError, match="ALA.*CB.*CH3.*lj_radius"):
        term.setup_block_type(ala)


@pytest.mark.parametrize("term_class", [LJLKEnergyTerm, LKBallEnergyTerm])
def test_cached_packed_annotation_does_not_hide_missing_parameters(
    default_database, fresh_default_restype_set, ala, torch_device, term_class
):
    packed = PackedBlockTypes.from_restype_list(
        default_database.chemical, fresh_default_restype_set, [ala], torch_device
    )
    term_class(default_database, torch_device).setup_packed_block_types(packed)
    incomplete = change_ljlk(
        default_database, lambda rows: (r for r in rows if r.name != "CH3")
    )
    with pytest.raises(ValueError, match="ALA.*CB.*CH3"):
        term_class(incomplete, torch_device).setup_packed_block_types(packed)
    # A failed validation must not poison the original configuration.
    term_class(default_database, torch_device).setup_packed_block_types(packed)


@pytest.mark.parametrize("term_class", [LJLKEnergyTerm, LKBallEnergyTerm])
def test_unused_missing_parameters_and_zero_virtual_parameters_allowed(
    default_database, fresh_default_restype_set, ala, torch_device, term_class
):
    incomplete = change_ljlk(
        default_database, lambda rows: (r for r in rows if r.name != "CH2")
    )
    term = term_class(incomplete, torch_device)
    term.setup_block_type(ala)
    vrt = next(rt for rt in fresh_default_restype_set.residue_types if rt.name == "VRT")
    term.setup_block_type(vrt)


def test_low_level_invalid_parameter_row_preserved(default_database, torch_device):
    resolver = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch_device
    )
    assert resolver.atom_type_index[-1] is None
    assert numpy.isnan(resolver.type_params.lj_radius[-1].cpu().item())
