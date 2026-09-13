"""Declared hydrogen-bond families require a complete, finite pair table."""

import attr
import numpy
import pytest
import torch

from tmol.score.hbond._params import HBondParamResolver, CompactedHBondDatabase


@pytest.mark.parametrize("resolver", [HBondParamResolver, CompactedHBondDatabase])
def test_missing_declared_pair_rejected(default_database, torch_device, resolver):
    hb = default_database.scoring.hbond
    missing = hb.pair_parameters[0]
    hb = attr.evolve(hb, pair_parameters=hb.pair_parameters[1:])
    with pytest.raises(
        ValueError,
        match=f"Missing HBond pair.*{missing.donor_type}.*{missing.acceptor_type}",
    ):
        resolver.from_database(default_database.chemical, hb, torch_device)


@pytest.mark.parametrize(
    "field,value",
    [("c_0", float("nan")), ("xmin", float("inf")), ("min_val", -float("inf"))],
)
def test_nonfinite_referenced_polynomial_rejected(
    default_database, torch_device, field, value
):
    hb = default_database.scoring.hbond
    name = hb.pair_parameters[0].AHdist
    hb = attr.evolve(
        hb,
        polynomial_parameters=tuple(
            attr.evolve(p, **{field: value}) if p.name == name else p
            for p in hb.polynomial_parameters
        ),
    )
    with pytest.raises(ValueError, match=f"HBond polynomial.*{name}.*non-finite"):
        HBondParamResolver.from_database(default_database.chemical, hb, torch_device)


@pytest.mark.parametrize("kind", ["donor", "acceptor"])
@pytest.mark.parametrize("value", [float("nan"), 1e100])
def test_nonfinite_family_weight_rejected(default_database, torch_device, kind, value):
    hb = default_database.scoring.hbond
    key = kind + "_type_params"
    rows = getattr(hb, key)
    hb = attr.evolve(hb, **{key: (attr.evolve(rows[0], weight=value), *rows[1:])})
    with pytest.raises(ValueError, match=f"HBond {kind}.*{rows[0].name}.*non-finite"):
        HBondParamResolver.from_database(default_database.chemical, hb, torch_device)


@pytest.mark.parametrize("kind", ["donor_type", "acceptor_type", "AHdist"])
def test_unknown_pair_reference_is_actionable(default_database, torch_device, kind):
    hb = default_database.scoring.hbond
    hb = attr.evolve(
        hb,
        pair_parameters=(
            attr.evolve(hb.pair_parameters[0], **{kind: "UNREGISTERED"}),
            *hb.pair_parameters[1:],
        ),
    )
    with pytest.raises(ValueError, match="HBond.*UNREGISTERED"):
        HBondParamResolver.from_database(default_database.chemical, hb, torch_device)


@pytest.mark.parametrize("empty", ["donor", "acceptor", "both"])
def test_empty_family_can_omit_polynomials(default_database, torch_device, empty):
    hb = default_database.scoring.hbond
    fields = dict(pair_parameters=(), polynomial_parameters=())
    for kind in (["donor", "acceptor"] if empty == "both" else [empty]):
        fields[kind + "_type_params"] = ()
        fields[kind + "_atom_types"] = ()
        fields[kind + "_type_mapper"] = (
            getattr(hb, kind + "_type_mapper").iloc[:0].copy()
        )
    hb = attr.evolve(hb, **fields)
    result = HBondParamResolver.from_database(
        default_database.chemical, hb, torch_device
    )
    assert result.pair_params.AHdist.coeffs.numel() == 0
    compact = CompactedHBondDatabase.from_database(
        default_database.chemical, hb, torch_device
    )
    assert compact.pair_poly_table.numel() == 0
    assert torch.isfinite(compact.global_param_table).all()


def test_last_pair_and_polynomial_definition_wins(default_database, torch_device):
    hb = default_database.scoring.hbond
    first = hb.pair_parameters[0]
    poly = next(p for p in hb.polynomial_parameters if p.name == first.AHdist)
    replacement = attr.evolve(poly, name="replacement", c_0=0.123456789123)
    pair = attr.evolve(first, AHdist=replacement.name)
    hb = attr.evolve(
        hb,
        pair_parameters=(*hb.pair_parameters, pair),
        polynomial_parameters=(
            *hb.polynomial_parameters,
            attr.evolve(replacement, c_0=99.0),
            replacement,
        ),
    )
    result = HBondParamResolver.from_database(
        default_database.chemical, hb, torch_device
    )
    i = result.donor_type_index.get_loc(first.donor_type)
    j = result.acceptor_type_index.get_loc(first.acceptor_type)
    assert result.pair_params.AHdist.coeffs[i, j, -1].item() == replacement.c_0
    numpy.testing.assert_array_equal(
        result.pair_params.AHdist.range[i, j].cpu(),
        [replacement.xmin, replacement.xmax],
    )
