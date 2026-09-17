"""Optional GLY symmetry includes generated geometry and equivalent H targets."""

import attr
import numpy
import pytest

from tmol.tests.database.test_d_amino_acids import ideal_coords


def test_symmetric_gly_geometry_is_private_and_idempotent(default_database):
    source = default_database
    symmetric = source.with_symmetric_gly()
    repeated = symmetric.with_symmetric_gly()
    checked = 0
    for original, modified, twice in zip(
        source.chemical.residues,
        symmetric.chemical.residues,
        repeated.chemical.residues,
    ):
        if original.base_name != "GLY":
            assert original is modified is twice
            continue
        checked += 1
        old = {ic.name: ic for ic in original.icoors}
        new = {ic.name: ic for ic in modified.icoors}
        assert old["HA2"].d != old["HA3"].d
        mean = (old["HA2"].d + old["HA3"].d) / 2
        assert new["HA2"].d == new["HA3"].d == mean
        assert modified == twice
        xyz = ideal_coords(modified)
        normal = numpy.cross(xyz["N"] - xyz["CA"], xyz["C"] - xyz["CA"])
        normal /= numpy.linalg.norm(normal)
        first, second = (xyz[name] - xyz["CA"] for name in ("HA2", "HA3"))
        reflected = first - 2 * numpy.dot(first, normal) * normal
        numpy.testing.assert_allclose(reflected, second, atol=2e-6, rtol=0)
    assert checked >= 4  # base, N/C termini, and combined termini
    before = source.scoring.cartbonded
    after = symmetric.scoring.cartbonded
    assert before.hash != after.hash == repeated.scoring.cartbonded.hash
    assert before.connection_params is after.connection_params
    for name, record in before.residue_params.items():
        if name != "GLY":
            assert after.residue_params[name] is record
    old_pair = [
        p for p in before.residue_params["GLY"].length_parameters if "HA" in p.atm2
    ]
    pair = [p for p in after.residue_params["GLY"].length_parameters if "HA" in p.atm2]
    assert old_pair[0].x0 != old_pair[1].x0
    assert pair[0].x0 == pair[1].x0 == pytest.approx(sum(p.x0 for p in old_pair) / 2)


def test_symmetric_gly_averages_private_unequal_force_constants(default_database):
    from tmol.database.scoring._cartbonded import CartBondedDatabase

    original = default_database.scoring.cartbonded
    params = dict(original.residue_params)
    gly = params["GLY"]
    params["GLY"] = attr.evolve(
        gly,
        length_parameters=tuple(
            attr.evolve(p, K=2 * p.K) if p.atm2 == "HA2" else p
            for p in gly.length_parameters
        ),
    )
    private = attr.evolve(
        default_database,
        scoring=attr.evolve(
            default_database.scoring,
            cartbonded=CartBondedDatabase.from_cartres_dict(params),
        ),
    ).with_symmetric_gly()
    pair = [
        p
        for p in private.scoring.cartbonded.residue_params["GLY"].length_parameters
        if p.atm2 in ("HA2", "HA3")
    ]
    assert pair[0].K == pair[1].K == 495

    def energy(a, b):
        return sum(p.K * (length - p.x0) ** 2 for p, length in zip(pair, (a, b)))

    for first, second in ((1.08, 1.11), (1.1, 1.0)):
        assert energy(first, second) == pytest.approx(energy(second, first))
