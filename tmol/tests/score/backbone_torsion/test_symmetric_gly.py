"""Mirror-symmetric glycine backbone tables.

Glycine is achiral, so its backbone preferences must be invariant under
phi,psi -> -phi,-psi. The tables derived from PDB statistics are not, which is
why a structure and its mirror image do not score alike by default.
``ParameterDatabase.with_symmetric_gly()`` selects symmetrized tables instead.
"""

import numpy
import pytest

from tmol.database import ParameterDatabase

GLY_TABLES = {"GLY": "GLY_symm", "GLY_prepro": "GLY_prepro_symm"}


def mirror(grid):
    """Reflect a periodic phi/psi grid through the origin (bin k -> -k)."""
    return numpy.roll(numpy.roll(grid[::-1, ::-1], 1, axis=0), 1, axis=1)


def rama_tables(param_db):
    return {
        t.table_id: numpy.asarray(t.table) for t in param_db.scoring.rama.rama_tables
    }


def rama_rows(param_db):
    return {
        (m.res_middle, m.res_upper): m.table_id
        for m in param_db.scoring.rama.rama_lookup
    }


def omega_rows(param_db):
    return {
        (m.res_middle, m.res_upper): m.table_id
        for m in param_db.scoring.omega_bbdep.bbdep_omega_lookup
    }


@pytest.mark.parametrize("source,symm", sorted(GLY_TABLES.items()))
def test_symmetrized_table_is_symmetric(source, symm) -> None:
    tables = rama_tables(ParameterDatabase.get_default())
    assert numpy.abs(tables[symm] - mirror(tables[symm])).max() == pytest.approx(
        0.0, abs=1e-5
    )
    # the table it replaces is not, which is the whole reason it exists
    assert numpy.abs(tables[source] - mirror(tables[source])).max() > 1.0


@pytest.mark.parametrize("source,symm", sorted(GLY_TABLES.items()))
def test_symmetrized_table_stays_on_the_same_energy_scale(source, symm) -> None:
    """Symmetrizing redistributes the potential; it must not shift it bodily."""
    tables = rama_tables(ParameterDatabase.get_default())
    shift = tables[symm] - tables[source]
    assert numpy.abs(shift.mean()) < 0.5
    assert numpy.sqrt((shift**2).mean()) < 1.5


def test_with_symmetric_gly_retargets_only_glycine() -> None:
    base = ParameterDatabase.get_default()
    symm = base.with_symmetric_gly()

    changed = {k: v for k, v in rama_rows(symm).items() if rama_rows(base)[k] != v}
    assert changed == {("GLY", "_"): "GLY_symm", ("GLY", "PRO"): "GLY_prepro_symm"}

    changed = {k: v for k, v in omega_rows(symm).items() if omega_rows(base)[k] != v}
    assert changed == {
        ("GLY", "_"): "gly_symm",
        ("GLY", "PRO"): "prepro_gly_symm",
    }
    # the prepro table glycine used to share is still what everyone else uses
    assert omega_rows(symm)[("ALA", "PRO")] == "prepro"


def test_with_symmetric_gly_changes_the_cache_identity() -> None:
    """The databases differ, so anything keyed on their id must see two keys."""
    base = ParameterDatabase.get_default()
    symm = base.with_symmetric_gly()
    assert symm.scoring.rama.uniq_id != base.scoring.rama.uniq_id
    assert symm.scoring.omega_bbdep.uniq_id != base.scoring.omega_bbdep.uniq_id
    assert symm.scoring.rama.uniq_id == symm.scoring.rama.content_id()


def test_with_symmetric_gly_leaves_the_default_alone() -> None:
    base = ParameterDatabase.get_default()
    before = rama_rows(base)
    base.with_symmetric_gly()
    assert rama_rows(ParameterDatabase.get_default()) == before
    assert before[("GLY", "_")] == "GLY"


@pytest.mark.parametrize("table_id", ["gly_symm", "prepro_gly_symm"])
def test_symmetric_gly_omega_is_uniformly_trans(table_id) -> None:
    """An achiral residue has no backbone-dependent omega preference."""
    tables = {
        t.table_id: t
        for t in ParameterDatabase.get_default().scoring.omega_bbdep.bbdep_omega_tables
    }
    mu = numpy.asarray(tables[table_id].mu)
    sigma = numpy.asarray(tables[table_id].sigma)
    assert mu.min() == mu.max() == pytest.approx(180.0)
    assert sigma.min() == sigma.max() == pytest.approx(6.0)


def test_shared_omega_tables_are_untouched() -> None:
    """Glycine got its own copies; the tables it shared keep their variation."""
    tables = {
        t.table_id: t
        for t in ParameterDatabase.get_default().scoring.omega_bbdep.bbdep_omega_tables
    }
    for table_id in ("gly", "prepro"):
        assert numpy.asarray(tables[table_id].mu).std() > 0.0
