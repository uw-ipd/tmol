"""Mirrored rama and bbdep-omega tables for D-amino acids.

A D residue's backbone statistics are its L counterpart's read at negated
phi/psi. Rather than negate torsions in the kernel, the resolver stores a
mirrored copy of every table and a D block type selects it by index.
"""

import cattr
import pandas
import pytest
import torch

from tmol.chemical._restypes import RefinedResidueType
from tmol.database import ParameterDatabase
from tmol.score.backbone_torsion._bb_torsion_energy_term import (
    BackboneTorsionEnergyTerm,
    _invert_flag,
)
from tmol.score.backbone_torsion._params import (
    BackboneTorsionParamResolver,
    mirror_grid,
)


@pytest.fixture(scope="module")
def param_db():
    return ParameterDatabase.get_default()


@pytest.fixture(scope="module")
def resolver(param_db):
    return BackboneTorsionParamResolver.from_database(
        param_db.scoring.rama, param_db.scoring.omega_bbdep, torch.device("cpu")
    )


@pytest.fixture(scope="module")
def term(param_db):
    return BackboneTorsionEnergyTerm(param_db=param_db, device=torch.device("cpu"))


def block_type_params(term, param_db, name):
    residue = next(r for r in param_db.chemical.residues if r.name == name)
    refined = cattr.structure(cattr.unstructure(residue), RefinedResidueType)
    term.setup_block_type(refined)
    return refined.backbone_torsion_params


def test_mirror_is_an_involution(param_db) -> None:
    for table in param_db.scoring.rama.rama_tables:
        raw = table.table.to(torch.float)
        once = mirror_grid(raw, table.bbstart, table.bbstep)
        twice = mirror_grid(once, table.bbstart, table.bbstep)
        assert torch.equal(twice, raw), table.table_id


def test_mirror_follows_the_grid_registration(param_db) -> None:
    """The rama and omega grids are offset by half a bin from each other.

    rama bins are centered on -180 + 10k and omega on -175 + 10k, so the bin
    holding -x is not at the same offset in the two; a single hardcoded
    reflection would be wrong for one of them.
    """
    rama = param_db.scoring.rama.rama_tables[0]
    omega = param_db.scoring.omega_bbdep.bbdep_omega_tables[0]
    assert rama.bbstart[0] != omega.bbstart[0]

    ramp = torch.arange(36, dtype=torch.float).expand(36, 36).clone()
    rama_mirror = mirror_grid(ramp, rama.bbstart, rama.bbstep)
    omega_mirror = mirror_grid(ramp, omega.bbstart, omega.bbstep)
    # rama sends bin k to -k, omega sends it to n-1-k
    assert rama_mirror[0, 0].item() == 0.0
    assert omega_mirror[0, 0].item() == 35.0


def test_symmetric_table_is_its_own_mirror(param_db) -> None:
    """Cross-check against the symmetrized glycine tables."""
    tables = {t.table_id: t for t in param_db.scoring.rama.rama_tables}
    for table_id in ("GLY_symm", "GLY_prepro_symm"):
        table = tables[table_id]
        raw = table.table.to(torch.float)
        assert torch.allclose(
            mirror_grid(raw, table.bbstart, table.bbstep), raw, atol=1e-5
        )


def test_resolver_stacks_a_mirror_for_every_table(resolver, param_db) -> None:
    n_rama = len(param_db.scoring.rama.rama_tables)
    n_omega = len(param_db.scoring.omega_bbdep.bbdep_omega_tables)
    assert resolver.n_rama_tables == n_rama
    assert resolver.n_omega_tables == n_omega
    assert resolver.rama_params.tables.shape[0] == 2 * n_rama
    assert resolver.omega_params.tables.shape[0] == 2 * n_omega
    # the mirrored half is on the same grid as the half it mirrors
    assert torch.equal(
        resolver.rama_params.bbstarts[:n_rama], resolver.rama_params.bbstarts[n_rama:]
    )


def test_d_block_types_select_the_mirrored_tables(term, param_db) -> None:
    """Every D type, variants included, must mirror its L counterpart exactly."""
    by_name = {r.name: r for r in param_db.chemical.residues}
    n_rama = term.param_resolver.n_rama_tables
    n_omega = term.param_resolver.n_omega_tables

    pairs = [
        (n[1:], n) for n in sorted(by_name) if n.startswith("D") and n[1:] in by_name
    ]
    assert len(pairs) > 80, "expected the D residues and their termini variants"

    for l_name, d_name in pairs:
        left = block_type_params(term, param_db, l_name)
        right = block_type_params(term, param_db, d_name)
        for l_ind, d_ind, offset in (
            (left.rama_table_inds, right.rama_table_inds, n_rama),
            (left.omega_table_inds, right.omega_table_inds, n_omega),
        ):
            expected = [i + offset if i >= 0 else -1 for i in l_ind]
            assert list(d_ind) == expected, f"{l_name} vs {d_name}"
        assert left.is_pro[0] == right.is_pro[0], d_name


def test_a_terminus_keeps_its_base_type_tables(term, param_db) -> None:
    """A terminus still bonds to its neighbour, so it still needs an omega.

    The kernel skips omega entirely when the table index is negative, so a
    terminus that resolved to no table would silently lose the term.
    """
    for base in ("ALA", "DALA"):
        expected = block_type_params(term, param_db, base)
        for variant in (f"{base}:nterm", f"{base}:cterm"):
            params = block_type_params(term, param_db, variant)
            assert list(params.omega_table_inds) == list(
                expected.omega_table_inds
            ), variant
            assert list(params.rama_table_inds) == list(
                expected.rama_table_inds
            ), variant
            assert params.omega_table_inds[0] >= 0, variant


def test_half_inverted_lookup_row_is_rejected() -> None:
    """A mirror negates both torsions; inverting one is not expressible."""
    rows = pandas.DataFrame({"invert_phi": [True], "invert_psi": [False]})
    with pytest.raises(ValueError, match="only one of"):
        _invert_flag(rows, "DALA")


def test_every_amino_acid_block_type_has_backbone_tables(term, param_db) -> None:
    """A negative omega index makes the kernel skip the term entirely.

    CYD and HIS_POS are separate residue types sharing an equivalence class
    with CYS and HIS, so they must borrow that class's tables rather than
    resolve to nothing.
    """
    uncovered = []
    for restype in param_db.chemical.residues:
        if restype.properties.polymer.polymer_type != "amino_acid":
            continue
        params = block_type_params(term, param_db, restype.name)
        if params.omega_table_inds[0] < 0 or params.rama_table_inds[0] < 0:
            uncovered.append(restype.name)
    assert uncovered == []


@pytest.mark.parametrize(
    "borrower,parent", [("CYD", "CYS"), ("HIS_POS", "HIS"), ("HIS_D", "HIS")]
)
def test_tautomers_share_their_class_tables(term, param_db, borrower, parent) -> None:
    borrowed = block_type_params(term, param_db, borrower)
    expected = block_type_params(term, param_db, parent)
    assert list(borrowed.rama_table_inds) == list(expected.rama_table_inds)
    assert list(borrowed.omega_table_inds) == list(expected.omega_table_inds)
