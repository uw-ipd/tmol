"""Tests for the mol2/``.params`` fixture-integrity gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tmol.ligand.fixture_integrity import (
    FixtureMismatch,
    read_mol2_summary,
    require_paired_fixture,
)
from tmol.ligand.params_reference import parse_reference_params

_GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
_ORPHAN_MOL2 = _GROUND_TRUTH / "ref1.mol2"
_ORPHAN_PARAMS = _GROUND_TRUTH / "ref1.params"
_DUD80 = _GROUND_TRUTH / "dud80"

# Ethane: heavy C1, C2; six hydrogens. The mol2 keeps generic H names while the
# params re-derive them by attachment (HC*), exactly as mol2genparams does.
_ETHANE_MOL2 = """@<TRIPOS>MOLECULE
{name}
 8 7 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
      1 C1   0.000 0.000 0.000 C.3 1 LIG -0.060
      2 C2   1.500 0.000 0.000 C.3 1 LIG -0.060
      3 H1   -0.4  1.0   0.0   H   1 LIG  0.020
      4 H2   -0.4  -0.5  0.9   H   1 LIG  0.020
      5 H3   -0.4  -0.5 -0.9   H   1 LIG  0.020
      6 H4   1.9   1.0   0.0   H   1 LIG  0.020
      7 H5   1.9   -0.5  0.9   H   1 LIG  0.020
      8 H6   1.9   -0.5 -0.9   H   1 LIG  0.020
@<TRIPOS>BOND
     1 1 2 1
     2 1 3 1
     3 1 4 1
     4 1 5 1
     5 2 6 1
     6 2 7 1
     7 2 8 1
"""

_ETHANE_PARAMS = """NAME {name}
IO_STRING LIG Z
TYPE LIGAND
AA UNK
ATOM C1   CT  X  -0.060
ATOM C2   CT  X  -0.060
ATOM HC1  Hapo X  0.020
ATOM HC2  Hapo X  0.020
ATOM HC3  Hapo X  0.020
ATOM HC4  Hapo X  0.020
ATOM HC5  Hapo X  0.020
ATOM HC6  Hapo X  0.020
BOND_TYPE C1 C2 1
BOND_TYPE C1 HC1 1
BOND_TYPE C1 HC2 1
BOND_TYPE C1 HC3 1
BOND_TYPE C2 HC4 1
BOND_TYPE C2 HC5 1
BOND_TYPE C2 HC6 1
NBR_ATOM C1
NBR_RADIUS 999.0
"""


def _write_pair(tmp_path, mol2_name="etha", params_name="etha"):
    mol2 = tmp_path / "lig.mol2"
    params = tmp_path / "lig.params"
    mol2.write_text(_ETHANE_MOL2.format(name=mol2_name))
    params.write_text(_ETHANE_PARAMS.format(name=params_name))
    return mol2, params


def test_read_mol2_summary_counts():
    summary = read_mol2_summary(_ORPHAN_MOL2)
    # ref1.mol2 is the orphan molecule: 27 heavy of 51 atoms.
    assert summary.n_atoms == 51
    assert len(summary.heavy_names) == 27
    assert summary.has_hydrogen


def test_paired_fixture_passes(tmp_path):
    mol2, params = _write_pair(tmp_path)
    summary = require_paired_fixture(mol2, parse_reference_params(params))
    assert summary.title == "etha"
    assert summary.heavy_names == frozenset({"C1", "C2"})


def test_paired_fixture_accepts_params_path(tmp_path):
    mol2, params = _write_pair(tmp_path)
    # reference may be passed as a path, not just a parsed object.
    require_paired_fixture(mol2, params)


def test_orphan_ref1_is_rejected_with_heavy_count_mismatch():
    ref = parse_reference_params(_ORPHAN_PARAMS)
    with pytest.raises(FixtureMismatch) as exc:
        require_paired_fixture(_ORPHAN_MOL2, ref)
    msg = str(exc.value)
    assert "heavy-atom count" in msg
    # 27 heavy in the mol2 vs 17 heavy in the params.
    assert "mol2=27" in msg and "params=17" in msg


def test_residue_name_mismatch_is_rejected(tmp_path):
    mol2, params = _write_pair(tmp_path, mol2_name="etha", params_name="other")
    with pytest.raises(FixtureMismatch) as exc:
        require_paired_fixture(mol2, params)
    assert "residue name" in str(exc.value)


@pytest.mark.skipif(
    not (_DUD80 / "params" / "ace_1.params").exists(),
    reason="DUD80 dataset not present (git-ignored); paired-fixture check skipped",
)
def test_dud80_ace_1_is_a_valid_pair():
    mol2 = _DUD80 / "mol2" / "ace_1.mol2"
    params = _DUD80 / "params" / "ace_1.params"
    summary = require_paired_fixture(mol2, params)
    assert summary.title == "ace_1"
