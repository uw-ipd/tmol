"""Smoke tests for the mol2 -> params helper and the workflow example (AC-8)."""

from __future__ import annotations

import tmol.ligand
from tmol.ligand import write_params_from_mol2
from tmol.ligand.params_reference import parse_reference_params

# A minimal, MMFF-trivial aliphatic ligand (ethane) so the smoke test does not
# depend on git-ignored data and is not affected by host-rdkit aromatic-MMFF
# limitations. Realistic staggered coordinates so 3D/ICOOR construction works.
_ETHANE_MOL2 = """@<TRIPOS>MOLECULE
etha
 8 7 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
      1 C1   0.000  0.000  0.000 C.3 1 LIG -0.060
      2 C2   1.540  0.000  0.000 C.3 1 LIG -0.060
      3 H1  -0.363  1.027  0.000 H   1 LIG  0.020
      4 H2  -0.363 -0.513  0.889 H   1 LIG  0.020
      5 H3  -0.363 -0.513 -0.889 H   1 LIG  0.020
      6 H4   1.903  0.513  0.889 H   1 LIG  0.020
      7 H5   1.903  0.513 -0.889 H   1 LIG  0.020
      8 H6   1.903 -1.027  0.000 H   1 LIG  0.020
@<TRIPOS>BOND
     1 1 2 1
     2 1 3 1
     3 1 4 1
     4 1 5 1
     5 2 6 1
     6 2 7 1
     7 2 8 1
"""


def test_write_params_from_mol2_is_exported():
    assert hasattr(tmol.ligand, "write_params_from_mol2")
    assert "write_params_from_mol2" in tmol.ligand.__all__


def test_example_module_imports_against_real_api():
    # The example previously imported undefined names and raised ImportError.
    import tmol.ligand.examples.workflow_from_mol2 as example

    assert hasattr(example, "run")
    assert hasattr(example, "main")


def test_write_params_from_mol2_produces_valid_params(tmp_path):
    mol2 = tmp_path / "etha.mol2"
    mol2.write_text(_ETHANE_MOL2)
    out = tmp_path / "out.params"
    prep = write_params_from_mol2(str(mol2), str(out), res_name="LG1")
    assert out.exists()
    ref = parse_reference_params(out)
    assert len(ref.atoms) == 8  # 2 carbons + 6 hydrogens
    assert ref.name
    assert len(ref.charges) == len(ref.atoms)
    assert len(prep.residue_type.atoms) == len(ref.atoms)
