"""CIF charge-source independence.

A CIF's heavy-atom types must not depend on whether its partial charges are
trusted (present in the file) or recomputed (absent -> OpenBabel MMFF94).
"""

import glob
import os
from pathlib import Path

import attr
import pytest

from tmol.ligand.detect import nonstandard_residue_info_from_cif
from tmol.ligand.preparation import prepare_single_ligand

PLI_CIF = Path(__file__).parent.parent / "data" / "protein_ligand_test" / "cif_inputs"
TARGETS = sorted(
    os.path.basename(p)[: -len(".ligand.cif")]
    for p in glob.glob(str(PLI_CIF / "*.ligand.cif"))
)


def _heavy_types(prep) -> dict[str, str]:
    return {
        a.name: a.atom_type
        for a in prep.residue_type.atoms
        if not a.name.startswith("H")
    }


@pytest.mark.parametrize("target", TARGETS)
def test_types_independent_of_charge_source(target):
    info = nonstandard_residue_info_from_cif(
        PLI_CIF / f"{target}.ligand.cif", res_name="LG1"
    )
    trust = prepare_single_ligand(info, ph=7.4)  # CIF charges trusted
    recompute = prepare_single_ligand(  # charges dropped -> recomputed
        attr.evolve(info, partial_charges=None, skip_protonation=True),
        ph=7.4,
        charge_mode="mmff94",
    )
    assert _heavy_types(trust) == _heavy_types(recompute)
    assert all(c == c for c in recompute.partial_charges.values())  # finite
