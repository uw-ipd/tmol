import pytest

from tmol.tests.data.pdb import data as test_pdbs
from tmol.system.residue.io import read_pdb
from tmol.system.residue.packed import PackedResidueSystem

test_names = ["1QYS", "1UBQ"]


@pytest.mark.parametrize("structure", test_names)
def test_smoke_io(structure):

    for tname in test_names:
        pdb = test_pdbs[tname]
        read_pdb(pdb)


def test_water_system(water_box_res):
    water_system = PackedResidueSystem.from_residues(water_box_res)

    nwat = len(water_box_res)

    assert len(water_system.residues) == nwat

    assert water_system.block_size > 3
    assert len(water_system.atom_metadata) == nwat * water_system.block_size

    assert len(water_system.torsion_metadata) == 0
    assert len(water_system.connection_metadata) == 0
