from .rosetta import requires_pyrosetta

@requires_pyrosetta
def test_fixture(pyrosetta):
    assert pyrosetta
