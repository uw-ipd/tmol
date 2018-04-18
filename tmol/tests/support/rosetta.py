import os
import importlib
import pytest

pyrosetta_available = True if importlib.util.find_spec("pyrosetta") else False

requires_pyrosetta = pytest.mark.skipif(
    not pyrosetta_available, reason="Requires pyrosetta."
)

rosetta_database_available = pyrosetta_available or "ROSETTA_DATABASE" in os.environ

requires_rosetta_database = pytest.mark.skipif(
    not rosetta_database_available,
    reason="Requires rosetta database via pyrosetta or ROSETTA_DATABASE env."
)
