import toolz
import os
import importlib
import pytest


@pytest.fixture(scope="session")
def pyrosetta():
    "Import and initialize pyrosetta."
    try:
        from tmol.support.rosetta.init import pyrosetta

        return pyrosetta
    except ImportError:
        return None


@pytest.fixture(scope="session")
def rosetta_database(pyrosetta):
    "Resolve path to rosetta database."

    if "ROSETTA_DATABASE" in os.environ:
        normpath = toolz.compose(
            os.path.abspath, os.path.expanduser, os.path.expandvars
        )
        return normpath(os.environ.get("ROSETTA_DATABASE"))
    elif pyrosetta:
        return pyrosetta._rosetta_database_from_env()
    else:
        return None


pyrosetta_available = True if importlib.util.find_spec("pyrosetta") else False

requires_pyrosetta = pytest.mark.skipif(
    not pyrosetta_available, reason="Requires pyrosetta."
)

rosetta_database_available = pyrosetta_available or "ROSETTA_DATABASE" in os.environ

requires_rosetta_database = pytest.mark.skipif(
    not rosetta_database_available,
    reason="Requires rosetta database via pyrosetta or ROSETTA_DATABASE env.",
)
