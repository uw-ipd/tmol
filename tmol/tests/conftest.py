import toolz
import os
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
            os.path.abspath,
            os.path.expanduser,
            os.path.expandvars,
        )
        return normpath(os.environ.get("ROSETTA_DATABASE"))
    elif pyrosetta:
        return pyrosetta._rosetta_database_from_env()
    else:
        return None


def pytest_collection_modifyitems(session, config, items):

    # Run all linting-tests *after* the functional tests
    items[:] = sorted(
        items, key=lambda i: i.nodeid.startswith("tmol/tests/linting")
    )
