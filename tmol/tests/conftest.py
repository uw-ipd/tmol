# Import support fixtures
from .support.rosetta import (  # noqa: F401
    pyrosetta, rosetta_database,
)

# Import basic data fixtures
from .data import (  # noqa: F401
    ubq_pdb, ubq_res, ubq_system
)


def pytest_collection_modifyitems(session, config, items):

    # Run all linting-tests *after* the functional tests
    items[:] = sorted(
        items, key=lambda i: i.nodeid.startswith("tmol/tests/linting")
    )
