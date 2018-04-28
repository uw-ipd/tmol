# Import support fixtures
from .support.rosetta import (
    pyrosetta,
    rosetta_database,
)

# Import basic data fixtures
from .data import (
    ubq_pdb,
    ubq_res,
    ubq_system,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
)


def pytest_collection_modifyitems(session, config, items):

    # Run all linting-tests *after* the functional tests
    items[:] = sorted(
        items, key=lambda i: i.nodeid.startswith("tmol/tests/linting")
    )


__all__ = (
    pyrosetta,
    rosetta_database,
    ubq_pdb,
    ubq_res,
    ubq_system,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
)
