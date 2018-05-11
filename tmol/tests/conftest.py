import threading
import pytest

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


@pytest.fixture
def pytorch_backward_coverage(cov):
    """Torch hook to enable coverage in backward pass.

    Returns a hook function used to enable coverage tracing during
    pytorch backward passes. Torch runs all backward passes in a
    non-main thread, not spawned by the standard 'threading'
    interface, so coverage does not trace the thread.

    Example:

    result = custom_func(input)

    # enable the hook
    result.register_hook(pytorch_backward_coverage)

    # call backward via sum so hook fires before custom_op backward
    result.sum().backward()
    """

    if cov:
        cov.collector.added_tracers = {threading.get_ident()}

        def add_tracer(_):
            tid = threading.get_ident()
            if tid not in cov.collector.added_tracers:
                print(f"pytorch backward trace: {tid}")
                cov.collector.added_tracers.add(tid)
                cov.collector._start_tracer()
    else:

        def add_tracer(_):
            pass

    return add_tracer


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
