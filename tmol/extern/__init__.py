from .test_toposort import TestAll, TestCase, TestCaseAll  # noqa: F401
from .toposort import CircularDependencyError, toposort, toposort_flatten  # noqa: F401

def include_paths():
    """Get -I compatible include dirs for external modules."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__))]
