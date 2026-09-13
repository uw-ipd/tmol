"""Run fragment regressions against an isolated upstream selection module.

The fixtures and their preparation use the improvement branch. Only selection
functions come from the requested Git revision. A regression reproduction
returns pytest's nonzero exit status; this is not a pristine-tree import test.
"""

import argparse
import subprocess
import sys
import types

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default="0f4c3bc426bca78e8681f0b730fa23c3e26ef261"
    )
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", args.revision + ":tmol/io/details/_select_from_canonical.py"],
        text=True,
    )
    name = "tmol.io.details._review_upstream_selection"
    module = types.ModuleType(name)
    module.__package__ = "tmol.io.details"
    sys.modules[name] = module
    exec(compile(source, "upstream_select_from_canonical.py", "exec"), module.__dict__)

    class SelectionOverride:
        def pytest_collection_modifyitems(self, items):
            # Apply after pytest imports/rewrites the test module.
            for item in items:
                item.module.selection = module

    print("Selection baseline:", args.revision, flush=True)
    return pytest.main(
        ["-q", "tmol/tests/io/details/test_fragment_selection_identity.py"],
        plugins=[SelectionOverride()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
