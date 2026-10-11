"""Small runtime helpers stay independent of the native extension test suite."""

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_extern_import_does_not_load_tests_and_vendored_tests_still_pass():
    # Load the real package without importing tmol's unrelated native runtime.
    code = """
import importlib, pathlib, sys, types, unittest
tmol = types.ModuleType('tmol')
tmol.__path__ = [str(pathlib.Path.cwd() / 'tmol')]
sys.modules['tmol'] = tmol
extern = importlib.import_module('tmol.extern')
assert 'tmol.extern.test_toposort' not in sys.modules
assert list(extern.toposort({'a': set()})) == [{'a'}]
tests = importlib.import_module('tmol.extern.test_toposort')
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(tests))
raise SystemExit(not result.wasSuccessful())
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_mapping_values_are_a_live_view():
    spec = importlib.util.spec_from_file_location(
        "_tmol_dicttoolz", ROOT / "tmol/utility/_dicttoolz.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = {"first": 1}
    view = module.vals(mapping)
    mapping["second"] = 2
    assert list(view) == [1, 2]
