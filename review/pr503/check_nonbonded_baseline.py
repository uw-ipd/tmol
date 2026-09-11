"""Run current reuse regressions with the preceding Python scoring implementations.

Native sources did not change; the import overlay preserves the same checkout
paths and compiled extensions. No repository files or installed modules change.
"""

import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys

BASELINE = "841d594d5bd963755093ff903014e953bd7776ad"
ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "tmol.score._atom_type_dependent_term",
    "tmol.score.ljlk._ljlk_energy_term",
    "tmol.score.hbond._hbond_dependent_term",
    "tmol.score.hbond._hbond_energy_term",
    "tmol.score.lk_ball._lk_ball_energy_term",
}


class BaselineImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname in MODULES:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__.replace(".", "/") + ".py"
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BASELINE}:{relative}"], text=True
        )
        module.__file__ = str(ROOT / relative)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


if __name__ == "__main__":
    sys.meta_path.insert(0, BaselineImports())
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
