from os.path import dirname, basename, isfile, join
import glob


modules = glob.glob(join(dirname(__file__), "*.py"))
exclude = [join(dirname(__file__), f) for f in ["score_type_factory.py", "__init__.py"]]

__all__ = [
    basename(f)[:-3]
    for f in modules
    if isfile(f) and not f.endswith("__init__.py") and f not in exclude
]
