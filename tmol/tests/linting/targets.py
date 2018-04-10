import os
import fnmatch
import glob
import tmol

basedir = os.path.dirname(tmol.__file__)
ignore_dirs = ["*/tmol/extern/*"]

lint_files = [
    f for f in glob.glob(f"{basedir}/**/*.py", recursive=True)
    if not any(fnmatch.fnmatch(f, d) for d in ignore_dirs)
]
