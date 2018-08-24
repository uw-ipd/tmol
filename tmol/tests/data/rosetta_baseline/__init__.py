import os
from glob import glob

from tmol.tests.data.util import LazyPickleMapping

data = LazyPickleMapping.from_list(
    glob(os.path.dirname(__file__) + "/*.scores.pickle"),
    norm=lambda f: os.path.basename(f).split(".")[0].upper(),
)
