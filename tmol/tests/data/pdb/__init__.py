import os
from glob import glob

from tmol.tests.data.util import LazyContentsMapping

data = LazyContentsMapping.from_list(
    glob(os.path.dirname(__file__) + "/*.pdb"),
    norm=lambda f: os.path.basename(f).split(".")[0].upper()
)
