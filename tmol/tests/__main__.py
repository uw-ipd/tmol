import sys
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    import warnings
    warnings.warn("Unable to import faulthandler, install for compiled-module traceback support.")
    pass

from ..tests import run
sys.exit(not run().wasSuccessful())
