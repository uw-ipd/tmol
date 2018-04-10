import os
import toolz

try:
    import pyrosetta
except ImportError:
    pyrosetta = None

if "ROSETTA_DATABASE" in os.environ:
    normpath = toolz.compose(
        os.path.abspath,
        os.path.expanduser,
        os.path.expandvars,
    )
    rosetta_database = normpath(os.environ.get("ROSETTA_DATABASE"))
elif pyrosetta is not None:
    rosetta_database = os.path.join(
        os.path.dirname(pyrosetta.__file__), "database"
    )
else:
    rosetta_database = None
