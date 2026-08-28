"""Compiled constraint potentials."""

# get_torsion_angle is not re-exported here: callers import from .compiled directly
# so that monkeypatching sys.modules['...compiled'] in tests still works.
