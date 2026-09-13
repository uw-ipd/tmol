"""Pytest plugin replaying the pre-fix Python predicates on current regressions.

This is a controlled module comparison, not an unmodified upstream checkout.
"""

import subprocess
from types import FunctionType

BASELINE = "77f5d94190f4c358fc4cad13b9e97d975c446fb8"


def load(path):
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], text=True)
    namespace = {}
    exec(compile(source, f"baseline:{path}", "exec"), namespace)
    return namespace


def pytest_configure(config):
    import tmol.io.details._build_missing_leaf_atoms as leaf
    import tmol.pose._conjugated_groups as groups
    from tmol.pack.rotamer import OptHSampler

    original = load("tmol/io/details/_build_missing_leaf_atoms.py")
    leaf._uaid_for_at = original["_uaid_for_at"]
    original = load("tmol/pose/_conjugated_groups.py")
    function = original["lockstep_group_for_block"]
    groups.lockstep_group_for_block = FunctionType(function.__code__, groups.__dict__)
    original = load("tmol/pack/rotamer/_opth_sampler.py")["OptHSampler"]
    for name in (
        "_annotate_residue_type",
        "defines_rotamers_for_rt",
        "first_sc_atoms_for_rt",
    ):
        setattr(OptHSampler, name, getattr(original, name))
