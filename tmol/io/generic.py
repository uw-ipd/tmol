from functools import singledispatch

@singledispatch
def to_pdb(system):
    raise NotImplementedError(f"Unknown system: {system}")
