from tmol.utility.cpp_extension import load, relpaths, modulename

_distpairs = load(modulename(__name__), relpaths(__file__, ["dist.cu"]))


def triu_distpairs(*args, **kwargs):
    return _distpairs.triu_distpairs(*args, **kwargs)
