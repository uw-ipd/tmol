import numpy

from functools import singledispatch


@singledispatch
def to_pdb(system):
    raise NotImplementedError(f"Unknown system: {system}")


@singledispatch
def to_cdjson(system):
    raise NotImplementedError(f"Unknown system: {system}")


def pack_cdjson(coords, elems, bonds):
    """Pack coordinate list, element list, and bond pairs into cdjson format.

    See:
    https://github.com/3dmol/3Dmol.js/blob/master/3Dmol/parsers.js#L698
    """

    return {"m": [{
        "a": [
            {
                "l": str(e),
                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])
            }
            for e, c in zip(elems, coords)
        ],
        "b": [
            {
                "b": int(b),
                "e": int(e)
            }
            for b, e in bonds
            if not numpy.any(numpy.isnan(coords[b]) | numpy.isnan(coords[e]))
        ]
    }]}  # yapf: disable
