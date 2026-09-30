"""One alternate location per linked group of residues, as AtomWorks keeps."""

import numpy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# heavy atoms of two residues this close are bonded, coordinated or overlapping
LINK_DISTANCE = 2.2
# shorter than any bond between two residues: the atoms are alternates of one site
OVERLAP_DISTANCE = 1.2
NO_ALTERNATE = ("", " ", ".", "?")


def one_alternate_per_group(residue, chain, altloc, heavy, coord):
    """Mask of the atoms keeping one alternate per linked group of residues.

    Residues with alternates are linked when they share a chain or when an
    alternate heavy atom lies within ``LINK_DISTANCE`` of another residue; each
    group keeps its first letter. A residue without that letter keeps its first
    alternate overlapping no kept atom, or none. 1I54 writes its heme (altloc A)
    and Zn-porphyrin (altloc B) as two residues bonded to the same cysteines.

    ``residue`` and ``chain`` label each atom's residue (chain, number and
    insertion code) and chain; ``heavy`` masks the non-hydrogen atoms.
    """
    altloc = numpy.asarray(altloc).astype(str)
    is_alt = ~numpy.isin(altloc, NO_ALTERNATE)
    keep = ~is_alt
    if not is_alt.any():
        return keep
    _, res = numpy.unique(numpy.asarray(residue), return_inverse=True)
    _, chain = numpy.unique(numpy.asarray(chain), return_inverse=True)
    heavy = numpy.flatnonzero(heavy & numpy.isfinite(coord).all(axis=-1))
    alt_heavy = heavy[is_alt[heavy]]
    near = cKDTree(coord[heavy]).query_ball_point(coord[alt_heavy], LINK_DISTANCE)
    rows = numpy.concatenate(
        [numpy.repeat(res[alt_heavy], [len(n) for n in near]), res[is_alt]]
    )
    cols = numpy.concatenate(
        [
            res[heavy[[j for n in near for j in n]]].reshape(-1),
            res.max() + 1 + chain[is_alt],
        ]
    )
    size = res.max() + chain.max() + 2
    graph = coo_matrix((numpy.ones(len(rows)), (rows, cols)), shape=(size, size))
    group = connected_components(graph)[1][res]
    for g in numpy.unique(group[is_alt]):
        keep |= (group == g) & (altloc == min(altloc[is_alt & (group == g)]))
    lacking = numpy.flatnonzero(is_alt & ~numpy.isin(res, res[keep & is_alt]))
    kept = heavy[keep[heavy]]
    tree = cKDTree(coord[kept])
    for r in numpy.unique(res[lacking]):
        for letter in sorted(set(altloc[lacking[res[lacking] == r]])):
            own = lacking[(res[lacking] == r) & (altloc[lacking] == letter)]
            probe = own[numpy.isin(own, heavy)]
            hits = tree.query_ball_point(coord[probe], OVERLAP_DISTANCE)
            if not any((res[kept[n]] != r).any() for n in hits if n):
                keep[own] = True
                break
    return keep


def selected_residue_names(residue, res_name, altloc, keep):
    """Each atom's residue name after selection: that of the kept alternate.

    At a microheterogeneity site (1EJG A:22 PRO/SER) atoms without a letter may be
    written under the residue that was not kept.
    """
    residue = numpy.asarray(residue)
    res_name = numpy.asarray(res_name).astype(object)
    kept_alt = keep & ~numpy.isin(numpy.asarray(altloc).astype(str), NO_ALTERNATE)
    selected = dict(zip(residue[kept_alt], res_name[kept_alt], strict=True))
    return numpy.array(
        [selected.get(r, n) for r, n in zip(residue, res_name, strict=True)],
        dtype=object,
    )
