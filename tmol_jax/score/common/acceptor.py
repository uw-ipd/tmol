"""Host-side acceptor frames shared by hydrogen bonds and implicit waters."""


def acceptor_bases(atom, hybridization, neighbors, is_hydrogen, block=lambda _: 0):
    """Select B/B0 using current native sp2, sp3 and ring rules.

    neighbors preserves the chemical bond order. For sp2, a walk that leaves
    the starting residue cannot return to that residue for the second base.
    """
    adjacent = tuple(neighbors(atom))
    base = next((i for i in adjacent if not is_hydrogen(i)), -1)
    if base < 0:
        return -1, -1
    if hybridization == 1:
        left_residue = block(base) != block(atom)
        second = next(
            (
                i
                for i in neighbors(base)
                if i != atom and not (left_residue and block(i) == block(atom))
            ),
            -1,
        )
    elif hybridization == 2:
        second = next((i for i in adjacent if i != base and is_hydrogen(i)), -1)
        if second < 0:
            second = next((i for i in adjacent if i != base), base)
    elif hybridization == 3:
        second = next((i for i in adjacent if i != base and not is_hydrogen(i)), -1)
    else:
        return -1, -1
    return (base, second) if second >= 0 else (-1, -1)
