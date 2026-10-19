"""Chemical (Lewis) bond orders for a residue type's intra-residue bonds."""

from collections import defaultdict

from tmol.chemical._restypes import BondType


def _order_name(order):
    return order.upper() if isinstance(order, str) else BondType(int(order)).name


def chemical_bond_orders(block_type, chem_db):
    """One Lewis structure for a block type's bonds, by atom-name pair: io_bond_orders,
    then non-ring AROMATIC: one double to a terminal heteroatom per atom, rest single.
    Ring AROMATIC bonds stay AROMATIC.
    """
    element_for_type = {at.name: at.element for at in chem_db.atom_types}
    element_of = {a.name: element_for_type[a.atom_type] for a in block_type.atoms}
    order_of, in_ring = {}, {}
    for a, b, order, *rest in block_type.bonds:
        key = frozenset((a, b))
        order_of[key] = _order_name(order)
        in_ring[key] = bool(rest[0]) if rest else False
    for a, b, order in block_type.io_bond_orders:
        order_of[frozenset((a, b))] = order
    neighbors = defaultdict(set)
    for key in order_of:
        a, b = tuple(key)
        neighbors[a].add(b)
        neighbors[b].add(a)

    def terminal_heteroatom(name):
        heavy = [n for n in neighbors[name] if element_of[n] != "H"]
        return element_of[name] in ("O", "N", "S") and len(heavy) == 1

    delocalized = [
        k for k, order in order_of.items() if order == "AROMATIC" and not in_ring[k]
    ]
    ends_of = defaultdict(list)
    for key in delocalized:
        for center, end in (tuple(key), tuple(key)[::-1]):
            if terminal_heteroatom(end):
                ends_of[center].append(end)
    for center in sorted(ends_of):
        has_double = any(
            order_of[frozenset((center, n))] == "DOUBLE"
            and element_of[n] in ("O", "N", "S")
            for n in neighbors[center]
        )
        for end in sorted(ends_of[center]):
            key = frozenset((center, end))
            if order_of[key] == "AROMATIC":
                order_of[key] = "SINGLE" if has_double else "DOUBLE"
                has_double = True
    for key in delocalized:
        if order_of[key] == "AROMATIC":
            order_of[key] = "SINGLE"
    return order_of
