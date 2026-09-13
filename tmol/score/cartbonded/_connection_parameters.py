"""Validate and pack explicit, connection-owned length/angle parameters."""

import math

import numpy

from tmol.score.common import add_to_hashtable


def _canonical_path(path):
    return min(tuple(path), tuple(reversed(path)))


def _validated_rows(record, first, second):
    """Encode block-local atoms: positive first-block, negative second-block.

    Adding one to atom indices reserves zero for padding. Validate complete
    coverage from chemical bonds, independently of native path enumeration.
    """
    types = (first, second)
    connection_names = (record.connection1, record.connection2)
    roots = []
    neighbors = []
    for side, (bt, connection) in enumerate(zip(types, connection_names)):
        conn = next((c for c in bt.connections if c.name == connection), None)
        if conn is None:
            raise ValueError(f"{bt.name}: no connection {connection!r}")
        root = bt.atom_to_idx[conn.atom]
        sign = 1 if side == 0 else -1
        roots.append(sign * (root + 1))
        nbrs = {
            bt.atom_to_idx[b] if a == conn.atom else bt.atom_to_idx[a]
            for a, b, *_ in bt.bonds
            if a == conn.atom or b == conn.atom
        }
        neighbors.append([sign * (atom + 1) for atom in sorted(nbrs)])
    expected = {_canonical_path(roots)}
    for side in (0, 1):
        expected.update(
            _canonical_path((neighbor, roots[side], roots[1 - side]))
            for neighbor in neighbors[side]
        )
    rows = {}
    for kind, parameters in enumerate(
        (record.length_parameters, record.angle_parameters)
    ):
        for parameter in parameters:
            if parameter.type != kind:
                raise ValueError("Connection parameter type disagrees with its group")
            if not math.isfinite(parameter.K) or parameter.K < 0:
                raise ValueError(
                    "Connection force constant must be finite and nonnegative"
                )
            if (
                not math.isfinite(parameter.x0)
                or parameter.x0 <= 0
                or (kind == 1 and parameter.x0 > math.pi)
            ):
                raise ValueError("Connection equilibrium distance/angle is invalid")
            path = []
            for field in ("atm1", "atm2", "atm3")[: kind + 2]:
                name = getattr(parameter, field)
                side = int(name.startswith("+"))
                atom_name = name[1:] if side else name
                if atom_name not in types[side].atom_to_idx:
                    raise ValueError(
                        f"{types[side].name}: no connection parameter atom {atom_name!r}"
                    )
                path.append(
                    (1 if side == 0 else -1) * (types[side].atom_to_idx[atom_name] + 1)
                )
            path = _canonical_path(path)
            if path in rows:
                raise ValueError("Duplicate connection bond/angle parameter")
            rows[path] = (kind, parameter.x0, parameter.K, 0, 0, 0, 0)
    if set(rows) != expected:
        raise ValueError(
            f"Incomplete or invalid connection parameters for {record.block_type1}/{record.connection1}"
            f" -- {record.block_type2}/{record.connection2}: "
            f"{len(expected - set(rows))} missing, {len(set(rows) - expected)} extraneous paths"
        )
    return rows


def compile_connection_parameters(records, packed_block_types, first_parameter):
    """Sparse pair lookup plus compact local paths and ordinary harmonic rows.

    Both orientations share one path/parameter span. A negative span count
    switches the two blocks; it does not reverse or rescale the potential.
    No table proportional to the square of the block-type count is allocated.
    """
    by_name = {
        bt.name: (index, bt)
        for index, bt in enumerate(packed_block_types.active_block_types)
    }
    pairs = {}
    for record in records:
        if record.block_type1 not in by_name or record.block_type2 not in by_name:
            continue
        index1, first = by_name[record.block_type1]
        index2, second = by_name[record.block_type2]
        rows = _validated_rows(record, first, second)
        conn1 = next(
            i for i, c in enumerate(first.connections) if c.name == record.connection1
        )
        conn2 = next(
            i for i, c in enumerate(second.connections) if c.name == record.connection2
        )
        key = (index1, conn1, index2, conn2)
        reverse = (index2, conn2, index1, conn1)
        flipped = {
            _canonical_path([-a for a in path]): value for path, value in rows.items()
        }
        if key == reverse and rows != flipped:
            raise ValueError(
                "Identical connection types require exchange-symmetric parameters"
            )
        if key > reverse:
            key, rows = reverse, flipped
        if key in pairs and pairs[key] != rows:
            raise ValueError("Conflicting parameters for the same connection pair")
        pairs[key] = rows

    n_entries = sum(1 if key[:2] == key[2:] else 2 for key in pairs)
    # Flat allocation followed by reshape gives valid inner strides even
    # for empty tables; native vector-valued tensor views require them.
    hash_keys = numpy.full(10 * n_entries, -1, dtype=numpy.int32).reshape(-1, 5)
    spans = numpy.zeros(2 * n_entries, dtype=numpy.int32).reshape(-1, 2)
    paths, values = [], []
    entry = 0
    for key, rows in sorted(pairs.items()):
        start = len(paths)
        for path, value in sorted(rows.items()):
            paths.append(
                (*path, *((0,) * (4 - len(path))), first_parameter + len(values))
            )
            values.append(value)
        add_to_hashtable(hash_keys, spans, entry, key, (start, len(rows)))
        entry += 1
        reverse = (*key[2:], *key[:2])
        if reverse != key:
            add_to_hashtable(hash_keys, spans, entry, reverse, (start, -len(rows)))
            entry += 1
    return (
        hash_keys,
        spans,
        numpy.asarray(paths, dtype=numpy.int32).reshape(-1, 5),
        numpy.asarray(values, dtype=numpy.float32).reshape(-1, 7),
    )
