"""Complete anchored non-polymers from their prepared conformer geometry."""

import math

import torch


def _frame(points):
    """A proper rotation with its z axis along the first two points."""
    z = torch.nn.functional.normalize(points[1] - points[0], dim=-1)
    y = points[2] - points[0]
    y = torch.nn.functional.normalize(y - (y * z).sum() * z, dim=-1)
    return torch.stack((torch.linalg.cross(y, z), y, z), dim=-1)


def _anchor_triangle(reference, observed):
    """Choose a well-separated axis and plane in linear work; reject collinearity."""
    vectors = [(points - points[0]).detach() for points in (reference, observed)]
    lengths = [torch.linalg.vector_norm(v, dim=-1) for v in vectors]
    second = int((lengths[0] * lengths[1]).argmax())
    if min(float(length[second]) for length in lengths) < 1e-6:
        return None
    sines = [
        torch.linalg.vector_norm(torch.linalg.cross(v[second].expand_as(v), v), dim=-1)
        / (length[second] * length).clamp_min(1e-12)
        for v, length in zip(vectors, lengths)
    ]
    quality = torch.minimum(*sines)
    third = int(quality.argmax())
    if float(quality[third]) < 1e-4:
        return None
    return [0, second, third]


def _sampled_attachment_frame(bt, names, ideal, observed, downstream):
    """Use an explicitly sampled linkage when only its two ends are resolved.

    The local ring neighbor, attachment atom, partner and partner's declared
    downstream atom must form the sampled torsion. No laboratory axis or
    unspecified torsion supplies an orientation.
    """
    samples = {cs.chi_dihedral: cs.samples for cs in bt.chi_samples if cs.samples}
    for torsion in bt.torsions:
        if (
            torsion.name not in samples
            or torsion.a.atom not in bt.atom_to_idx
            or torsion.b.atom != names[0]
            or torsion.c.connection != names[1]
            or torsion.c.bond_sep_from_conn != 0
            or torsion.d.connection != names[1]
            or torsion.d.bond_sep_from_conn != 1
        ):
            continue
        third = downstream(bt.connection_to_cidx[names[1]], 1)
        if third is None:
            continue
        reference = ideal[[bt.icoors_index[name] for name in (*names, torsion.a.atom)]]
        actual = torch.stack((*observed, third)).to(dtype=ideal.dtype)
        if _anchor_triangle(reference, actual) is None:
            continue
        phi = math.radians(samples[torsion.name][0])
        c, s = math.cos(phi), math.sin(phi)
        rotation = ideal.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return reference, actual, rotation
    return None


def build_missing_nonpolymer_atoms(pbt, coords, targets, offsets, types, connections):
    """Finish blocks whose ordinary construction frames made no progress.

    Apply a proper rigid transform of the prepared ideal conformer to missing
    atoms only. Resolved heavy atoms and resolved connection partners supply
    the frame. With just one local anchor and its partner, an explicitly
    sampled attachment torsion can define the plane using the next partner
    atom. Its first declared sample seeds construction; packing can sample the
    other states. Fewer anchors, absent samples and degenerate frames stay NaN.

    Supplied coordinates remain exact and generated coordinates differentiate
    through the selected anchors. This is conformer initialization, not a
    ring-closure fit to mutually inconsistent supplied coordinates.
    """
    updates, indices = [], []
    for pi, bi in torch.nonzero(targets.any(-1)).tolist():
        ti = int(types[pi, bi])
        bt = pbt.active_block_types[ti]
        if bt.properties.polymer.is_polymer:
            continue
        offset = int(offsets[pi, bi])
        current = coords[pi, offset : offset + bt.n_atoms]
        finite = torch.isfinite(current).all(-1)
        local = (
            torch.nonzero(finite & ~pbt.atom_is_hydrogen[ti, : bt.n_atoms].bool())
            .flatten()
            .tolist()
        )
        names = [bt.atoms[ai].name for ai in local]
        observed = [current[ai] for ai in local]

        def downstream(ci, separation):
            other, port = connections[pi, bi, ci].tolist()
            if other < 0:
                return None
            oi = int(
                pbt.atom_downstream_of_conn[int(types[pi, other]), port, separation]
            )
            if oi < 0:
                return None
            xyz = coords[pi, int(offsets[pi, other]) + oi]
            return xyz if torch.isfinite(xyz).all() else None

        for ci, connection in enumerate(bt.connections):
            xyz = downstream(ci, 0)
            if xyz is not None:
                names.append(connection.name)
                observed.append(xyz)
        if len(names) < 2:
            continue
        # These small frame products must not use reduced-precision CUDA matmul.
        ideal = coords.new_tensor(bt.ideal_coords, dtype=torch.float64)
        if not torch.isfinite(ideal).all():
            continue
        rotation = None
        if len(names) == 2:
            if len(local) != 1:
                continue
            sampled = _sampled_attachment_frame(bt, names, ideal, observed, downstream)
            if sampled is None:
                continue
            reference, actual, rotation = sampled
        else:
            reference = ideal[[bt.icoors_index[name] for name in names]]
            actual = torch.stack(observed).to(dtype=ideal.dtype)
            triangle = _anchor_triangle(reference, actual)
            if triangle is None:
                continue
            reference, actual = reference[triangle], actual[triangle]
        transform = _frame(reference)
        if rotation is not None:
            transform = transform @ rotation.T
        transform = transform @ _frame(actual).T
        missing = torch.nonzero(
            targets[pi, bi, : bt.n_atoms] & torch.isnan(current).any(-1)
        ).flatten()
        generated = (ideal[bt.at_to_icoor_ind] - reference[0]) @ transform + actual[0]
        updates.append(generated[missing].to(dtype=coords.dtype))
        indices.append(torch.stack((torch.full_like(missing, pi), offset + missing)))
    if not updates:
        return coords
    # One functional scatter preserves autograd through all input views. A block
    # completed here can anchor another block on the next construction pass.
    return coords.index_put(tuple(torch.cat(indices, dim=1)), torch.cat(updates))
