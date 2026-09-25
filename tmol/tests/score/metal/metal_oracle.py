"""Plain-torch reference for the metal_coordination kernel."""

from itertools import combinations

import numpy
import torch

from tmol.database.chemical import ideal_distances, metal_table


def restraints(param_db, pose_stack):
    """Site and fan rows for every metal in the stack.

    Site rows are (pose, metal block, metal atom, virtual atom or -1, donor
    block, donor atom) with (d0, well depth, radial sd, lateral sd); fan rows
    are (pose, block, atom, atom) with (l0, sd).
    """
    params = param_db.scoring.metal_coordination
    table = metal_table()
    ion_for_atom_type = {ion["atom_type"]: ion for ion in table["ions"]}
    donor_radii = table["donor_radii"]
    vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
    atom_types = {at.name: at for at in param_db.chemical.atom_types}
    pbt = pose_stack.packed_block_types
    atom_types.update({at.name: at for at in pbt.chem_db.atom_types})

    def donor_key(atom_type_name):
        at = atom_types[atom_type_name]
        return at.name if at.name in donor_radii else at.element

    bt_inds = pose_stack.block_type_ind64.cpu().numpy()
    irc = pose_stack.inter_residue_connections64.cpu().numpy()
    site_rows, site_params, fan_rows, fan_params = [], [], [], []
    for pose, block in zip(*numpy.nonzero(bt_inds >= 0)):
        bt = pbt.active_block_types[bt_inds[pose, block]]
        for site in bt.metal_sites:
            metal = bt.atom_to_idx[site.metal_atom]
            metal_type = bt.atoms[metal].atom_type
            dist = ideal_distances(ion_for_atom_type[metal_type], donor_radii)
            radial_sd, lateral_sd = params.widths(metal_type)
            virts = [bt.atom_to_idx[v] for v in site.site_virts]

            for k, name in enumerate(site.site_connections):
                partner, partner_conn = irc[pose, block, bt.connection_to_cidx[name]]
                if partner < 0:
                    continue
                other = pbt.active_block_types[bt_inds[pose, partner]]
                donor = other.atom_to_idx[other.connections[partner_conn].atom]
                key = donor_key(other.atoms[donor].atom_type)
                site_rows.append(
                    (pose, block, metal, virts[k] if virts else -1, partner, donor)
                )
                site_params.append(
                    (
                        dist[key],
                        params.well_depth(metal_type, key),
                        radial_sd,
                        lateral_sd,
                    )
                )

            if not virts:
                continue
            if site.internal_satisfiers:
                # each free-site virtual against the metal, its satisfiers and
                #    the virtuals before it, at the residue's ideal geometry
                coords = bt.compute_ideal_coords()  # in icoor order
                anchors = [
                    metal,
                    *(bt.atom_to_idx[a] for a in site.internal_satisfiers),
                ]
                pairs = [
                    (v, other)
                    for n, v in enumerate(virts)
                    for other in anchors + virts[:n]
                ]
                ideal = {
                    a: coords[bt.icoors_index[bt.atoms[a].name]]
                    for pair in pairs
                    for a in pair
                }
            else:
                d = {ic.name: ic.d for ic in bt.icoors}
                verts = numpy.asarray(vertices_for[site.geometry], dtype=numpy.float64)
                verts /= numpy.linalg.norm(verts, axis=1, keepdims=True)
                ideal = {metal: numpy.zeros(3)}
                for vertex, name in zip(verts, site.site_virts):
                    ideal[bt.atom_to_idx[name]] = vertex * d[name]
                pairs = list(combinations(list(ideal), 2))
            for a, b in pairs:
                fan_rows.append((pose, block, a, b))
                fan_params.append(
                    (
                        numpy.linalg.norm(
                            numpy.asarray(ideal[a]) - numpy.asarray(ideal[b])
                        ),
                        params.global_parameters.fan_sd,
                    )
                )
    return site_rows, site_params, fan_rows, fan_params


def site_energies(metal, donor, virt, has_virt, params):
    """Radial, lateral and well-depth energy per site."""
    d0, depth, radial_sd, lateral_sd = params.unbind(-1)
    delta = donor - metal
    r = torch.linalg.norm(delta, dim=-1)
    radial = ((r - d0) / radial_sd) ** 2
    ray = torch.where(has_virt.unsqueeze(-1), virt - metal, delta)
    u = ray / torch.linalg.norm(ray, dim=-1, keepdim=True)
    off_ray = delta - (delta * u).sum(-1, keepdim=True) * u
    lateral = (off_ray * off_ray).sum(-1) / lateral_sd**2
    return radial + torch.where(has_virt, lateral, torch.zeros_like(lateral)) + depth


def fan_energies(a, b, params):
    """Harmonic on each metal-virtual and virtual-virtual separation."""
    l0, sd = params.unbind(-1)
    return ((torch.linalg.norm(a - b, dim=-1) - l0) / sd) ** 2


def block_pair_energies(param_db, pose_stack, coords):
    """[n_poses, n_blocks, n_blocks] energies, upper triangle."""
    site_rows, site_params, fan_rows, fan_params = restraints(param_db, pose_stack)
    n_poses, max_n_blocks = pose_stack.block_type_ind64.shape
    # a structure whose sites are all filled internally carries no restraint at
    #    all, and the zero it scores still has to reach the caller's backward
    out = (
        torch.zeros(
            (n_poses, max_n_blocks, max_n_blocks),
            dtype=coords.dtype,
            device=coords.device,
        )
        + 0.0 * coords.sum()
    )
    offsets = pose_stack.block_coord_offset64.to(coords.device)

    def atom(pose, block, at):
        return coords[pose, offsets[pose, block] + at]

    def rows(values):
        return torch.tensor(values, dtype=torch.int64, device=coords.device)

    def params(values):
        # parameters are stored in single precision, as the kernel reads them
        return torch.tensor(values, dtype=torch.float32).to(coords)

    if site_rows:
        pose, mblock, matom, vatom, dblock, datom = rows(site_rows).unbind(1)
        e = site_energies(
            atom(pose, mblock, matom),
            atom(pose, dblock, datom),
            atom(pose, mblock, vatom.clamp_min(0)),
            vatom >= 0,
            params(site_params),
        )
        lo, hi = torch.minimum(mblock, dblock), torch.maximum(mblock, dblock)
        out = out.index_put((pose, lo, hi), e, accumulate=True)
    if fan_rows:
        pose, block, a, b = rows(fan_rows).unbind(1)
        e = fan_energies(
            atom(pose, block, a),
            atom(pose, block, b),
            params(fan_params),
        )
        out = out.index_put((pose, block, block), e, accumulate=True)
    return out
