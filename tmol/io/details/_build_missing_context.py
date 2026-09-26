"""Build missing heavy atoms outward from observed ones, choosing torsions that do not clash.

Each block's missing atoms are placed along a tree grown from its observed atoms (and
its connection partners): every atom takes its distance, angle and dihedral from the
block type's ideal coordinates against three already-placed references, so bonded
geometry, rings and chirality are ideal. A rotation left free by the observed atoms
-- about a rotatable bond whose far side is entirely missing, or the spin of a
fragment anchored by fewer than three atoms -- is chosen greedily in tree order: each
candidate is scored by building everything still missing, later rotations at their
ideal values, against a heavy-atom clash penalty. Observed atoms that only the
missing atoms would tell apart (a phosphate's oxygens) may trade labels, where that
lowers the penalty.
"""

import itertools
import math
from collections import deque
from typing import Dict, List, Tuple

import numpy
import torch
from scipy.spatial import cKDTree

from tmol.chemical._restypes import BondType

# a heavy-atom pair closer than this clashes; donor-acceptor pairs may come closer
CONTACT = 3.0
POLAR_CONTACT = 2.6
# pairs within this many bonds are not scored
EXCLUDED_BONDS = 3
SP3_STAGGERED = (60.0, 180.0, 300.0)
SP3_EXPANSION = 20.0
SP2_PLANAR = (0.0, 180.0)
# orientations tried where no bond fixes a fragment's spin
FREE_SPINS = tuple(float(x) for x in range(0, 360, 30))
UNSATURATED = (int(BondType.DOUBLE), int(BondType.TRIPLE), int(BondType.AROMATIC))


def _unit(v):
    n = numpy.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _angle(a, b, c):
    u, v = _unit(a - b), _unit(c - b)
    return math.acos(max(-1.0, min(1.0, float(u @ v))))


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, _unit(c - b), d - c
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return math.atan2(float(numpy.cross(b1, v) @ w), float(v @ w))


def _cross(u, v):
    return numpy.stack(
        (
            u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
            u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
            u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0],
        ),
        axis=-1,
    )


def _normalize(v):
    return v / numpy.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12, None)


def _place_batch(a, b, c, dist, theta, phi):
    """_place over candidates: references [..., 3] (broadcast), phi [C]."""
    bc = _normalize(c - b)
    n = _normalize(_cross(b - a, bc))
    m = _cross(n, bc)
    phi = numpy.asarray(phi)[..., None]
    return c + dist * (
        -math.cos(theta) * bc
        + math.sin(theta) * numpy.cos(phi) * m
        + math.sin(theta) * numpy.sin(phi) * n
    )


def _place_torch(a, b, c, dist, theta, phi):
    bc = torch.nn.functional.normalize(c - b, dim=-1)
    n = torch.nn.functional.normalize(torch.linalg.cross(b - a, bc), dim=-1)
    m = torch.linalg.cross(n, bc)
    return c + dist * (
        -math.cos(theta) * bc
        + math.sin(theta) * math.cos(phi) * m
        + math.sin(theta) * math.sin(phi) * n
    )


def _perpendicular_point(origin, through):
    """A point off the origin-through axis, fixed relative to it."""
    axis = _unit(through - origin)
    other = numpy.eye(3)[int(numpy.argmin(numpy.abs(axis)))]
    return origin + numpy.cross(axis, other)


class _BlockTypeGeometry:
    """What the builder reads from a block type, computed once."""

    def __init__(self, bt, atom_types):
        n = bt.n_atoms
        self.n = n
        types = [atom_types[a.atom_type] for a in bt.atoms]
        self.element = [t.element for t in types]
        self.heavy = numpy.array([t.element != "H" for t in types])
        self.donor = numpy.array([bool(t.is_donor) for t in types])
        self.acceptor = numpy.array([bool(t.is_acceptor) for t in types])
        self.metal = numpy.array([bool(t.is_metal) for t in types])
        self.metal_donor = numpy.array([bool(t.is_metal_donor) for t in types])
        ideal = bt.ideal_coords  # in icoor order
        self.ideal = numpy.array(
            [ideal[bt.icoors_index[a.name]] for a in bt.atoms], dtype=numpy.float64
        )
        self.neighbors: List[List[int]] = [[] for _ in range(n)]
        for i, j in bt.bond_indices.tolist():
            if i != j and j not in self.neighbors[i]:
                self.neighbors[i].append(j)
        self.bond_type = dict(bt.bond_to_type)
        self.in_ring = dict(bt.bond_to_ringness)
        self.unsaturated = numpy.array(
            [
                any(
                    self.bond_type.get((i, j)) in UNSATURATED for j in self.neighbors[i]
                )
                for i in range(n)
            ]
        )
        # graph distance, capped past the scoring exclusion
        cap = EXCLUDED_BONDS + 1
        self.bonds_apart = numpy.full((n, n), cap, dtype=numpy.int8)
        for s in range(n):
            self.bonds_apart[s, s] = 0
            frontier, depth = [s], 0
            while frontier and depth < EXCLUDED_BONDS:
                depth += 1
                nxt = []
                for u in frontier:
                    for v in self.neighbors[u]:
                        if self.bonds_apart[s, v] > depth:
                            self.bonds_apart[s, v] = depth
                            nxt.append(v)
                frontier = nxt
        # connection partner positions in the ideal frame, where the icoors give them
        self.conn_atom = [bt.atom_to_idx[c.atom] for c in bt.connections]
        self.conn_ideal = [
            (
                numpy.asarray(ideal[bt.icoors_index[c.name]], dtype=numpy.float64)
                if c.name in bt.icoors_index
                else None
            )
            for c in bt.connections
        ]
        self.conn_rotatable = [
            int(bt.connection_bond_types[k]) == int(BondType.SINGLE)
            and not bool(bt.connection_bond_in_ring[k])
            for k in range(len(bt.connections))
        ]
        # chi samples by rotation axis: (g-side atom, g, p, p-side atom) -> samples
        samples = {cs.chi_dihedral: cs for cs in bt.chi_samples if cs.samples}
        self.chi_for_axis: Dict[Tuple[int, int], Tuple[int, int, List[float]]] = {}
        for torsion in bt.torsions:
            cs = samples.get(torsion.name)
            names = [torsion.a.atom, torsion.b.atom, torsion.c.atom, torsion.d.atom]
            if cs is None or any(x not in bt.atom_to_idx for x in names):
                continue
            a, b, c, d = (bt.atom_to_idx[x] for x in names)
            values = [
                float(s + e * sign)
                for s in cs.samples
                for e in (0.0, *cs.expansions)
                for sign in ((1.0,) if e == 0.0 else (1.0, -1.0))
            ]
            # oriented both ways: dihedral(a,b,c,d) == dihedral(d,c,b,a)
            self.chi_for_axis[(b, c)] = (a, d, values)
            self.chi_for_axis[(c, b)] = (d, a, values)


def _geometry(bt, atom_types):
    geom = getattr(bt, "_missing_context_geometry", None)
    if geom is None:
        geom = _BlockTypeGeometry(bt, atom_types)
        setattr(bt, "_missing_context_geometry", geom)
    return geom


class _Step:
    """How one missing atom is placed: references, ideal geometry, its rotation."""

    __slots__ = ("atom", "refs", "dist", "theta", "phi", "axis")

    def __init__(self, atom, refs, dist, theta, phi, axis):
        self.atom, self.refs = atom, refs
        self.dist, self.theta, self.phi = dist, theta, phi
        self.axis = axis


def _plan(geom, known, targets, conn_nodes, ideal_of):
    """Order the missing atoms and pick each one's three placed references.

    known: nodes already placed (local atoms, and n + k for connection k).
    Returns the steps in build order and, per rotation axis, its candidates in
    radians as offsets from the ideal dihedral of the axis's first step.
    """
    n = geom.n
    neighbors = {i: list(geom.neighbors[i]) for i in range(n)}
    for k, (atom, node) in conn_nodes.items():
        neighbors[atom].append(node)
        neighbors[node] = [atom]
    placed = set(known)
    parent: Dict[int, int] = {}
    order: List[int] = []
    queue = deque(sorted(placed))
    while queue:
        u = queue.popleft()
        for v in neighbors.get(u, ()):
            if v in targets and v not in placed:
                placed.add(v)
                parent[v] = u
                order.append(v)
                queue.append(v)

    steps: List[_Step] = []
    axis_candidates: Dict[tuple, List[float]] = {}
    ready = set(known)
    for a in order:
        p = parent[a]
        g_options = [x for x in neighbors[p] if x in ready and x != a]
        if p in parent and parent[p] in g_options:
            g_options.remove(parent[p])
            g_options.insert(0, parent[p])
        axis = None
        if len(g_options) >= 2:
            g, gg = g_options[0], g_options[1]
        elif len(g_options) == 1:
            g = g_options[0]
            gg_options = [x for x in neighbors[g] if x in ready and x not in (p, a)]
            if g in parent and parent[g] in gg_options:
                gg_options.remove(parent[g])
                gg_options.insert(0, parent[g])
            if gg_options:
                gg = gg_options[0]
                if _rotatable(geom, g, p, n):
                    axis = (g, p)
            else:
                gg = ("perp", g, p)
                axis = (g, p)
        else:
            # a lone anchor: the first bond's direction is free
            g = ("perp", p, None)
            gg = ("perp2", p, None)
            axis = ("free", p)
        refs = (gg, g, p)
        ia = ideal_of(a)
        ip, ig, igg = ideal_of(p), _ideal_ref(ideal_of, g), _ideal_ref(ideal_of, gg)
        dist = float(numpy.linalg.norm(ia - ip))
        theta = _angle(ig, ip, ia)
        phi = _dihedral(igg, ig, ip, ia)
        steps.append(_Step(a, refs, dist, theta, phi, axis))
        if axis is not None and axis not in axis_candidates:
            axis_candidates[axis] = _candidates(geom, axis, phi, ideal_of, n)
        ready.add(a)
    return steps, axis_candidates


def _ideal_ref(ideal_of, ref):
    if isinstance(ref, tuple):
        kind, origin, through = ref
        o = ideal_of(origin)
        if kind == "perp":
            if through is None:
                return o + numpy.array([1.0, 0.0, 0.0])
            return _perpendicular_point(o, ideal_of(through))
        return o + numpy.array([0.0, 1.0, 0.0])
    return ideal_of(ref)


def _rotatable(geom, g, p, n):
    if g >= n:  # the bond to a connection partner
        return geom.conn_rotatable[g - n]
    if p >= n:
        return False
    return geom.bond_type.get((g, p)) == int(BondType.SINGLE) and not geom.in_ring.get(
        (g, p), False
    )


def _candidates(geom, axis, phi0, ideal_of, n):
    """Offsets from the axis's ideal dihedral, radians."""
    if axis[0] == "free" or not isinstance(axis[0], int) or not _is_bond_axis(axis, n):
        return [0.0] + [math.radians(x) for x in FREE_SPINS[1:]]
    g, p = axis
    offsets = [0.0]
    chi = geom.chi_for_axis.get((g, p)) if g < n and p < n else None
    if chi is not None:
        g_side, p_side, values = chi
        chi0 = _dihedral(ideal_of(g_side), ideal_of(g), ideal_of(p), ideal_of(p_side))
        offsets += [math.radians(v) - chi0 for v in values]
    else:
        planar = (g < n and geom.unsaturated[g]) or (p < n and geom.unsaturated[p])
        if planar:
            absolute = SP2_PLANAR
        else:
            absolute = [
                s + e
                for s in SP3_STAGGERED
                for e in (0.0, SP3_EXPANSION, -SP3_EXPANSION)
            ]
        offsets += [math.radians(v) - phi0 for v in absolute]
    return offsets


def _is_bond_axis(axis, n):
    return all(isinstance(x, int) for x in axis)


class _Environment:
    """Heavy atoms a block's rebuilt atoms may clash with, outside that block."""

    def __init__(self, xyz, block_of, donor, acceptor, metal, metal_donor, usable):
        self.xyz, self.block_of = xyz, block_of
        self.donor, self.acceptor = donor, acceptor
        self.metal, self.metal_donor = metal, metal_donor
        idx = numpy.flatnonzero(usable)
        self.idx = idx
        self.tree = cKDTree(xyz[idx]) if len(idx) else None
        self.extra: List[int] = []  # atoms built since the tree was made
        self.extra_tree = None

    def add(self, atoms):
        self.extra.extend(atoms)
        self.extra_tree = None

    def moved(self):
        """The extra atoms' coordinates changed."""
        self.extra_tree = None

    def pairs(self, points, excluded):
        """(point index, environment atom, distance) for every pair within CONTACT.

        points: a cKDTree of the query points.
        """
        rows, cols, dist = [], [], []
        if self.tree is not None:
            found = points.sparse_distance_matrix(
                self.tree, CONTACT, output_type="ndarray"
            )
            rows.append(found["i"])
            cols.append(self.idx[found["j"]])
            dist.append(found["v"])
        if self.extra:
            extra = numpy.asarray(self.extra)
            if self.extra_tree is None:
                self.extra_tree = cKDTree(self.xyz[extra])
            found = points.sparse_distance_matrix(
                self.extra_tree, CONTACT, output_type="ndarray"
            )
            rows.append(found["i"])
            cols.append(extra[found["j"]])
            dist.append(found["v"])
        if not rows:
            empty = numpy.zeros(0, dtype=int)
            return empty, empty, numpy.zeros(0)
        rows, cols, dist = (numpy.concatenate(x) for x in (rows, cols, dist))
        if len(cols) and len(excluded):
            keep = ~numpy.isin(cols, excluded)
            rows, cols, dist = rows[keep], cols[keep], dist[keep]
        return rows.astype(int), cols.astype(int), dist


def _overlap(d, a, b, atoms_a, atoms_b):
    """Pair overlaps; a and b hold donor, acceptor, metal and metal_donor flags.

    A metal is not scored against a metal or an atom that can coordinate it.
    """
    polar = (a.donor[atoms_a] & b.acceptor[atoms_b]) | (
        a.acceptor[atoms_a] & b.donor[atoms_b]
    )
    ma, mb = a.metal[atoms_a], b.metal[atoms_b]
    bound = (ma & (mb | b.metal_donor[atoms_b])) | (mb & a.metal_donor[atoms_a])
    r0 = numpy.where(polar, POLAR_CONTACT, CONTACT)
    return numpy.where(bound, 0.0, numpy.clip(r0 - d, 0.0, None) ** 2)


def _clash(moved, points, others, other_xyz, geom, env, excluded):
    """Per-candidate overlap of the moved atoms with the environment and the block.

    moved: local atoms; points: [C, m, 3] their positions for each candidate;
    others: local atoms at other_xyz, placed and not moved. Pairs within
    EXCLUDED_BONDS bonds, and hydrogens, are skipped.
    """
    n_cand = points.shape[0]
    moved = numpy.asarray(moved)
    scorable = geom.heavy[moved]
    moved, points = moved[scorable], points[:, scorable]
    score = numpy.zeros(n_cand)
    m = len(moved)
    if not m:
        return score
    flat = cKDTree(points.reshape(-1, 3))
    rows, cols, d = env.pairs(flat, excluded)
    if len(rows):
        atom = moved[rows % m]
        overlap = _overlap(d, geom, env, atom, cols)
        score += numpy.bincount(rows // m, weights=overlap, minlength=n_cand)

    # the moved atoms turn rigidly, so their distances to each other do not change
    others = numpy.asarray(others, dtype=int)
    if len(others):
        keep = geom.heavy[others]
        others, other_xyz = others[keep], other_xyz[keep]
    if len(others):
        found = flat.sparse_distance_matrix(
            cKDTree(other_xyz), CONTACT, output_type="ndarray"
        )
        a = moved[found["i"] % m]
        b = others[found["j"]]
        far = geom.bonds_apart[a, b] > EXCLUDED_BONDS
        if far.any():
            a, b = a[far], b[far]
            overlap = _overlap(found["v"][far], geom, geom, a, b)
            score += numpy.bincount(
                found["i"][far] // m, weights=overlap, minlength=n_cand
            )
    return score


def build_missing_by_context(
    pbt,
    pose_coords,
    missing,
    targets,
    offsets,
    block_types,
    connections,
    heavy_only=True,
):
    """Place each block's missing target atoms (only heavy ones, by default).

    pose_coords: [n_poses, n_atoms, 3]; missing and targets: [n_poses, n_atoms], the
    atoms without coordinates and those of them to build. Blocks grow together,
    each rotation seeing every block's current build. Returns the new coordinates and the mask
    of atoms built; a target no placed reference reaches is left unbuilt.
    """
    atom_types = {at.name: at for at in pbt.chem_db.atom_types}
    out = pose_coords.clone()
    xyz_all = pose_coords.detach().cpu().double().numpy().copy()
    xyz_all[missing.cpu().numpy()] = numpy.nan
    tgt_all = targets.cpu().numpy()
    built_mask = torch.zeros_like(targets)
    off_all = offsets.cpu().numpy()
    types_all = block_types.cpu().numpy()
    conn_all = connections.cpu().numpy()
    conn_atom = pbt.conn_atom.cpu().numpy()
    bts = pbt.active_block_types
    for pose in range(xyz_all.shape[0]):
        if not tgt_all[pose].any():
            continue
        xyz = xyz_all[pose]
        n_pose_atoms = xyz.shape[0]
        block_of = numpy.full(n_pose_atoms, -1, dtype=numpy.int64)
        donor = numpy.zeros(n_pose_atoms, dtype=bool)
        acceptor = numpy.zeros(n_pose_atoms, dtype=bool)
        metal = numpy.zeros(n_pose_atoms, dtype=bool)
        metal_donor = numpy.zeros(n_pose_atoms, dtype=bool)
        heavy = numpy.zeros(n_pose_atoms, dtype=bool)
        geoms = {}
        for b, t in enumerate(types_all[pose].tolist()):
            if t < 0:
                continue
            geom = geoms[b] = _geometry(bts[t], atom_types)
            s = int(off_all[pose, b])
            block_of[s : s + geom.n] = b
            donor[s : s + geom.n] = geom.donor
            acceptor[s : s + geom.n] = geom.acceptor
            metal[s : s + geom.n] = geom.metal
            metal_donor[s : s + geom.n] = geom.metal_donor
            heavy[s : s + geom.n] = geom.heavy
        finite = numpy.isfinite(xyz).all(axis=-1)
        env = _Environment(
            xyz, block_of, donor, acceptor, metal, metal_donor, heavy & finite
        )
        pending = {}

        for b, t in enumerate(types_all[pose].tolist()):
            if t < 0:
                continue
            geom = geoms[b]
            s = int(off_all[pose, b])
            local_targets = set(
                numpy.flatnonzero(tgt_all[pose, s : s + geom.n]).tolist()
            )
            if heavy_only:
                local_targets = {a for a in local_targets if geom.heavy[a]}
            if not local_targets:
                continue
            pending[b] = local_targets

        while pending:
            # a block reaching another block's missing atom waits for that one
            wave = {
                b: t
                for b, t in pending.items()
                if not _waits_on(
                    pose, b, geoms[b], pending, conn_all, conn_atom, types_all
                )
            } or dict(pending)
            for b in wave:
                del pending[b]
            builds = []
            for b, local_targets in wave.items():
                build = _BlockBuild(
                    pose,
                    b,
                    int(off_all[pose, b]),
                    geoms[b],
                    local_targets,
                    xyz,
                    conn_all,
                    conn_atom,
                    off_all,
                    types_all,
                    geoms,
                )
                if build.steps:
                    builds.append(build)
            _build_together(builds, xyz, out, env)
            for build in builds:
                built_mask[pose, [build.s + a for a in build.built]] = True
    return out, built_mask


def _waits_on(pose, b, geom, pending, conn_all, conn_atom, types_all):
    """Whether a connection partner's atom is a target of another pending block."""
    for k in range(len(geom.conn_atom)):
        partner, pconn = conn_all[pose, b, k]
        if partner < 0 or partner == b or int(partner) not in pending:
            continue
        patom = int(conn_atom[types_all[pose, partner], pconn])
        if patom in pending[int(partner)]:
            return True
    return False


def _build_together(builds, xyz, out, env):
    """Grow blocks at once: each rotation, shallowest first, sees every block's build.

    Each block starts at its ideal build; interchangeable anchors are settled per
    block against the others' ideal builds.
    """
    for build in builds:
        build.start(xyz)
        build.write(xyz, env)
    env.add([build.s + a for build in builds for a in build.built])
    for build in builds:
        swaps = _anchor_swaps(
            build.geom, build.known_atoms, build.targets, build.conn_atoms
        )
        if not swaps:
            continue
        score = build.greedy(xyz, env)
        for options in swaps:
            base = dict(build.relabel)
            for option in options[1:]:
                build.relabel = {**base, **option}
                trial = build.greedy(xyz, env)
                if trial < score - 1e-9:
                    score, base = trial, dict(build.relabel)
            build.relabel = base
        build.start(xyz)
        build.write(xyz, env)
    order = sorted(
        (
            (build.axis_depth[axis], k, i, axis)
            for k, build in enumerate(builds)
            for i, axis in enumerate(build.axes)
        ),
        key=lambda x: x[:3],
    )
    for _, k, _, axis in order:
        builds[k].choose(axis, env)
        builds[k].write(xyz, env)
    for build in builds:
        build.place(xyz, out)
    env.moved()


def _connection_nodes(
    pose, b, n, geom, xyz, conn_all, conn_atom, off_all, types_all, geoms
):
    """Placed connection partners as extra nodes n + k, and atoms to leave unscored.

    Returns {k: (local atom, node)}, node positions, node ideal positions, node pose
    indices, and the pose atoms not to score against (this block, and the partners'
    atoms within reach of the bond).
    """
    nodes, node_xyz, node_ideal, node_global = {}, {}, {}, {}
    excluded = set()
    for k, atom in enumerate(geom.conn_atom):
        partner, pconn = conn_all[pose, b, k]
        if partner < 0:
            continue
        patom = int(conn_atom[types_all[pose, partner], pconn])
        pstart = int(off_all[pose, partner])
        pgeom = geoms[int(partner)]
        for x in range(pgeom.n):
            if pgeom.bonds_apart[patom, x] <= EXCLUDED_BONDS - 1:
                excluded.add(pstart + x)
        pxyz = xyz[pstart + patom]
        if geom.conn_ideal[k] is None or not numpy.isfinite(pxyz).all():
            continue
        nodes[k] = (atom, n + k)
        node_xyz[n + k] = pxyz
        node_ideal[n + k] = geom.conn_ideal[k]
        node_global[n + k] = pstart + patom
    return nodes, node_xyz, node_ideal, node_global, excluded


def _resolve(ref, positions, offset_point):
    """A reference's position: a node, or a point built off one to fix a free spin."""
    if not isinstance(ref, tuple):
        return positions[ref]
    kind, origin, through = ref
    o = positions[origin]
    if kind == "perp" and through is not None:
        return offset_point(o, positions[through])
    return o + (
        numpy.array([1.0, 0.0, 0.0]) if kind == "perp" else numpy.array([0.0, 1.0, 0.0])
    )


class _BlockBuild:
    """One block's plan and its build while the rotations are chosen."""

    def __init__(
        self,
        pose,
        b,
        s,
        geom,
        targets,
        xyz,
        conn_all,
        conn_atom,
        off_all,
        types_all,
        geoms,
    ):
        self.pose, self.b, self.s, self.geom, self.targets = pose, b, s, geom, targets
        n = geom.n
        local_finite = numpy.isfinite(xyz[s : s + n]).all(axis=-1)
        self.known_atoms = [a for a in range(n) if local_finite[a] and a not in targets]
        nodes, self.node_xyz, node_ideal, self.node_global, excluded = (
            _connection_nodes(
                pose, b, n, geom, xyz, conn_all, conn_atom, off_all, types_all, geoms
            )
        )
        self.excluded = numpy.array(sorted(excluded | set(range(s, s + n))), dtype=int)
        self.conn_atoms = {atom for atom, _ in nodes.values()}

        def ideal_of(x):
            return node_ideal[x] if x >= n else geom.ideal[x]

        known = set(self.known_atoms) | set(self.node_xyz)
        self.steps, self.axis_candidates = _plan(geom, known, targets, nodes, ideal_of)
        self.built = [st.atom for st in self.steps]
        index_of = {st.atom: i for i, st in enumerate(self.steps)}
        depends = []
        for st in self.steps:
            axes = {st.axis} if st.axis is not None else set()
            for ref in st.refs:
                node = ref[1] if isinstance(ref, tuple) else ref
                if node in index_of:
                    axes |= depends[index_of[node]]
                if isinstance(ref, tuple) and ref[2] in index_of:
                    axes |= depends[index_of[ref[2]]]
            depends.append(axes)
        self.moved_by = {}
        for i, axes in enumerate(depends):
            for axis in axes:
                self.moved_by.setdefault(axis, []).append(i)
        # bonds from the placed atoms, to grow every block outward in step
        depth = {}
        for st in self.steps:
            depth[st.atom] = depth.get(st.refs[2], 0) + 1
        self.axes = sorted(self.moved_by, key=lambda axis: self.moved_by[axis][0])
        self.axis_depth = {
            axis: depth[self.steps[self.moved_by[axis][0]].atom] for axis in self.axes
        }
        self.relabel = {}

    def start(self, xyz):
        """Every step at its ideal, from the observed atoms as relabeled."""
        positions = {a: xyz[self.s + self.relabel.get(a, a)] for a in self.known_atoms}
        positions.update(self.node_xyz)
        for step in self.steps:
            gg, g, p = (_resolve(r, positions, _perpendicular_point) for r in step.refs)
            positions[step.atom] = _place_batch(
                gg, g, p, step.dist, step.theta, numpy.array([step.phi])
            )[0]
        self.positions, self.chosen = positions, {}

    def write(self, xyz, env):
        for a in self.built:
            xyz[self.s + a] = self.positions[a]
        env.moved()

    def choose(self, axis, env):
        """The lowest-clash candidate for one rotation, ties to the ideal."""
        n, positions = self.geom.n, self.positions
        which = self.moved_by[axis]
        moved = [self.steps[i].atom for i in which]
        moved_set = set(moved)
        others = [x for x in positions if x < n and x not in moved_set]
        other_xyz = numpy.array([positions[x] for x in others]).reshape(-1, 3)
        offsets = numpy.asarray(self.axis_candidates[axis])
        # a rotation turns everything it moves rigidly about its bond
        first = self.steps[which[0]]
        gg, g, p = (_resolve(r, positions, _perpendicular_point) for r in first.refs)
        sign = _rotation_sign(gg, g, p, first, positions[first.atom])
        xyz = numpy.array([positions[a] for a in moved])
        points = _rotate_about(xyz, g, p, sign * offsets)
        scores = _clash(moved, points, others, other_xyz, self.geom, env, self.excluded)
        best = int(numpy.flatnonzero(scores <= scores.min() + 1e-9)[0])
        self.chosen[axis] = float(offsets[best])
        for a, x in zip(moved, points[best]):
            positions[a] = x

    def score(self, env):
        return _total_clash(self.built, self.positions, self.geom, env, self.excluded)

    def greedy(self, xyz, env):
        """Choose this block's rotations alone, in order; returns the clash score."""
        self.start(xyz)
        for axis in self.axes:
            self.choose(axis, env)
        return self.score(env)

    def place(self, xyz, out):
        """Apply the relabeling and place the chosen build differentiably."""
        pose, s = self.pose, self.s
        if self.relabel:
            atoms = list(self.relabel)
            sources = [self.relabel[a] for a in atoms]
            out[pose, [s + a for a in atoms]] = out[
                pose, [s + a for a in sources]
            ].clone()
            xyz[[s + a for a in atoms]] = xyz[[s + a for a in sources]].copy()
            self.relabel = {}
        placed = _place_differentiably(
            self.steps, self.chosen, out, pose, s, self.known_atoms, self.node_global
        )
        for a, x in placed.items():
            xyz[s + a] = x.detach().cpu().double().numpy()


def _total_clash(built, positions, geom, env, excluded):
    """Overlap of the built atoms with everything, each other included."""
    n = geom.n
    built_set = set(built)
    others = [x for x in positions if x < n and x not in built_set]
    other_xyz = numpy.array([positions[x] for x in others]).reshape(-1, 3)
    xyz = numpy.array([positions[a] for a in built])
    score = float(_clash(built, xyz[None], others, other_xyz, geom, env, excluded)[0])
    built = numpy.asarray(built)
    keep = geom.heavy[built]
    built, xyz = built[keep], xyz[keep]
    if len(built) > 1:
        pairs = cKDTree(xyz).query_pairs(CONTACT, output_type="ndarray")
        a, b = built[pairs[:, 0]], built[pairs[:, 1]]
        far = geom.bonds_apart[a, b] > EXCLUDED_BONDS
        if far.any():
            d = numpy.linalg.norm(xyz[pairs[far, 0]] - xyz[pairs[far, 1]], axis=-1)
            score += float(_overlap(d, geom, geom, a[far], b[far]).sum())
    return score


def _anchor_swaps(geom, known_atoms, targets, conn_atoms):
    """Relabelings of observed atoms the observed atoms alone cannot tell apart.

    Observed heavy atoms of one element bonded only to the same observed atom
    (e.g. a phosphate's oxygens) are interchangeable when a missing branch grows
    from one of them. Returns, per such group, its relabelings as
    {atom: atom whose coordinates it takes}, the identity first.
    """
    known = set(known_atoms)
    groups: Dict[tuple, List[int]] = {}
    for a in known_atoms:
        if not geom.heavy[a] or a in conn_atoms:
            continue
        placed = [x for x in geom.neighbors[a] if x in known]
        if len(placed) != 1:
            continue
        groups.setdefault((placed[0], geom.element[a]), []).append(a)
    swaps = []
    for members in groups.values():
        roots = [a for a in members if any(x in targets for x in geom.neighbors[a])]
        if len(members) < 2 or not roots:
            continue
        rest = [a for a in members if a not in roots]
        options = [{}]
        for sources in itertools.permutations(members, len(roots)):
            remaining = [a for a in members if a not in sources]
            relabel = dict(zip(roots + rest, list(sources) + remaining))
            relabel = {a: x for a, x in relabel.items() if a != x}
            if relabel not in options:
                options.append(relabel)
        swaps.append(options)
    return swaps


def _rotate_about(xyz, origin, through, angles):
    """xyz [m, 3] turned about the origin-through axis by each angle: [C, m, 3]."""
    k = _normalize(through - origin)
    v = xyz - through
    c, s = numpy.cos(angles)[:, None, None], numpy.sin(angles)[:, None, None]
    kv = v @ k
    return through + v * c + _cross(k, v) * s + k * kv[:, None] * (1 - c)


def _rotation_sign(gg, g, p, step, at):
    """+1 if raising the step's dihedral turns its atom positively about g->p."""
    delta = 0.1
    moved = _place_batch(
        gg, g, p, step.dist, step.theta, numpy.array([step.phi + delta])
    )[0]
    turned = _rotate_about(at[None], g, p, numpy.array([delta]))[0, 0]
    return 1.0 if numpy.linalg.norm(turned - moved) < 1e-3 else -1.0


def _place_differentiably(steps, chosen, out, pose, s, known_atoms, node_global):
    """Rebuild the chosen placement in torch from the input coordinates."""
    local = {a: out[pose, s + a] for a in known_atoms}
    local.update({node: out[pose, g] for node, g in node_global.items()})

    def offset_point(o, through):
        axis = torch.nn.functional.normalize(through - o, dim=-1)
        other = torch.eye(3, dtype=o.dtype, device=o.device)[
            int(torch.argmin(axis.detach().abs()))
        ]
        return o + torch.linalg.cross(axis, other)

    def resolve(ref):
        if not isinstance(ref, tuple):
            return local[ref]
        kind, origin, through = ref
        o = local[origin]
        if kind == "perp" and through is not None:
            return offset_point(o, local[through])
        return o + o.new_tensor([1.0, 0.0, 0.0] if kind == "perp" else [0.0, 1.0, 0.0])

    for step in steps:
        offset = chosen.get(step.axis, 0.0) if step.axis is not None else 0.0
        gg, g, p = (resolve(r) for r in step.refs)
        local[step.atom] = _place_torch(
            gg, g, p, step.dist, step.theta, step.phi + offset
        )
        out[pose, s + step.atom] = local[step.atom]
    return {a: local[a] for a in (st.atom for st in steps)}
