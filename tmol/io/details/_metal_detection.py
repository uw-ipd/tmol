"""Decide what coordinates each metal, and with what geometry.

Three decisions, each overridable on its own: which atoms are metals, what
geometry each has, and which donor takes which site. Only the last two are
inferred here -- metal identity comes from the atom's element and is never
guessed.

Candidates are gathered without reference to any polyhedron, because choosing a
geometry needs a set of donors and the set must not presuppose the answer. Only
then is a geometry fitted, and only then are sites assigned.
"""

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, Optional, Sequence, Tuple

import attr
import numpy
import torch

from tmol.chemical import ResidueTypeSet
from tmol.chemical._ideal_coords import build_coords_from_icoors
from tmol.database.chemical import (
    Connection,
    VariantScope,
    VariantType,
    ideal_distances,
    is_metal_cluster,
    metal_geometry_variant_index,
    metal_table,
    special_case_variant_index,
)
from tmol.io.details._metal_geometry import (
    best_rotation,
    choose_geometry,
    fit_geometry,
    unit,
)
from tmol.pose import PackedBlockTypes

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MetalSiteAssignment:
    """One metal, the geometry chosen for it, and what fills its sites."""

    metal: int
    element: str
    oxidation_state: int
    # None when the ion is untemplated: distances are restrained, shape is not
    geometry: Optional[str]
    how_chosen: str
    # indices into the donor arrays, in site order
    donors: Tuple[int, ...]
    # the vertex each donor took; empty for an untemplated ion
    vertex_for_donor: Tuple[int, ...]
    n_open_sites: Optional[int]
    # candidates within range that no site could take
    rejected: Tuple[int, ...] = ()
    # (residue, canonical atom) of each donor, in site order
    donor_atoms: Tuple[Tuple[int, int], ...] = ()
    # carries the geometry's vertices onto the donors; None when nothing fixes it
    rotation: Optional[numpy.ndarray] = attr.ib(default=None, eq=False)
    # per cluster metal: row-vector map carrying its ideal atoms, about the
    #    metal, onto the observed ones; None where the metal was not fitted
    metal_rotations: Tuple[Optional[numpy.ndarray], ...] = attr.ib(default=(), eq=False)
    # given by the caller rather than inferred; its input virtuals are kept
    declared: bool = False

    @property
    def n_donors(self) -> int:
        return len(self.donors)


def gather_candidates(
    metal_xyz: numpy.ndarray,
    donor_xyz: numpy.ndarray,
    donor_elements: Sequence[str],
    distances: Dict[str, float],
    tolerance: float,
    required: Sequence[int] = (),
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Donors close enough to be coordinating, and how far each is past ideal.

    Geometry-free by construction: a donor is kept within ``tolerance`` A past
    its element's ideal distance, nothing more.
    Choosing a polyhedron needs this set, so this set must not depend on one.
    Required donors are kept at any distance.
    """
    if len(donor_xyz) == 0:
        return numpy.zeros(0, dtype=int), numpy.zeros(0)
    deltas = numpy.asarray(donor_xyz, dtype=numpy.float64) - numpy.asarray(
        metal_xyz, dtype=numpy.float64
    )
    dist = numpy.linalg.norm(deltas, axis=1)
    ideal = numpy.array(
        [distances.get(e, distances["O"]) for e in donor_elements], dtype=numpy.float64
    )
    excess = dist - ideal
    is_required = numpy.zeros(len(dist), dtype=bool)
    is_required[list(required)] = True
    keep = numpy.flatnonzero(((excess <= tolerance) & (dist > 1e-6)) | is_required)
    return keep, excess[keep]


def assign_one(
    metal_xyz: numpy.ndarray,
    ion: dict,
    donor_xyz: numpy.ndarray,
    donor_elements: Sequence[str],
    table: dict,
    metal_index: int = 0,
    geometry: Optional[str] = None,
    tolerance: float = 0.55,
    required: Sequence[int] = (),
) -> MetalSiteAssignment:
    """Gather, choose a geometry, then fill sites -- in that order.

    Required donors (declared bonds) are always gathered, and are the last to
    be dropped when there are more donors than sites.
    """
    vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
    distances = ideal_distances(ion, table["donor_radii"])
    element, ox = ion["element"], ion["oxidation_state"]

    keep, excess = gather_candidates(
        metal_xyz, donor_xyz, donor_elements, distances, tolerance, required
    )
    directions = numpy.asarray(donor_xyz, dtype=numpy.float64)[keep] - numpy.asarray(
        metal_xyz, dtype=numpy.float64
    )

    if geometry is not None:
        how = "declared"
        verts = vertices_for[geometry]
        fit = fit_geometry(directions, numpy.asarray(verts)) if verts else None
    else:
        fit, how = choose_geometry(directions, ion["geometries"], vertices_for)
        geometry = fit.geometry if fit is not None else None

    if fit is None and how != "untemplated" and len(keep):
        # too crowded for any candidate, which is not the same as untemplated:
        # the ion does have a shape, there are simply more contacts than it can
        # hold. Take its roomiest geometry and let the excess go.
        geometry = max(
            (g for g in ion["geometries"] if vertices_for[g]),
            key=lambda g: len(vertices_for[g]),
            default=None,
        )
        if geometry is not None:
            return _assign_crowded(
                metal_index,
                element,
                ox,
                geometry,
                vertices_for[geometry],
                directions,
                keep,
                numpy.where(numpy.isin(keep, list(required)), -numpy.inf, excess),
            )

    if fit is None:
        # untemplated, or nothing within range at all: no polyhedron to assign
        # against, so every candidate simply stands. A declared geometry still
        # holds -- it is an override, and nothing here is evidence against it.
        declared = how == "declared" and vertices_for.get(geometry)
        return MetalSiteAssignment(
            metal=metal_index,
            element=element,
            oxidation_state=ox,
            geometry=geometry if declared else None,
            how_chosen=how,
            donors=tuple(int(i) for i in keep),
            vertex_for_donor=(),
            n_open_sites=len(declared) - len(keep) if declared else None,
        )

    return MetalSiteAssignment(
        metal=metal_index,
        element=element,
        oxidation_state=ox,
        geometry=geometry,
        how_chosen=how,
        donors=tuple(int(i) for i in keep),
        vertex_for_donor=fit.vertex_for_donor,
        n_open_sites=fit.n_open_sites,
        rotation=fit.rotation,
    )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CanonicalMetalTables:
    """Per-equivalence-class lookups detection needs before block types exist.

    Geometry has to be chosen *before* a block type is picked, so none of this
    can come from one. It is all keyed on the io equivalence class, which the
    input names, and the canonical atom index, which the ordering fixes.
    """

    # equivalence class index -> its entry in metals.yaml, for the metal ions
    ion_for_class: Dict[int, dict]
    # equivalence class index -> canonical index of the metal atom itself
    metal_atom_index: Dict[int, int]
    # [n_classes, max_n_canonical_atoms], "" where the atom cannot donate
    donor_element: numpy.ndarray
    # equivalence class index -> its cluster's metals, for multi-metal cofactors
    cluster_for_class: Dict[int, "ClusterSites"] = attr.Factory(dict)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ClusterMetal:
    """One metal of a cluster: where it is, what fills it, and its free sites."""

    atom: int
    ion: dict
    satisfiers: Tuple[int, ...]
    # the cluster's global index of each free site
    sites: Tuple[int, ...]
    # ideal positions of the metal, its satisfiers, then its free-site virtuals
    ideal: numpy.ndarray = attr.ib(eq=False)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ClusterSites:
    """A cofactor holding several metals, each with sites its own atoms fill."""

    geometry: str
    variant: int
    metals: Tuple[ClusterMetal, ...]
    # every metal atom, those without sites (a calcium among manganese) too
    metal_atoms: Tuple[int, ...] = ()

    @property
    def n_sites(self) -> int:
        return sum(len(m.sites) for m in self.metals)


def build_canonical_metal_tables(
    canonical_ordering, chemical_db, table: dict
) -> CanonicalMetalTables:
    """Fold the chemical database down to what detection reads per atom."""
    atom_type = {at.name: at for at in chemical_db.atom_types}
    ion_for_name3 = {ion["name3"]: ion for ion in table["ions"]}

    members = defaultdict(list)
    for res in chemical_db.residues:
        members[res.io_equiv_class].append(res)

    classes = canonical_ordering.restype_io_equiv_classes
    n_atoms = canonical_ordering.max_n_canonical_atoms
    donor_element = numpy.full((len(classes), n_atoms), "", dtype=object)
    ion_for_class, metal_atom_index, cluster_for_class = {}, {}, {}

    for i, equiv_class in enumerate(classes):
        index_of = canonical_ordering.restypes_atom_index_mapping[equiv_class]
        # an atom donates if any type in its class lets it: which protonation
        # state the residue takes is decided by what it coordinates
        for res in members.get(equiv_class, ()):
            for atom in res.atoms:
                at = atom_type.get(atom.atom_type)
                if at is None or not at.is_metal_donor:
                    continue
                j = index_of.get(atom.name)
                if j is not None:
                    # water is typed apart from other oxygens: it is what an
                    # open site is assumed to hold, so it has its own distance
                    donor_element[i, j] = "Owat" if at.name == "Owat" else at.element
        if equiv_class in ion_for_name3:
            sites = [r.metal_sites[0] for r in members[equiv_class] if r.metal_sites]
            ion_for_class[i] = ion_for_name3[equiv_class]
            metal_atom_index[i] = index_of[sites[0].metal_atom]
            continue
        cluster = next(
            (r for r in members.get(equiv_class, ()) if is_metal_cluster(r)),
            None,
        )
        if cluster is not None:
            cluster_for_class[i] = _cluster_sites(cluster, index_of, atom_type, table)
    return CanonicalMetalTables(
        ion_for_class, metal_atom_index, donor_element, cluster_for_class
    )


def _raw_ideal_coords(res):
    """{atom name: ideal position} built from a raw residue's icoors."""
    names = [ic.name for ic in res.icoors]
    index = {n: i for i, n in enumerate(names)}
    ancestors = numpy.array(
        [
            [index[ic.parent], index[ic.grand_parent], index[ic.great_grand_parent]]
            for ic in res.icoors
        ],
        dtype=numpy.int32,
    )
    geom = numpy.array([[ic.phi, ic.theta, ic.d] for ic in res.icoors])
    coords = build_coords_from_icoors(ancestors, geom).astype(numpy.float64)
    return dict(zip(names, coords))


def _cluster_sites(res, index_of, atom_type, table):
    """Detection's view of a cluster: each metal, its satisfiers and free sites."""
    ion_for_atom_type = {ion["atom_type"]: ion for ion in table["ions"]}
    types = {a.name: a.atom_type for a in res.atoms}
    ideal = _raw_ideal_coords(res)
    metals, first = [], 0
    for site in res.metal_sites:
        names = (site.metal_atom, *site.internal_satisfiers, *site.site_virts)
        metals.append(
            ClusterMetal(
                atom=index_of[site.metal_atom],
                ion=ion_for_atom_type[types[site.metal_atom]],
                satisfiers=tuple(index_of[a] for a in site.internal_satisfiers),
                sites=tuple(range(first, first + len(site.site_connections))),
                ideal=numpy.array([ideal[n] for n in names]),
            )
        )
        first += len(site.site_connections)
    return ClusterSites(
        geometry=res.metal_sites[0].geometry,
        variant=special_case_variant_index(res),
        metals=tuple(metals),
        metal_atoms=tuple(
            index_of[a.name] for a in res.atoms if a.atom_type in ion_for_atom_type
        ),
    )


def _assign_cluster_site(
    pose,
    res,
    cluster,
    rt,
    xyz,
    excluded,
    tables,
    table,
    declared_sites,
    required_donors,
    tolerance,
    find_additional,
):
    """A cluster's donors: as declared with their sites, or detected around it."""
    if (pose, res) in (declared_sites or {}):
        _, filled = declared_sites[(pose, res)]
        first = cluster.metals[0].ion
        rotations = []
        for m in cluster.metals:
            local = {site: k for k, site in enumerate(m.sites)}
            mine = [
                (local[int(site)], xyz[pose, int(r), int(a)])
                for site, r, a in filled
                if int(site) in local
            ]
            rotations.append(
                _metal_rotation(
                    m, xyz[pose, res], [p for _, p in mine], [k for k, _ in mine]
                )
            )
        return MetalSiteAssignment(
            metal=res,
            element=first["element"],
            oxidation_state=first["oxidation_state"],
            geometry=cluster.geometry,
            how_chosen="declared",
            donors=tuple(range(len(filled))),
            vertex_for_donor=tuple(int(site) for site, _, _ in filled),
            n_open_sites=cluster.n_sites - len(filled),
            donor_atoms=tuple((int(r), int(a)) for _, r, a in filled),
            metal_rotations=tuple(rotations),
            declared=True,
        )
    donor_xyz, donor_elements, donor_atoms = [], [], []
    for other in range(rt.shape[1]):
        if other == res or rt[pose, other] < 0 or excluded[pose, other]:
            continue
        for j, element in enumerate(tables.donor_element[rt[pose, other]]):
            p = xyz[pose, other, j]
            if element and numpy.all(numpy.isfinite(p)):
                donor_xyz.append(p)
                donor_elements.append(element)
                donor_atoms.append((other, j))
    wanted = (required_donors or {}).get((pose, res), set())
    return _cluster_assignment(
        res,
        cluster,
        xyz[pose, res],
        donor_xyz,
        donor_elements,
        donor_atoms,
        wanted,
        table,
        tolerance,
        find_additional,
    )


def _cluster_assignment(
    res,
    cluster,
    xyz,
    donor_xyz,
    donor_elements,
    donor_atoms,
    wanted,
    table,
    tolerance,
    find_additional,
):
    """Fill each metal's free sites: declared donors first, then the closest.

    A site takes the donor its ideal direction points at best, the ideal
    directions coming from the cluster's own atoms as observed. A donor in
    range of several metals bridges them, filling a site on each.
    """
    donor_xyz = numpy.asarray(donor_xyz, dtype=numpy.float64).reshape(-1, 3)
    sites, chosen, rotations = [], [], []
    metal_xyz = [xyz[a] for a in cluster.metal_atoms]
    for m in cluster.metals:
        center = xyz[m.atom]
        frame = numpy.array([center, *(xyz[a] for a in m.satisfiers)])
        if not numpy.isfinite(frame).all() or not len(m.sites):
            rotations.append(None)
            continue
        distances = ideal_distances(m.ion, table["donor_radii"])
        order = []
        for i, element in enumerate(donor_elements):
            r = numpy.linalg.norm(donor_xyz[i] - center)
            nearest = numpy.argmin(
                [numpy.linalg.norm(donor_xyz[i] - c) for c in metal_xyz]
            )
            excess = r - distances.get(element, distances["O"])
            if donor_atoms[i] in wanted and (
                cluster.metal_atoms[nearest] == m.atom or excess <= tolerance
            ):
                order.append((-1.0, i))
            elif find_additional and excess <= tolerance:
                order.append((excess, i))
        order = [i for _, i in sorted(order)][: len(m.sites)]
        rotation = _metal_rotation(m, xyz, donor_xyz[order])
        rotations.append(rotation)
        n_frame = 1 + len(m.satisfiers)
        rays = unit((m.ideal[n_frame:] - m.ideal[0]) @ rotation)
        free = list(range(len(m.sites)))
        for i in order:
            direction = unit(donor_xyz[i] - center)
            best = max(free, key=lambda k: float(rays[k] @ direction))
            free.remove(best)
            sites.append(m.sites[best])
            chosen.append(i)
    first = cluster.metals[0].ion
    return MetalSiteAssignment(
        metal=res,
        element=first["element"],
        oxidation_state=first["oxidation_state"],
        geometry=cluster.geometry,
        how_chosen="cluster",
        donors=tuple(range(len(chosen))),
        vertex_for_donor=tuple(sites),
        n_open_sites=cluster.n_sites - len(chosen),
        donor_atoms=tuple(donor_atoms[i] for i in chosen),
        metal_rotations=tuple(rotations),
    )


def _metal_rotation(metal, xyz, donors, assigned=None):
    """Row-vector map, about the metal, carrying its ideal atoms onto ``xyz``.

    The metal's own satisfiers fix what they can: spread in three dimensions
    they fix the frame, in a plane they leave a mirror through it, on a line
    they leave the spin about it, and with none the donors fix it all. Donors choose among what is left: each
    donor scores the site ``assigned`` to it, or else its best site.
    """
    n_frame = 1 + len(metal.satisfiers)
    center = numpy.asarray(xyz[metal.atom], dtype=numpy.float64)
    b = numpy.array([xyz[j] for j in metal.satisfiers], dtype=numpy.float64)
    b = b.reshape(-1, 3) - center
    if not numpy.isfinite(center).all() or not numpy.isfinite(b).all():
        return None
    donors = numpy.asarray(donors, dtype=numpy.float64).reshape(-1, 3)
    placed = numpy.isfinite(donors).all(axis=1)
    donors = unit(donors[placed] - center)
    if assigned is not None:
        assigned = numpy.asarray(assigned, dtype=numpy.int64)[placed]
    a = metal.ideal[1:n_frame] - metal.ideal[0]
    free = unit(metal.ideal[n_frame:] - metal.ideal[0])
    candidates = _frame_candidates(a, b, free, donors)
    if len(candidates) == 1 or not len(donors):
        return candidates[0]

    def score(rotation):
        dots = donors @ (free @ rotation).T
        if assigned is not None:
            return float(dots[numpy.arange(len(donors)), assigned].sum())
        return float(dots.max(axis=1).sum())

    return max(candidates, key=score)


def _frame_candidates(a, b, free, donors):
    """Maps carrying satisfiers ``a`` onto ``b``, spun toward unit ``donors``."""
    if len(a) == 0:
        # no satisfiers: some site faces the first donor, spun toward the rest
        if not len(donors):
            return [numpy.eye(3)]
        out = []
        for r in free:
            out += _spins(_aligning(r, donors[0]), donors[0], free, donors[1:])
        return out
    u, s, vt = numpy.linalg.svd(a.T @ b)
    if s[1] < 1e-6 * s[0]:
        # collinear satisfiers fix an axis only
        axis = unit(b[0])
        return _spins(_aligning(unit(a[0]), axis), axis, free, donors)
    out = [u @ vt]
    if s[2] < 1e-6 * s[0]:
        # coplanar satisfiers leave a mirror through their plane
        out.append(u @ numpy.diag([1.0, 1.0, -1.0]) @ vt)
    return out


def _spins(start, axis, free, donors):
    """``start``, and ``start`` spun about ``axis`` to put a site on each donor."""
    out = [start]
    for d in donors:
        q = d - (d @ axis) * axis
        for r in free @ start:
            p = r - (r @ axis) * axis
            theta = math.atan2(float(axis @ numpy.cross(p, q)), float(p @ q))
            out.append(start @ _about(axis, theta).T)
    return out


def _about(axis, theta):
    """Column-vector rotation by ``theta`` about unit ``axis``."""
    k = numpy.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return numpy.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _aligning(u, v):
    """Row-vector rotation carrying unit ``u`` onto unit ``v``."""
    axis = numpy.cross(u, v)
    sin = numpy.linalg.norm(axis)
    if sin < 1e-9:
        if u @ v > 0:
            return numpy.eye(3)
        axis = numpy.cross(u, numpy.eye(3)[numpy.argmin(numpy.abs(u))])
        sin = 0.0
    return _about(unit(axis), math.atan2(sin, float(u @ v))).T


def find_metal_geometries(
    canonical_ordering,
    chemical_db,
    res_types,
    coords,
    table: Optional[dict] = None,
    geometries: Optional[Dict[Tuple[int, int], str]] = None,
    tolerance: float = 0.55,
    excluded_donor_residues=None,
    declared_sites: Optional[Dict[Tuple[int, int], tuple]] = None,
    find_additional: bool = True,
    required_donors: Optional[Dict[Tuple[int, int], set]] = None,
):
    """Choose a coordination geometry for every metal, as a res_type_variant.

    Mirrors find_disulfides: returns the variant index each residue should take,
    zero everywhere that is not a metal. ``geometries`` declares the geometry
    for a (pose, residue) outright and suppresses inference for it.
    excluded_donor_residues masks residues whose state is already fixed, such
    as disulfide-bonded cysteines, out of the donor candidates.

    ``declared_sites`` maps (pose, metal) to (geometry, ((site, donor, atom),
    ...)) and is taken as given. ``required_donors`` maps (pose, metal) to the
    (donor, atom) pairs it is declared bonded to, without sites: these are
    always kept, and the geometry and sites are chosen around them. Without
    ``find_additional``, a metal takes only its required donors.
    """
    table = metal_table() if table is None else table
    table_vertices = {g["name"]: g["vertices"] for g in table["geometries"]}
    tables = build_canonical_metal_tables(canonical_ordering, chemical_db, table)
    variants = torch.zeros_like(res_types, dtype=torch.int32)
    if not tables.ion_for_class and not tables.cluster_for_class:
        return variants, []

    rt = res_types.cpu().numpy()
    is_metal_class = numpy.isin(rt, [*tables.ion_for_class, *tables.cluster_for_class])
    if not is_metal_class.any():
        return variants, []
    # the choice is discrete, so coordinates enter it without gradient
    xyz = coords.detach().cpu().numpy()
    excluded = (
        numpy.zeros(rt.shape, dtype=bool)
        if excluded_donor_residues is None
        else excluded_donor_residues.cpu().numpy()
    )
    assignments = []

    for pose, res in zip(*numpy.nonzero(is_metal_class)):
        cls = int(rt[pose, res])
        if cls in tables.cluster_for_class:
            cluster = tables.cluster_for_class[cls]
            variants[pose, res] = cluster.variant
            got = _assign_cluster_site(
                int(pose),
                int(res),
                cluster,
                rt,
                xyz,
                excluded,
                tables,
                table,
                declared_sites,
                required_donors,
                tolerance,
                find_additional,
            )
            assignments.append(((int(pose), int(res)), got))
            continue
        ion = tables.ion_for_class[cls]
        metal_xyz = xyz[pose, res, tables.metal_atom_index[cls]]
        if not numpy.all(numpy.isfinite(metal_xyz)):
            continue

        donor_xyz, donor_elements, donor_atoms = [], [], []
        for other in range(rt.shape[1]):
            if other == res or rt[pose, other] < 0 or excluded[pose, other]:
                continue
            for j, element in enumerate(tables.donor_element[rt[pose, other]]):
                if not element:
                    continue
                p = xyz[pose, other, j]
                if numpy.all(numpy.isfinite(p)):
                    donor_xyz.append(p)
                    donor_elements.append(element)
                    donor_atoms.append((other, j))

        if (int(pose), int(res)) in (declared_sites or {}):
            geometry, filled = declared_sites[(int(pose), int(res))]
            got = _declared_assignment(
                int(res), ion, geometry, filled, metal_xyz, xyz[pose], table_vertices
            )
            variants[pose, res] = metal_geometry_variant_index(geometry)
            assignments.append(((int(pose), int(res)), got))
            continue
        wanted = (required_donors or {}).get((int(pose), int(res)), set())
        for missing in wanted - set(donor_atoms):
            logger.warning(
                "declared bond from residue %d to atom %d of residue %d is not "
                "to a resolved metal donor; ignored",
                res,
                missing[1],
                missing[0],
            )
        if not find_additional:
            kept = [i for i, atom in enumerate(donor_atoms) if atom in wanted]
            donor_xyz = [donor_xyz[i] for i in kept]
            donor_elements = [donor_elements[i] for i in kept]
            donor_atoms = [donor_atoms[i] for i in kept]
        required = [i for i, atom in enumerate(donor_atoms) if atom in wanted]

        declared = (geometries or {}).get((int(pose), int(res)))
        got = assign_one(
            metal_xyz,
            ion,
            numpy.asarray(donor_xyz).reshape(-1, 3),
            donor_elements,
            table,
            metal_index=int(res),
            geometry=declared,
            tolerance=tolerance,
            required=required,
        )
        # an untemplated ion, or one with nothing in range, still needs a block
        # type: fall back on the ion's most common geometry
        geometry = got.geometry or ion["geometries"][0]
        n_sites = len(table_vertices[geometry])
        if got.geometry is None and n_sites:
            got = attr.evolve(
                got,
                geometry=geometry,
                how_chosen=got.how_chosen + "; default geometry",
                n_open_sites=n_sites - got.n_donors,
            )
        variants[pose, res] = metal_geometry_variant_index(geometry)
        got = attr.evolve(got, donor_atoms=tuple(donor_atoms[i] for i in got.donors))
        assignments.append(((int(pose), int(res)), got))
    return variants, assignments


def _declared_assignment(metal, ion, geometry, filled, metal_xyz, xyz, vertices_for):
    """The caller's sites for one metal, with the fan fitted to its donors."""
    sites = tuple(int(site) for site, _, _ in filled)
    donor_atoms = tuple((int(res), int(atom)) for _, res, atom in filled)
    verts = numpy.asarray(vertices_for[geometry], dtype=numpy.float64)
    rotation = None
    if len(verts) and donor_atoms:
        directions = unit(numpy.array([xyz[r, a] for r, a in donor_atoms]) - metal_xyz)
        rotation = best_rotation(directions, unit(verts[list(sites)]))
    return MetalSiteAssignment(
        metal=metal,
        element=ion["element"],
        oxidation_state=ion["oxidation_state"],
        geometry=geometry,
        how_chosen="declared",
        donors=tuple(range(len(donor_atoms))),
        vertex_for_donor=sites,
        n_open_sites=len(verts) - len(sites) if len(verts) else None,
        donor_atoms=donor_atoms,
        rotation=rotation,
        declared=True,
    )


def donor_atoms_by_variant(canonical_ordering, chemical_db):
    """For each class, the canonical atoms each res_type_variant lets donate."""
    atom_type = {at.name: at for at in chemical_db.atom_types}
    class_index = {
        c: i for i, c in enumerate(canonical_ordering.restype_io_equiv_classes)
    }
    out = defaultdict(lambda: defaultdict(set))
    for res in chemical_db.residues:
        i = class_index.get(res.io_equiv_class)
        if i is None:
            continue
        index_of = canonical_ordering.restypes_atom_index_mapping[res.io_equiv_class]
        v = special_case_variant_index(res)
        for atom in res.atoms:
            at = atom_type.get(atom.atom_type)
            if at is not None and at.is_metal_donor and atom.name in index_of:
                out[i][v].add(index_of[atom.name])
    return out


def select_donor_variants(
    canonical_ordering, chemical_db, res_types, res_type_variants, assignments
):
    """Move each coordinating residue to a variant in which its donors donate.

    A thiol, a phenol or the protonated nitrogen of a histidine tautomer cannot
    coordinate; the deprotonated form, or the other tautomer, can. Residues
    already in a variant that satisfies every donor are left alone.
    """
    required = defaultdict(set)
    for (pose, _), got in assignments:
        for res, atom in got.donor_atoms:
            required[(pose, res)].add(atom)
    if not required:
        return res_type_variants

    donates = donor_atoms_by_variant(canonical_ordering, chemical_db)
    out = res_type_variants.clone()
    for (pose, res), atoms in required.items():
        cls = int(res_types[pose, res])
        current = int(out[pose, res])
        if atoms <= donates[cls][current]:
            continue
        options = sorted(v for v, can in donates[cls].items() if atoms <= can)
        if not options:
            logger.warning(
                "%s %d coordinates through atoms no form of it can donate",
                canonical_ordering.restype_io_equiv_classes[cls],
                res,
            )
            continue
        out[pose, res] = options[0]
    return out


def metal_donor_patch(base_name: str, atom: str, n_metals: int = 1) -> VariantType:
    """Give one atom of one residue type a connection per metal it coordinates.

    Connections are metal_<atom>, metal2_<atom>, ...: a count before the
    separator cannot be mistaken for part of an atom name. The patch is named
    for its last connection, so the name a patched type carries is one of its
    own connection names.
    """
    names = [f"metal_{atom}"] + [f"metal{k}_{atom}" for k in range(2, n_metals + 1)]
    prefix = f"metal{n_metals}" if n_metals > 1 else "metal"
    return VariantType(
        name=f"{prefix}_{base_name}_{atom}",
        display_name=names[-1],
        pattern="",
        remove_atoms=(),
        add_atoms=(),
        add_atom_aliases=(),
        modify_atoms=(),
        add_connections=tuple(
            Connection(name=name, atom=f"<{atom}>", kinematic=False) for name in names
        ),
        add_bonds=(),
        icoors=(),
        applies_to=VariantScope(base_names=(base_name,)),
    )


def donor_patches(canonical_ordering, chemical_db, res_types, assignments):
    """One patch per (base type, atom, metal count) an assigned donor needs.

    Every base type of the donor's class where that atom donates gets one,
    since which of them the residue becomes is decided after this. An atom
    bridging two metals needs two connections on it.
    """
    atom_type = {at.name: at for at in chemical_db.atom_types}
    bases = defaultdict(list)
    for res in chemical_db.residues:
        if res.name == res.base_name:
            bases[res.io_equiv_class].append(res)
    classes = canonical_ordering.restype_io_equiv_classes
    n_metals = Counter(
        (pose, res, atom)
        for (pose, _), got in assignments
        for res, atom in got.donor_atoms
    )
    patches = {}
    for (pose, res, atom), n in n_metals.items():
        equiv_class = classes[int(res_types[pose, res])]
        name = canonical_ordering.restypes_ordered_atom_names[equiv_class][atom]
        for base in bases[equiv_class]:
            # the atom may be the base's own or one a variant adds (a C-terminal OXT)
            atoms = [
                *base.atoms,
                *(
                    a
                    for v in chemical_db.variants
                    if v.applies_to.matches(base)
                    for a in v.add_atoms
                ),
            ]
            if any(
                a.name == name and atom_type[a.atom_type].is_metal_donor for a in atoms
            ):
                patches[(base.name, name, n)] = metal_donor_patch(base.name, name, n)
    return tuple(patches.values())


def with_donor_patches(pbt: PackedBlockTypes, patches) -> PackedBlockTypes:
    """The newest packed block types grown from pbt that carry every patch.

    Donor forms accumulate: each extension appends to the newest generation
    grown from the same root, so every pose built from one context shares a
    packed set until a donor it has not seen arrives, and each generation's
    residues are a leading run of the next's.
    """
    if not patches:
        return pbt
    root = getattr(pbt, "_donor_patch_root", pbt)
    newest = getattr(root, "_donor_patch_newest", root)
    known = {v.name for v in newest.chem_db.variants}
    extra = tuple(p for p in patches if p.name not in known)
    if extra:
        chem_db = newest.chem_db.with_variants_applied(extra)
        rts = ResidueTypeSet.from_database(chem_db)
        newest = PackedBlockTypes.from_restype_list(
            chem_db, rts, rts.residue_types, pbt.device
        )
        setattr(newest, "_donor_patch_root", root)
        setattr(root, "_donor_patch_newest", newest)
    return newest


def metal_connection_rows(assignments):
    """(pose, metal, site, donor, donor canonical atom) for every filled site.

    A templated ion's donor fills the site of the vertex it was fitted to; an
    untemplated ion's donors fill its sites in order.
    """
    rows = []
    for (pose, metal), got in assignments:
        sites = got.vertex_for_donor or range(len(got.donor_atoms))
        for site, (res, atom) in zip(sites, got.donor_atoms):
            rows.append((pose, metal, int(site), int(res), int(atom)))
    return rows


def _assign_crowded(
    metal_index, element, ox, geometry, vertices, directions, keep, excess
):
    """More candidates than the geometry has sites: keep the closest that fit.

    Each site takes at most one donor, so something has to give. Ranking by how
    far past ideal a contact sits drops the weakest ones, which are what a
    generous cutoff pulled in. Rosetta hard-exits on this case; it is common
    enough in real structures to deserve an answer.
    """
    n_sites = len(vertices)
    order = numpy.argsort(excess)
    chosen = numpy.sort(order[:n_sites])
    dropped = numpy.sort(order[n_sites:])
    fit = fit_geometry(directions[chosen], numpy.asarray(vertices))
    logger.info(
        "%s%d+ has %d candidates for %d sites; dropped %d beyond ideal",
        element,
        ox,
        len(keep),
        n_sites,
        len(dropped),
    )
    return MetalSiteAssignment(
        metal=metal_index,
        element=element,
        oxidation_state=ox,
        geometry=geometry,
        how_chosen="directions, crowded",
        donors=tuple(int(keep[i]) for i in chosen),
        vertex_for_donor=fit.vertex_for_donor if fit else (),
        n_open_sites=fit.n_open_sites if fit else None,
        rejected=tuple(int(keep[i]) for i in dropped),
        rotation=fit.rotation if fit else None,
    )


def _place_cluster_virtuals(bt, coords, missing, rotations):
    """Carry each metal's ideal free-site virtuals by its fitted rotation."""
    ideal = bt.ideal_coords  # in icoor order
    for site, rotation in zip(bt.metal_sites, rotations):
        virts = [bt.atom_to_idx[v] for v in site.site_virts]
        if rotation is None or not virts or not bool(missing[virts].any()):
            continue
        metal = coords[bt.atom_to_idx[site.metal_atom]].detach().cpu().numpy()
        template = ideal[bt.icoors_index[site.metal_atom]]
        for j in virts:
            offset = ideal[bt.icoors_index[bt.atoms[j].name]] - template
            coords[j] = coords.new_tensor(metal + offset @ rotation)
            missing[j] = False


def place_site_virtuals(pbt, block_types64, block_coords, missing_atoms, assignments):
    """Build each free ion's site virtuals along its fitted vertex directions.

    A lone metal gives the icoor builder no frame to orient them against. The
    fitted rotation carries vertex k onto the donor that took it, so the fan
    starts aligned with the site; with no donors the orientation is arbitrary.
    Virtuals sit at the distance the block type's icoors give them. A declared
    site whose virtuals all came with the input keeps them.
    """
    vertices_for = {g["name"]: g["vertices"] for g in metal_table()["geometries"]}
    for (pose, res), got in assignments:
        bt_index = int(block_types64[pose, res])
        if bt_index < 0:
            continue
        bt = pbt.active_block_types[bt_index]
        if is_metal_cluster(bt):
            _place_cluster_virtuals(
                bt,
                block_coords[pose, res],
                missing_atoms[pose, res],
                got.metal_rotations,
            )
            continue
        site = bt.metal_sites[0]
        if not site.site_virts or site.internal_satisfiers:
            continue
        if {a.name for a in bt.atoms} != {site.metal_atom, *site.site_virts}:
            continue  # a cofactor orients its virtuals from its own atoms
        virts = [bt.atom_to_idx[name] for name in site.site_virts]
        if got.declared and not bool(missing_atoms[pose, res, virts].any()):
            continue
        rotation = got.rotation if got.geometry == site.geometry else None
        rotation = numpy.eye(3) if rotation is None else rotation
        dist = {ic.name: ic.d for ic in bt.icoors}
        verts = numpy.asarray(vertices_for[site.geometry], dtype=numpy.float64)
        verts /= numpy.linalg.norm(verts, axis=1, keepdims=True)
        metal = block_coords[pose, res, bt.atom_to_idx[site.metal_atom]]
        for vertex, name in zip(verts @ rotation.T, site.site_virts):
            j = bt.atom_to_idx[name]
            block_coords[pose, res, j] = metal + metal.new_tensor(vertex * dist[name])
            missing_atoms[pose, res, j] = False
    return block_coords, missing_atoms
