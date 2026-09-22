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
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import attr
import numpy

from tmol.io.details._metal_geometry import choose_geometry, fit_geometry

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

    @property
    def n_donors(self) -> int:
        return len(self.donors)


def ideal_distances(ion: dict, donor_radii: Dict[str, float]) -> Dict[str, float]:
    """Measured metal-ligand distances, completed by ionic_radius + donor_radius."""
    out = dict(ion["distances"])
    for donor, radius in donor_radii.items():
        out.setdefault(donor, round(ion["ionic_radius"] + radius, 3))
    return out


def gather_candidates(
    metal_xyz: numpy.ndarray,
    donor_xyz: numpy.ndarray,
    donor_elements: Sequence[str],
    distances: Dict[str, float],
    tolerance: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Donors close enough to be coordinating, and how far each is past ideal.

    Geometry-free by construction: a cutoff per donor element, nothing more.
    Choosing a polyhedron needs this set, so this set must not depend on one.
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
    ratio = dist / ideal
    keep = numpy.flatnonzero((ratio <= tolerance) & (dist > 1e-6))
    return keep, ratio[keep]


def assign_one(
    metal_xyz: numpy.ndarray,
    ion: dict,
    donor_xyz: numpy.ndarray,
    donor_elements: Sequence[str],
    table: dict,
    metal_index: int = 0,
    geometry: Optional[str] = None,
    tolerance: float = 1.25,
) -> MetalSiteAssignment:
    """Gather, choose a geometry, then fill sites -- in that order."""
    vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
    distances = ideal_distances(ion, table["donor_radii"])
    element, ox = ion["element"], ion["oxidation_state"]

    keep, ratio = gather_candidates(
        metal_xyz, donor_xyz, donor_elements, distances, tolerance
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
                ratio,
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


_METAL_TABLE = None


def metal_table() -> dict:
    """The geometry reference table, loaded once.

    Reads chemical/metals.yaml directly rather than riding on ParameterDatabase.
    It is reference data of the same kind and belongs there eventually; this
    keeps the schema change out of the detection work.
    """
    global _METAL_TABLE
    if _METAL_TABLE is None:
        import os

        from yaml import safe_load

        import tmol.database

        path = os.path.join(
            os.path.dirname(tmol.database.__file__),
            "default",
            "chemical",
            "metals.yaml",
        )
        with open(path) as infile:
            _METAL_TABLE = safe_load(infile)
    return _METAL_TABLE


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
    ion_for_class, metal_atom_index = {}, {}

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
    return CanonicalMetalTables(ion_for_class, metal_atom_index, donor_element)


def find_metal_geometries(
    canonical_ordering,
    chemical_db,
    res_types,
    coords,
    table: Optional[dict] = None,
    geometries: Optional[Dict[Tuple[int, int], str]] = None,
    tolerance: float = 1.25,
    excluded_donor_residues=None,
):
    """Choose a coordination geometry for every metal, as a res_type_variant.

    Mirrors find_disulfides: returns the variant index each residue should take,
    zero everywhere that is not a metal. ``geometries`` declares the geometry
    for a (pose, residue) outright and suppresses inference for it.
    excluded_donor_residues masks residues whose state is already fixed, such
    as disulfide-bonded cysteines, out of the donor candidates.
    """
    import torch

    from tmol.database.chemical import metal_geometry_variant_index

    table = metal_table() if table is None else table
    table_vertices = {g["name"]: g["vertices"] for g in table["geometries"]}
    tables = build_canonical_metal_tables(canonical_ordering, chemical_db, table)
    variants = torch.zeros_like(res_types, dtype=torch.int32)
    if not tables.ion_for_class:
        return variants, []

    rt = res_types.cpu().numpy()
    is_metal_class = numpy.isin(rt, list(tables.ion_for_class))
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


def donor_atoms_by_variant(canonical_ordering, chemical_db):
    """For each class, the canonical atoms each res_type_variant lets donate."""
    from tmol.database.chemical import special_case_variant_index

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


def _assign_crowded(
    metal_index, element, ox, geometry, vertices, directions, keep, ratio
):
    """More candidates than the geometry has sites: keep the closest that fit.

    Each site takes at most one donor, so something has to give. Ranking by how
    far past ideal a contact sits drops the weakest ones, which are what a
    generous cutoff pulled in. Rosetta hard-exits on this case; it is common
    enough in real structures to deserve an answer.
    """
    n_sites = len(vertices)
    order = numpy.argsort(ratio)
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


def place_site_virtuals(pbt, block_types64, block_coords, missing_atoms, assignments):
    """Build each free ion's site virtuals along its fitted vertex directions.

    A lone metal gives the icoor builder no frame to orient them against. The
    fitted rotation carries vertex k onto the donor that took it, so the fan
    starts aligned with the site; with no donors the orientation is arbitrary.
    Virtuals sit at the distance the block type's icoors give them.
    """
    vertices_for = {g["name"]: g["vertices"] for g in metal_table()["geometries"]}
    for (pose, res), got in assignments:
        bt_index = int(block_types64[pose, res])
        if bt_index < 0:
            continue
        bt = pbt.active_block_types[bt_index]
        site = bt.metal_sites[0]
        if not site.site_virts or site.internal_satisfiers:
            continue
        if {a.name for a in bt.atoms} != {site.metal_atom, *site.site_virts}:
            continue  # a cofactor orients its virtuals from its own atoms
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
