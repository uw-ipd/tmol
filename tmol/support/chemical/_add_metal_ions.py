"""Generate the free metal-ion residue types and their electrostatic charges.

Reads ``chemical/metals.yaml`` and writes ``chemical/metal_ions.yaml`` plus
``scoring/elec_metal_ions.yaml``. One residue type per realised (element,
oxidation state, coordination geometry) triple: the metal atom, and one virtual
atom marking each vertex of the geometry.

The virtuals sit at the ion's measured metal-water distance, because an
unoccupied site is taken to hold a water and that is where lk_ball builds it.
An untemplated geometry has no vertices, so those ions get no virtuals and no
site waters -- their metal-ligand distances are restrained but their geometry
is not.
"""

import math
import os

import numpy
import yaml
from yaml import safe_load

CHEM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "default", "chemical"
)
SCORING_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "default", "scoring"
)

# donors we place virtuals for; the site marks where a water would sit
SITE_DONOR = "Owat"


def fill_distances(ion, donor_radii):
    """Measured distances, completed by ionic_radius + donor_radius."""
    out = dict(ion["distances"])
    for donor, radius in donor_radii.items():
        out.setdefault(donor, round(ion["ionic_radius"] + radius, 3))
    return out


def normalize(v):
    return v / numpy.linalg.norm(v)


def frame_from_coords(p1, p2, p3):
    """The builder's frame: origin at p3, z from p2 to p3."""
    z = normalize(p3 - p2)
    v21 = normalize(p1 - p2)
    y = normalize(v21 - numpy.dot(z, v21) * z)
    x = normalize(numpy.cross(y, z))
    return x, y, z


def icoors_for_sites(vertices, dist):
    """Internal coordinates placing the metal at the origin and one virtual per vertex.

    Inverts tmol.chemical._ideal_coords.build_coords_from_icoors, whose theta is
    the supplement of the conventional bond angle: an atom is placed by rotating
    phi about the frame's z, then -theta about x, then stepping d along z.
    """
    sites = [numpy.array(v, dtype=numpy.float64) * dist for v in vertices]
    rows = []
    # the metal is the root; the first three atoms name each other as the frame
    rows.append((0.0, 0.0, 0.0))
    if not sites:
        return rows
    rows.append((0.0, 180.0, dist))  # V1 goes on +x; theta is unused here
    if len(sites) > 1:
        # V2 hangs off the metal, so the builder takes pi - theta as its angle
        cos_a = numpy.dot(normalize(sites[0]), normalize(sites[1]))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
        rows.append((0.0, 180.0 - angle, dist))
    if len(sites) > 2:
        # the dihedral frame is built from V1 and V2, so they must not be
        # collinear through the metal -- order trans pairs apart in the table
        cos_12 = numpy.dot(normalize(sites[0]), normalize(sites[1]))
        if abs(cos_12) > 0.999:
            raise ValueError(
                "first two vertices are collinear through the metal, leaving "
                "the dihedral frame undefined; reorder the geometry's vertices"
            )
    for site in sites[2:]:
        x, y, z = frame_from_coords(sites[1], sites[0], numpy.zeros(3))
        u = normalize(site)
        a, b, c = numpy.dot(u, x), numpy.dot(u, y), numpy.dot(u, z)
        theta = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        phi = math.degrees(math.atan2(-a, b))
        rows.append((phi, theta, dist))
    return rows


def residue_for(ion, geometry, vertices, distances, n_sites):
    """The metal, a virtual per templated site, and a connection per site."""
    # a single-ion component names its atom by element, whatever its code
    metal = ion["element"].upper()
    name3 = ion["name3"]
    name = f"{name3}_{geometry}"
    virts = [f"V{i + 1}" for i in range(len(vertices))]
    sites = [f"site{i + 1}" for i in range(n_sites)]
    icoor_rows = icoors_for_sites(vertices, distances[SITE_DONOR])

    # the first three atoms name each other, as HOH does
    stub = [metal] + virts
    stub = (stub + [metal, metal])[:3]

    atoms = [{"name": metal, "atom_type": ion["atom_type"]}]
    atoms += [{"name": v, "atom_type": "Vrt"} for v in virts]
    icoors = []
    for (phi, theta, d), atom in zip(icoor_rows, [metal] + virts):
        icoors.append(
            {
                "name": atom,
                "phi": f"{phi:.6f} deg",
                "theta": f"{theta:.6f} deg",
                "d": round(d, 6),
                "parent": stub[0],
                "grand_parent": stub[1],
                "great_grand_parent": stub[2],
            }
        )
    return {
        "name": name,
        "base_name": name,
        "name3": name3,
        "io_equiv_class": name3,
        "atoms": atoms,
        "atom_aliases": [],
        "bonds": [[metal, v, "SINGLE", False] for v in virts],
        "connections": [
            {"name": c, "atom": metal, "type": "SINGLE", "kinematic": False}
            for c in sites
        ],
        "torsions": [],
        "icoors": icoors,
        "properties": {
            "is_canonical": False,
            "polymer": {
                "is_polymer": False,
                "polymer_type": None,
                "backbone_type": None,
                "mainchain_atoms": [],
                "sidechain_chirality": "NA",
                "termini_variants": [],
            },
            "chemical_modifications": [],
            "connectivity": [],
            "protonation": {
                "protonated_atoms": [],
                "protonation_state": "neutral",
                "pH": 7,
            },
            "virtual": virts,
        },
        "chi_samples": [],
        "default_jump_connection_atom": metal,
        "metal_sites": [
            {
                "metal_atom": metal,
                "geometry": geometry,
                "internal_satisfiers": [],
                "site_virts": virts,
                "site_connections": sites,
            }
        ],
    }


def main():
    with open(os.path.join(CHEM_DIR, "metals.yaml")) as infile:
        table = safe_load(infile)
    vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
    n_sites_for = {
        g["name"]: len(g["vertices"]) or g["max_sites"] for g in table["geometries"]
    }

    residues, charges = [], []
    for ion in table["ions"]:
        distances = fill_distances(ion, table["donor_radii"])
        for geometry in ion["geometries"]:
            res = residue_for(
                ion, geometry, vertices_for[geometry], distances, n_sites_for[geometry]
            )
            residues.append(res)
            # the metal carries no charge: metal_coordination supplies the
            # attraction, so charging it here would count it twice
            charges += [
                {"res": res["name"], "atom": a["name"], "charge": 0.0}
                for a in res["atoms"]
            ]

    with open(os.path.join(CHEM_DIR, "metal_ions.yaml"), "w") as out:
        yaml.safe_dump(
            {"residues": residues}, out, default_flow_style=None, sort_keys=False
        )
    with open(os.path.join(SCORING_DIR, "elec_metal_ions.yaml"), "w") as out:
        yaml.safe_dump(
            {"atom_charge_parameters": charges, "atom_cp_reps_parameters": []},
            out,
            default_flow_style=None,
            sort_keys=False,
        )
    print(f"wrote {len(residues)} metal residue types, {len(charges)} charges")


if __name__ == "__main__":
    main()
