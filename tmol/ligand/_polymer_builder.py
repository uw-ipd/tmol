"""Turn a prepared capped molecule into a polymer residue type.

The ligand pipeline types and charges a capped molecule; this replaces the caps
with polymer connections, re-roots the atom tree on the first mainchain atom,
and declares the polymer properties the score terms read.
"""

import math
from collections import deque

import numpy

from tmol.database.chemical import (
    ChemicalProperties,
    Connection,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
    Torsion,
    UnresolvedAtom,
)
from tmol.ligand._polymer_profile import PolymerProfile
from tmol.ligand._residue_builder import _angle, _dihedral, _distance


def _adjacency(bonds):
    adj = {}
    for bond in bonds:
        adj.setdefault(bond[0], []).append(bond[1])
        adj.setdefault(bond[1], []).append(bond[0])
    return adj


def _dropped_atoms(cap_names, adj, hydrogens):
    """Cap atoms plus the hydrogens hanging off them."""
    dropped = set(cap_names.values())
    for cap in list(dropped):
        for nbr in adj.get(cap, ()):
            if nbr in hydrogens:
                dropped.add(nbr)
    return dropped


def sidechain_roots(profile, adj, kept, hydrogens):
    """Heavy atoms hanging off the sidechain root that are not on the mainchain.

    Structural, so no sidechain atom name is assumed; an alpha-disubstituted
    residue has two and glycine has none.
    """
    mainchain = set(profile.mainchain_atoms)
    return [
        n
        for n in sorted(adj.get(profile.sidechain_root_atom, ()))
        if n in kept and n not in hydrogens and n not in mainchain
    ]


def _backbone_hydrogen_names(profile, adj, hydrogens, taken):
    """Canonical names for hydrogens on the mainchain; sidechain H keep theirs.

    The cartbonded wildcard terms key on these names (H-N-+C), so the backbone
    hydrogens have to carry them. Numbering follows the pdb convention: a lone
    hydrogen is unnumbered, a methylene pair on carbon starts at 2.
    """
    renames = {}
    for parent, _type in profile.backbone_h_types:
        attached = sorted(n for n in adj.get(parent, ()) if n in hydrogens)
        if not attached:
            continue
        suffix = parent[1:]
        if len(attached) == 1:
            names = [f"H{suffix}"]
        else:
            start = 2 if parent[0] == "C" and len(attached) == 2 else 1
            names = [f"H{suffix}{start + i}" for i in range(len(attached))]
        clash = sorted(set(names) & (taken - set(attached)))
        if clash:
            raise ValueError(f"backbone hydrogen name(s) {clash} already in use")
        renames.update(zip(attached, names))
    return renames


def backbone_renames(restype, profile, elements, cap_names):
    """The atom renames to_polymer_residue_type applies, and the atoms it drops.

    Callers that carry per-atom data alongside the residue (charges, coords)
    need the same mapping.
    """
    hydrogens = {n for n, e in elements.items() if e == "H"}
    adj = _adjacency(restype.bonds)
    dropped = _dropped_atoms(cap_names, adj, hydrogens)
    kept = {a.name for a in restype.atoms if a.name not in dropped}
    remaining = {h for h in hydrogens if h not in dropped}
    return _backbone_hydrogen_names(profile, adj, remaining, kept), dropped


def _backbone_atom_types(profile, adj, hydrogens, current):
    """Protein atom types for the backbone; the sidechain keeps ligand types."""
    types = dict(profile.backbone_types)
    if profile.down is not None and profile.amide_n_types is not None:
        n_atom = profile.down[1]
        with_h, substituted = profile.amide_n_types
        has_h = any(x in hydrogens for x in adj.get(n_atom, ()))
        types[n_atom] = with_h if has_h else substituted
    for parent, h_type in profile.backbone_h_types:
        for nbr in adj.get(parent, ()):
            if nbr in hydrogens:
                types[nbr] = h_type
    return {name: t for name, t in types.items() if name in current}


def _unresolved(spec):
    """An UnresolvedAtom from "NAME" or "conn:bond_separation"."""
    if ":" in spec:
        conn, separation = spec.split(":")
        return UnresolvedAtom(connection=conn, bond_sep_from_conn=int(separation))
    return UnresolvedAtom(atom=spec)


def _mainchain_torsions(profile, kept):
    """phi / psi / omega, skipping any the residue lacks the atoms for."""
    out = []
    for name, specs in profile.mainchain_torsions:
        if any(":" not in s and s not in kept for s in specs):
            continue
        a, b, c, d = (_unresolved(s) for s in specs)
        out.append(Torsion(name=name, a=a, b=b, c=c, d=d))
    return out


def _sidechain_traversal(profile, adj, roots, kept, hydrogens):
    """Visit order for sidechain heavy atoms, deepest branch first."""
    blocked = set(profile.mainchain_atoms)

    def children(node, prev, seen):
        return [
            n
            for n in sorted(adj.get(node, ()))
            if n != prev
            and n in kept
            and n not in hydrogens
            and n not in blocked
            and n not in seen
        ]

    def depth(node, prev, seen):
        seen = seen | {node}
        return 1 + max(
            (depth(c, node, seen) for c in children(node, prev, seen)), default=0
        )

    index = {}

    def walk(node, prev):
        index[node] = len(index)
        for child in sorted(
            children(node, prev, set(index)),
            key=lambda n: (-depth(n, node, {node}), n),
        ):
            if child not in index:
                walk(child, node)

    for root in roots:
        if root not in index:
            walk(root, profile.sidechain_root_atom)
    return index


def _chi_torsions(profile, restype, adj, kept, hydrogens, roots, rename, elements):
    """Sidechain chi, renumbered outward from the root along the longest branch.

    A torsion whose central bond lies wholly on the mainchain is a backbone
    torsion, not a chi. A proton chi (fourth atom a hydrogen) is only emitted
    off an O or S: those keep a lone pair when protonated, where an N-H is
    either a planar amide or a lone-pair-less ammonium.
    """
    mainchain = set(profile.mainchain_atoms) | {n for n, _ in profile.connections}
    index = _sidechain_traversal(profile, adj, roots, kept, hydrogens)

    candidates = []
    for torsion in restype.torsions:
        atoms = [torsion.a.atom, torsion.b.atom, torsion.c.atom, torsion.d.atom]
        if any(a is None for a in atoms):
            continue
        atoms = [rename(a) for a in atoms]
        if any(a not in kept for a in atoms):
            continue
        b, c = atoms[1], atoms[2]
        if b in mainchain and c in mainchain:
            continue
        # orient outward from the backbone: dunbrack reads the chi-defining
        #    atom off position 2, so a reversed chi rotates the wrong bond
        if index.get(b, -1) > index.get(c, -1):
            atoms = atoms[::-1]
            b, c = atoms[1], atoms[2]
        if atoms[3] in hydrogens and elements.get(atoms[2]) not in ("O", "S"):
            continue
        outer = max((index.get(b, -1), index.get(c, -1)))
        if outer < 0:
            continue
        candidates.append((outer, atoms, torsion.name))

    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    torsions, renumbered = [], {}
    for n, (_outer, atoms, old_name) in enumerate(candidates, start=1):
        name = f"chi{n}"
        renumbered[old_name] = name
        a, b, c, d = (UnresolvedAtom(atom=x) for x in atoms)
        torsions.append(Torsion(name=name, a=a, b=b, c=c, d=d))
    proton = {
        f"chi{n}"
        for n, (_o, atoms, _name) in enumerate(candidates, start=1)
        if atoms[3] in hydrogens
    }
    return torsions, renumbered, proton


def _icoor_order(profile, adj, kept, hydrogens):
    """Traversal order: mainchain, up, O, sidechain, its H, CA-H, down, N-H.

    Mirrors the canonical residues' ordering so parents always precede children.
    """
    down_name, down_atom = profile.down if profile.down else (None, None)
    up_name, _up_atom = profile.up if profile.up else (None, None)
    mainchain = [a for a in profile.mainchain_atoms if a in kept]
    roots = sidechain_roots(profile, adj, kept, hydrogens)

    placed = list(mainchain)
    order = list(mainchain) + ([up_name] if up_name else [])
    # the carbonyl oxygen and anything else hanging off the mainchain heavy atoms
    bb_leaves = []
    for mc in mainchain:
        for nbr in sorted(adj.get(mc, ())):
            if nbr in kept and nbr not in placed and nbr not in hydrogens:
                if nbr in roots:
                    continue
                bb_leaves.append(nbr)
                placed.append(nbr)
    order.extend(bb_leaves)

    # sidechain heavy atoms, breadth first from each root
    sidechain = []
    queue = deque(roots)
    seen = set(placed) | set(roots)
    while queue:
        cur = queue.popleft()
        sidechain.append(cur)
        for nbr in sorted(adj.get(cur, ())):
            if nbr in kept and nbr not in seen and nbr not in hydrogens:
                seen.add(nbr)
                queue.append(nbr)
    order.extend(sidechain)

    # hydrogens follow their heavy atom's block
    def hydrogens_on(names):
        out = []
        for heavy in names:
            for nbr in sorted(adj.get(heavy, ())):
                if nbr in hydrogens and nbr in kept and nbr not in out:
                    out.append(nbr)
        return out

    # every heavy atom's hydrogens, not just the sidechain's: a ring closing on
    #    the mainchain (proline's CD) is placed as a backbone leaf
    heavy = [a for a in order if a in kept and a not in hydrogens]
    order.extend(hydrogens_on([a for a in heavy if a not in mainchain]))
    order.extend(hydrogens_on([a for a in mainchain if a != down_atom]))
    if down_name:
        order.append(down_name)
        order.extend(hydrogens_on([down_atom]))

    missing = kept - set(order)
    if missing:
        raise ValueError(f"atoms left out of the icoor order: {sorted(missing)}")
    return order


def _parents(profile, adj, order):
    """(parent, grand_parent, great_grand_parent) for each entry in order."""
    mainchain = [a for a in profile.mainchain_atoms if a in set(order)]
    position = {name: i for i, name in enumerate(order)}

    # the connection pseudo-atoms hang off the terminal mainchain atoms
    parent = {name: atom for name, atom in profile.connections}
    for i, name in enumerate(mainchain):
        parent[name] = mainchain[max(i - 1, 0)]
    for name in order:
        if name in parent:
            continue
        placed = [
            n
            for n in adj.get(name, ())
            if n in position and position[n] < position[name]
        ]
        parent[name] = min(placed, key=lambda n: position[n])

    def placed_sibling(name, par, exclude):
        for n in sorted(adj.get(par, ()), key=lambda x: position.get(x, len(order))):
            if n in exclude or n not in position:
                continue
            if position[n] < position[name]:
                return n
        return None

    result = {}
    for name in order:
        par = parent[name]
        if name == mainchain[0]:
            # root names the next two mainchain atoms as its frame
            gp = mainchain[1] if len(mainchain) > 1 else par
            ggp = mainchain[2] if len(mainchain) > 2 else gp
            result[name] = (par, gp, ggp)
            continue
        gp = parent.get(par, par)
        if gp == par or gp == name:
            gp = placed_sibling(name, par, {name, par}) or gp
        ggp = parent.get(gp, gp)
        if ggp in (name, par, gp):
            # a ring closing onto the root (proline's CD) has no usable sibling
            #    of the parent; step out to the grandparent's neighbours
            ggp = (
                placed_sibling(name, par, {name, par, gp})
                or placed_sibling(name, gp, {name, par, gp})
                or ggp
            )
        result[name] = (par, gp, ggp)
    return result


def _computed_icoors(order, frames, coords):
    """Internal coordinates measured off the generated conformer."""
    icoors = []
    for i, name in enumerate(order):
        par, gp, ggp = frames[name]
        if i == 0:
            d, theta, phi = 0.0, 0.0, 0.0
        elif i == 1:
            d, theta, phi = _distance(coords[name], coords[par]), 180.0, 0.0
        elif i == 2:
            d = _distance(coords[name], coords[par])
            theta = 180.0 - _angle(coords[name], coords[par], coords[gp])
            phi = 0.0
        else:
            d = _distance(coords[name], coords[par])
            theta = 180.0 - _angle(coords[name], coords[par], coords[gp])
            phi = -_dihedral(coords[name], coords[par], coords[gp], coords[ggp])
        icoors.append(
            Icoor(
                name=name,
                phi=math.radians(phi),
                theta=math.radians(theta),
                d=d,
                parent=par,
                grand_parent=gp,
                great_grand_parent=ggp,
            )
        )
    return icoors


def sidechain_chirality(profile, coords, roots):
    """l / d / achiral from the signed volume at the alpha carbon.

    CIP cannot be used: L-cysteine is (R) where every other L-aa is (S).
    Zero sidechain branches (glycine) or two (alpha-disubstituted) are achiral.
    """
    if len(roots) != 1:
        return "achiral"
    n, ca, c = (coords[a] for a in profile.mainchain_atoms[:3])
    volume = numpy.dot(numpy.cross(n - ca, c - ca), coords[roots[0]] - ca)
    return "l" if volume > 0 else "d"


def to_polymer_residue_type(
    restype: RawResidueType,
    coords: dict,
    profile: PolymerProfile,
    reference: RawResidueType,
    elements: dict,
    cap_names: dict,
) -> RawResidueType:
    """Replace a capped molecule's stubs with polymer connections.

    coords maps atom name to the generated conformer position; elements maps it
    to its element symbol.
    """
    hydrogen_names = {n for n, e in elements.items() if e == "H"}
    adj = _adjacency(restype.bonds)
    dropped = _dropped_atoms(cap_names, adj, hydrogen_names)

    kept = {a.name for a in restype.atoms if a.name not in dropped}
    hydrogens = {h for h in hydrogen_names if h not in dropped}
    renames = _backbone_hydrogen_names(profile, adj, hydrogens, kept)

    def rename(name):
        return renames.get(name, name)

    # the stub atoms mark where the neighbouring residues attach
    placed = {name: coords[cap_names[cap]] for name, cap in profile.connection_partners}
    coords = {rename(k): v for k, v in coords.items() if k not in dropped}
    coords.update(placed)
    kept = {rename(n) for n in kept}
    hydrogens = {rename(n) for n in hydrogens}
    adj = {
        rename(k): [rename(x) for x in v] for k, v in adj.items() if k not in dropped
    }

    # only the backbone is renamed and retyped; the sidechain keeps both
    retyped = _backbone_atom_types(profile, adj, hydrogens, kept)
    atoms = tuple(
        type(a)(
            name=rename(a.name),
            atom_type=retyped.get(rename(a.name), a.atom_type),
        )
        for a in restype.atoms
        if a.name not in dropped
    )
    bonds = tuple(
        (rename(b[0]), rename(b[1]), *b[2:])
        for b in restype.bonds
        if b[0] not in dropped and b[1] not in dropped
    )
    connections = tuple(
        Connection(name=name, atom=atom, type=profile.connection_bond_type)
        for name, atom in profile.connections
    )

    adj = _adjacency(bonds)
    # connections are not bonds, but icoor frames reference them (H's ggp is down)
    for conn_name, conn_atom in profile.connections:
        adj.setdefault(conn_atom, []).append(conn_name)
        adj[conn_name] = [conn_atom]

    roots = sidechain_roots(profile, adj, kept, hydrogens)
    order = _icoor_order(profile, adj, kept, hydrogens)
    frames = _parents(profile, adj, order)
    icoors = _computed_icoors(order, frames, coords)

    # the mainchain records are shared by every residue of this backbone; take
    #    them from the reference so the backbone sits on database geometry
    ref_icoors = {ic.name: ic for ic in reference.icoors} if reference else {}
    icoors = tuple(
        (
            ref_icoors[ic.name]
            if ic.name in profile.transplant_icoors and ic.name in ref_icoors
            else ic
        )
        for ic in icoors
    )

    chi, renumbered, proton = _chi_torsions(
        profile, restype, adj, kept, hydrogens, roots, rename, elements
    )
    torsions = _mainchain_torsions(profile, kept) + chi
    # proton chi keep their samples for optH; heavy chi are packed as rotamers
    chi_samples = [
        type(s)(
            chi_dihedral=renumbered[s.chi_dihedral],
            samples=s.samples,
            expansions=s.expansions,
        )
        for s in restype.chi_samples
        if s.chi_dihedral in renumbered and renumbered[s.chi_dihedral] in proton
    ]

    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=True,
            polymer_type=profile.polymer_type,
            backbone_type=profile.backbone_type,
            mainchain_atoms=tuple(profile.mainchain_atoms),
            sidechain_chirality=sidechain_chirality(profile, coords, roots),
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )

    return RawResidueType(
        name=restype.name,
        base_name=restype.base_name,
        name3=restype.name3,
        io_equiv_class=restype.io_equiv_class,
        atoms=atoms,
        atom_aliases=restype.atom_aliases,
        bonds=bonds,
        connections=connections,
        torsions=tuple(torsions),
        icoors=icoors,
        properties=properties,
        chi_samples=tuple(chi_samples),
        default_jump_connection_atom=profile.mainchain_atoms[1],
        hydrogens_regenerated=restype.hydrogens_regenerated,
    )
