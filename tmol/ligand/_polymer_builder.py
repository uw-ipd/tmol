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


def turnable_bonds(bonds, adj, hydrogens) -> frozenset:
    """Heavy-atom bonds a torsion may turn, as frozensets of atom names.

    Single, out of any ring, and with a heavy atom on either side to move: a
    bond to a terminal atom turns nothing that is not a proton chi.
    """

    def heavy_degree(name):
        return sum(1 for n in adj.get(name, ()) if n not in hydrogens)

    turnable = set()
    for bond in bonds:
        a, b = bond[0], bond[1]
        order = bond[2] if len(bond) > 2 else "SINGLE"
        in_ring = bool(bond[3]) if len(bond) > 3 else False
        if a in hydrogens or b in hydrogens or order != "SINGLE" or in_ring:
            continue
        if heavy_degree(a) > 1 and heavy_degree(b) > 1:
            turnable.add(frozenset((a, b)))
    return frozenset(turnable)


def _branch_is_sidechain(root, anchor, adj, turnable, backbone) -> bool:
    """Whether a branch off the backbone has any bond a chi could turn.

    The backbone's own substituents -- a carbonyl oxygen, a methyl on an amide
    nitrogen -- turn nothing, which is what separates them from a sidechain.
    """
    if frozenset((anchor, root)) in turnable:
        return True
    seen = {anchor, root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, ()):
            if nbr in seen or nbr in backbone:
                continue
            if frozenset((node, nbr)) in turnable:
                return True
            seen.add(nbr)
            queue.append(nbr)
    return False


def sidechain_roots(profile, bonds, adj, kept, hydrogens):
    """Heavy atoms hanging off the sidechain root that are not backbone.

    Structural, so no sidechain atom name is assumed; an alpha-disubstituted
    residue has two and glycine has none. Backbone is not only the mainchain:
    a nucleotide's sugar hangs off it and is backbone all the same, so the
    atoms the profile gives backbone types to are excluded too.

    A backbone the profile does not describe names no such atoms, so there a
    branch is a sidechain only if something along it can be turned.
    """
    mainchain = set(profile.mainchain_atoms) | {n for n, _t in profile.backbone_types}
    # a backbone with no privileged root may carry a sidechain anywhere along it
    declared = profile.sidechain_root_atoms is not None
    roots = profile.sidechain_root_atoms if declared else profile.mainchain_atoms
    turnable = None if declared else turnable_bonds(bonds, adj, hydrogens)
    return [
        n
        for root in roots
        for n in sorted(adj.get(root, ()))
        if n in kept
        and n not in hydrogens
        and n not in mainchain
        and (declared or _branch_is_sidechain(n, root, adj, turnable, mainchain))
    ]


def _backbone_hydrogen_names(profile, adj, hydrogens, taken):
    """Canonical names for hydrogens on the mainchain; sidechain H keep theirs.

    The cartbonded wildcard rows are the terms that span the peptide bond, and
    they key on atom names, so the amide hydrogen has to be called H (H-N-+C,
    and the torsions onto the next residue). No wildcard row names an alpha
    hydrogen, so HA numbering is convention only: it follows the pdb rule that
    a lone hydrogen is unnumbered and a methylene pair on carbon starts at 2.
    """
    renames = {}
    for parent in profile.renamed_h_parents:
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
            walk(root, None)
    return index


def _closes_backbone_ring(b, c, adj, backbone) -> bool:
    """Whether the b-c bond closes a ring made only of backbone atoms.

    A nucleotide's sugar is such a ring: its bonds cannot be turned one at a
    time without tearing it open, and the canonical nucleotides declare no chi
    for them. Proline's ring is not one -- CB, CG and CD are sidechain -- so
    its ring chi are unaffected.
    """
    if b not in backbone or c not in backbone:
        return False
    seen, queue = {b}, deque([b])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, ()):
            if node == b and nbr == c:
                continue
            if nbr == c:
                return True
            if nbr in seen or nbr not in backbone:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return False


def _ring_closure_chi(profile, adj, index, kept, hydrogens):
    """The last chi of a sidechain that closes back onto the mainchain.

    Proline's chi3 turns CG-CD and is measured into N. No tree rooted on the
    mainchain reaches that bond as a parent-child edge -- the ring is entered
    from both ends and it becomes the back edge -- so it is derived here from
    the order the sidechain was walked in.

    Backbone is not only the mainchain: a nucleotide's sugar ring closes onto
    atoms the profile types as backbone, and closing onto backbone is not a
    chi however the ring is entered.
    """
    mainchain = set(profile.mainchain_atoms) | {n for n, _t in profile.backbone_types}

    def previous(atom):
        """The sidechain atom walked immediately before this one."""
        before = [
            n
            for n in adj.get(atom, ())
            if n in index and index[n] < index.get(atom, -1)
        ]
        return max(before, key=lambda n: index[n]) if before else None

    for atom in sorted(index, key=lambda n: index[n], reverse=True):
        if atom in hydrogens or atom not in kept:
            continue
        onto = sorted(n for n in adj.get(atom, ()) if n in mainchain and n in kept)
        if not onto:
            continue
        c = previous(atom)
        if c is None:
            continue
        b = previous(c)
        if b is None:
            continue
        closing = [b, c, atom, onto[0]]
        if len(set(closing)) != 4:
            continue
        return (index[atom], closing, "ring_closure")
    return None


def _chi_torsions(
    profile, restype, adj, kept, hydrogens, roots, rename, elements, borrowed=()
):
    """Sidechain chi, renumbered outward from the root along the longest branch.

    A torsion whose central bond lies wholly on the mainchain is a backbone
    torsion, not a chi. A proton chi (fourth atom a hydrogen) is only emitted
    off an O or S: those keep a lone pair when protonated, where an N-H is
    either a planar amide or a lone-pair-less ammonium.

    ``borrowed`` are the chi a rotamer library defines for this residue, taken
    from the reference it corresponds to. They lead, in the reference's own
    order, and whatever the residue turns beyond them follows. A branch point
    has no local rule to settle it -- hydroxyproline's CG carries both CD and
    OD1 -- so where a library is borrowed it decides.
    """
    mainchain = set(profile.mainchain_atoms) | {n for n, _ in profile.connections}
    backbone = set(profile.mainchain_atoms) | {n for n, _t in profile.backbone_types}
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
        if _closes_backbone_ring(b, c, adj, backbone):
            continue
        # a sidechain leaves the backbone at its root and nowhere else: the
        #    far end of a ring that closes back onto the mainchain is reached
        #    around the ring, not across the bond it closes
        if b in mainchain and c not in roots and c not in mainchain:
            continue
        if c in mainchain and b not in roots and b not in mainchain:
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

    # the ring closure is proline's chi3; a nucleotide's rings are its sugar
    #    and its base, whose torsions the na_torsion term owns
    closing = (
        _ring_closure_chi(profile, adj, index, kept, hydrogens)
        if profile.polymer_type == "amino_acid"
        else None
    )
    if closing is not None:
        # no bond may carry two chi: they would turn it twice, and the second
        #    would undo whatever the first placed
        turned = {frozenset(atoms[1:3]) for _o, atoms, _n in candidates}
        if frozenset(closing[1][1:3]) not in turned:
            candidates.append(closing)

    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    # a borrowed chi turns its bond already; the residue's own reading of that
    #    bond would turn it a second time
    lent = {frozenset(atoms[1:3]) for atoms in borrowed}
    candidates = [c for c in candidates if frozenset(c[1][1:3]) not in lent]

    ordered = [(list(atoms), None) for atoms in borrowed]
    ordered += [(atoms, old_name) for _outer, atoms, old_name in candidates]

    torsions, renumbered, proton = [], {}, set()
    for n, (atoms, old_name) in enumerate(ordered, start=1):
        name = f"chi{n}"
        if old_name is not None:
            renumbered[old_name] = name
        a, b, c, d = (UnresolvedAtom(atom=x) for x in atoms)
        torsions.append(Torsion(name=name, a=a, b=b, c=c, d=d))
        if atoms[3] in hydrogens:
            proton.add(name)
    return torsions, renumbered, proton


def _transplantable(profile, ref_icoors, icoor, kept):
    """Whether the reference residue's own placement of this atom can be used.

    Only where the donor builds the atom against the same three, and in the
    same order. An icoor is a length, an angle and a dihedral measured in one
    frame, so taking it into another is not a change of geometry but a
    different atom position -- and the two trees can be rooted differently, a
    nucleotide's at its 5' oxygen and this one's at the phosphate before it.
    """
    if icoor.name not in profile.transplant_icoors or icoor.name not in ref_icoors:
        return False
    donor = ref_icoors[icoor.name]
    frame = (donor.parent, donor.grand_parent, donor.great_grand_parent)
    if not all(a in kept for a in frame):
        return False
    return frame == (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)


def _glycosidic_torsion(profile, adj, elements, kept, chi, renumbered, proton):
    """The nucleotide's chi, measured the way the torsion tables were built.

    A rotatable-bond search finds the bond to the base but not which four atoms
    to measure it by, and the tables are calibrated on a particular four. The
    chi it found is dropped in favour of the derived one, and anything further
    out renumbered around it.
    """
    from tmol.ligand._polymer_profile import glycosidic_torsion_atoms

    name = profile.glycosidic_torsion
    if name is None:
        return (), chi, renumbered, proton

    real = {a for a in kept if elements.get(a) != "H"}
    heavy = {
        atom: {n for n in neighbours if n in real}
        for atom, neighbours in adj.items()
        if atom in real
    }
    atoms = glycosidic_torsion_atoms(profile, heavy, elements)
    if atoms is None or any(a not in kept for a in atoms):
        return (), chi, renumbered, proton

    bond = {atoms[1], atoms[2]}
    kept_chi = [t for t in chi if {t.b.atom, t.c.atom} != bond]
    dropped = {t.name for t in chi} - {t.name for t in kept_chi}
    # chi1 is the glycosidic torsion, so anything the search found starts at 2
    renamed = {t.name: f"chi{i}" for i, t in enumerate(kept_chi, start=2)}
    kept_chi = [
        Torsion(name=renamed[t.name], a=t.a, b=t.b, c=t.c, d=t.d) for t in kept_chi
    ]
    renumbered = {
        old: renamed[new] for old, new in renumbered.items() if new not in dropped
    }
    proton = {renamed[n] for n in proton if n not in dropped}
    a, b, c, d = (UnresolvedAtom(atom=x) for x in atoms)
    return (Torsion(name=name, a=a, b=b, c=c, d=d),), kept_chi, renumbered, proton


def icoor_mainchain(profile, present):
    """The mainchain in the order the icoor tree walks it.

    Rooted where the profile says, which is where the canonical residue roots
    its own. That is not idle: everything is placed against the root, so a
    patch that removes it orphans the residue. A nucleotide roots at its 5'
    oxygen rather than at the phosphate before it, which is what lets the 5'
    terminus patch take the phosphate away.
    """
    mainchain = [a for a in profile.mainchain_atoms if a in present]
    root = profile.icoor_root
    if root is None or root not in mainchain:
        return mainchain
    start = mainchain.index(root)
    # what preceded the root comes after it, still bonded to what is placed
    return mainchain[start:] + mainchain[start - 1 :: -1] if start else mainchain


def _icoor_order(profile, bonds, adj, kept, hydrogens):
    """Traversal order: mainchain, up, O, sidechain, its H, CA-H, down, N-H.

    Mirrors the canonical residues' ordering so parents always precede children.
    """
    down_name, down_atom = profile.down if profile.down else (None, None)
    up_name, _up_atom = profile.up if profile.up else (None, None)
    mainchain = icoor_mainchain(profile, kept)
    roots = sidechain_roots(profile, bonds, adj, kept, hydrogens)

    placed = list(mainchain)
    order = list(mainchain) + ([up_name] if up_name else [])
    # the rest of the backbone: what hangs off the mainchain, and then onward
    #    through backbone atoms, since a nucleotide's sugar reaches further
    #    from the mainchain than a carbonyl oxygen does
    backbone = set(mainchain) | {n for n, _t in profile.backbone_types}
    queue = deque(mainchain)
    while queue:
        current = queue.popleft()
        for nbr in sorted(adj.get(current, ())):
            if nbr not in kept or nbr in placed or nbr in hydrogens or nbr in roots:
                continue
            if current not in backbone:
                continue
            placed.append(nbr)
            order.append(nbr)
            if nbr in backbone:
                queue.append(nbr)

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


def _parents(profile, adj, order, hydrogens=frozenset()):
    """(parent, grand_parent, great_grand_parent) for each entry in order."""
    mainchain = icoor_mainchain(profile, set(order))
    position = {name: i for i, name in enumerate(order)}

    # the connection pseudo-atoms hang off the terminal mainchain atoms
    parent = {name: atom for name, atom in profile.connections}
    # the tree's root is its own parent; every other atom hangs off the nearest
    #    already-placed atom it is bonded to, the rest of the mainchain included
    parent[mainchain[0]] = mainchain[0]
    for name in order:
        if name in parent:
            continue
        placed = [
            n
            for n in adj.get(name, ())
            if n in position and position[n] < position[name]
        ]
        parent[name] = min(placed, key=lambda n: position[n])

    def placed_sibling(name, par, exclude, heavy_only=False):
        for n in sorted(adj.get(par, ()), key=lambda x: position.get(x, len(order))):
            if n in exclude or n not in position:
                continue
            if heavy_only and n in hydrogens:
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
        # A hydrogen is measured against a heavy atom on its own parent rather
        #    than up the chain: the torsion it would otherwise use is free to
        #    differ between the conformer the icoors were measured on and the
        #    structure the pose is built from, and the hydrogen would not
        #    follow its parent's other substituents.
        if name in hydrogens:
            ggp = placed_sibling(name, par, {name, gp}, heavy_only=True) or ggp
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


def _carboxyl_neighbor(center, adj, elements):
    """A carbon bonded to center that carries a terminal oxygen, or None.

    Bond order cannot be used: a carboxylate records both its C-O bonds
    single. Two such carbons are ambiguous and give None, so the caller keeps
    its mainchain-order frame rather than guessing.
    """

    def heavy(name):
        return [n for n in adj.get(name, ()) if elements.get(n) not in (None, "H")]

    found = [
        nbr
        for nbr in adj.get(center, ())
        if elements.get(nbr) == "C"
        and any(elements.get(o) == "O" and len(heavy(o)) == 1 for o in adj.get(nbr, ()))
    ]
    return found[0] if len(found) == 1 else None


def sidechain_chirality(profile, coords, roots, adj, elements):
    """l / d / achiral from the signed volume at the sidechain-bearing atom.

    CIP cannot be used: L-cysteine is (R) where every other L-aa is (S).
    Zero sidechain branches (glycine) or two (alpha-disubstituted) are achiral.
    Only an amino acid has one: a nucleotide's sidechain hangs off a mainchain
    that carries no stereocenter, so the volume is a conformer artifact. Every
    other polymer takes the not-applicable sentinel the canonical ones use.

    The mainchain atom the sidechain hangs off is the stereocenter. Its frame
    is amine, carboxyl, then the remaining substituent -- picked by chemistry,
    because a chain that leaves through a side branch (a gamma linkage, say)
    puts the carboxyl on the branch and the mainchain on the R group, and a
    frame taken in mainchain order would mirror the label. A backbone whose
    stereocenter carries no carboxyl falls back to its mainchain neighbours.
    """
    if profile.polymer_type != "amino_acid":
        return "NA"
    mainchain = tuple(profile.mainchain_atoms)
    if len(roots) != 1 or len(mainchain) < 3:
        return "achiral"
    root = roots[0]
    carries = [i for i, a in enumerate(mainchain) if root in adj.get(a, ())]
    if len(carries) != 1 or not 0 < carries[0] < len(mainchain) - 1:
        return "achiral"
    index = carries[0]
    stereocenter = mainchain[index]
    amine = mainchain[index - 1]
    carboxyl = _carboxyl_neighbor(stereocenter, adj, elements)
    if carboxyl is None:
        forward, tip = mainchain[index + 1], root
    else:
        rest = [
            n
            for n in adj.get(stereocenter, ())
            if n not in (amine, carboxyl) and elements.get(n) not in (None, "H")
        ]
        if len(rest) != 1:
            return "achiral"
        forward, tip = carboxyl, rest[0]
    center = coords[stereocenter]
    volume = numpy.dot(
        numpy.cross(coords[amine] - center, coords[forward] - center),
        coords[tip] - center,
    )
    return "l" if volume > 0 else "d"


def _borrowed_chi(
    name, atoms, bonds, coords, profile, chirality, references, atom_type_index
):
    """The rotamer library this residue takes its chi from, and those chi.

    Decided here rather than after the fact: which torsions the residue calls
    chi1, chi2 and so on is exactly what corresponding to a library means.
    """
    from tmol.ligand._rotamer_reference import (
        SidechainGraph,
        planar_atoms,
        reference_chi_for_graph,
    )

    if not references or atom_type_index is None:
        return None, ()
    if profile.polymer_type != "amino_acid":
        return None, ()
    described = {a.name: atom_type_index.get(a.atom_type) for a in atoms}
    if any(record is None for record in described.values()):
        return None, ()

    element = {n: record.element for n, record in described.items()}
    heavy = {n for n, e in element.items() if e.upper() != "H"}
    adjacency = {n: set() for n in element}
    rigid = set()
    for bond in bonds:
        first, second = bond[0], bond[1]
        if first not in adjacency or second not in adjacency:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
        order = bond[2] if len(bond) > 2 else "SINGLE"
        if order != "SINGLE" and first in heavy and second in heavy:
            rigid.add(frozenset((first, second)))

    graph = SidechainGraph(
        name=name,
        element=element,
        donor=frozenset(n for n, r in described.items() if r.is_donor),
        acceptor=frozenset(n for n, r in described.items() if r.is_acceptor),
        adjacency={
            n: frozenset(x for x in nbrs if x in heavy)
            for n, nbrs in adjacency.items()
            if n in heavy
        },
        rigid=frozenset(rigid),
        planar=planar_atoms(adjacency, coords),
        mainchain=tuple(profile.mainchain_atoms),
        backbone_type=profile.backbone_type,
        chirality=chirality,
        coords=coords,
    )
    return reference_chi_for_graph(graph, references)


def to_polymer_residue_type(
    restype: RawResidueType,
    coords: dict,
    profile: PolymerProfile,
    reference: RawResidueType,
    elements: dict,
    cap_names: dict,
    rotamer_references=(),
    atom_type_index=None,
) -> RawResidueType:
    """Replace a capped molecule's stubs with polymer connections.

    coords maps atom name to the generated conformer position; elements maps it
    to its element symbol. rotamer_references are the canonical libraries this
    residue may take its chi from, which is decided here because it decides
    what the chi are.
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

    roots = sidechain_roots(profile, restype.bonds, adj, kept, hydrogens)
    chirality = sidechain_chirality(profile, coords, roots, adj, elements)
    dunbrack_reference, borrowed = _borrowed_chi(
        restype.name,
        atoms,
        bonds,
        coords,
        profile,
        chirality,
        rotamer_references,
        atom_type_index,
    )
    order = _icoor_order(profile, restype.bonds, adj, kept, hydrogens)
    frames = _parents(profile, adj, order, hydrogens)
    icoors = _computed_icoors(order, frames, coords)

    # the mainchain records are shared by every residue of this backbone; take
    #    them from the reference so the backbone sits on database geometry
    ref_icoors = {ic.name: ic for ic in reference.icoors} if reference else {}
    icoors = tuple(
        (ref_icoors[ic.name] if _transplantable(profile, ref_icoors, ic, kept) else ic)
        for ic in icoors
    )

    chi, renumbered, proton = _chi_torsions(
        profile, restype, adj, kept, hydrogens, roots, rename, elements, borrowed
    )
    glycosidic, chi, renumbered, proton = _glycosidic_torsion(
        profile, adj, elements, kept, chi, renumbered, proton
    )
    torsions = _mainchain_torsions(profile, kept) + list(glycosidic) + chi
    # proton chi are optH's as well as the packer's; a heavy chi is the
    #    packer's alone. Which is which follows the renumbered chi, not the
    #    capped molecule's, since renumbering can change what a chi measures.
    chi_samples = [
        type(s)(
            chi_dihedral=renumbered[s.chi_dihedral],
            samples=s.samples,
            expansions=s.expansions,
            is_proton=renumbered[s.chi_dihedral] in proton,
        )
        for s in restype.chi_samples
        if s.chi_dihedral in renumbered
    ]

    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=True,
            polymer_type=profile.polymer_type,
            backbone_type=profile.backbone_type,
            mainchain_atoms=tuple(profile.mainchain_atoms),
            sidechain_chirality=chirality,
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
        # the atom a jump anchors on: the second mainchain atom for a backbone
        #    that has one, the only atom for a cap that does not
        default_jump_connection_atom=profile.mainchain_atoms[
            1 if len(profile.mainchain_atoms) > 1 else 0
        ],
        hydrogens_regenerated=restype.hydrogens_regenerated,
        dunbrack_reference=dunbrack_reference,
    )
