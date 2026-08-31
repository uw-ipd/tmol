"""Declarative description of each polymer backbone the NCAA path supports.

A noncanonical polymer residue is prepared as a molecule: its polymer connections
are replaced by chemical stubs (caps), the ligand pipeline types and charges the
capped molecule, then the caps are stripped back to connections. A profile says
which atoms form the backbone, what the caps are, and what the resulting residue
type must declare.
"""

import math
from collections import Counter, defaultdict, deque
from typing import Optional, Tuple

import attr
import numpy
import biotite.structure as struc

from tmol.ligand._detect import ccd_chain_end_atoms


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CapAtom:
    """One stub atom, placed by internal coordinates against three placed atoms."""

    name: str
    element: str
    # frame for place_atom(a, b, c, ...); a residue with only two heavy atoms
    #    cannot supply three, so a two-name frame gets a synthetic first point
    refs: Tuple[str, ...]
    d: float
    angle: float
    dihedral: float
    bond_to: str
    bond_order: str = "SINGLE"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerProfile:
    """How to cap, build and declare one class of polymer residue."""

    name: str
    polymer_type: str
    backbone_type: str
    # geometry/type donor for the backbone transplant
    reference_restype: Optional[str]
    mainchain_atoms: Tuple[str, ...]
    # (connection name, atom it attaches to); None where the residue does not
    #    continue the chain, as a terminal cap does not
    down: Optional[Tuple[str, str]]
    up: Optional[Tuple[str, str]]
    connection_bond_type: str
    caps: Tuple[CapAtom, ...]
    # cap atom standing in for the neighbour across each connection
    down_partner: Optional[str]
    up_partner: Optional[str]
    # mainchain atoms whose non-mainchain heavy neighbours root a sidechain;
    #    identified structurally, so no sidechain atom name is ever assumed.
    #    Empty where the backbone is the whole residue, None where a sidechain
    #    may hang anywhere along the mainchain
    sidechain_root_atoms: Optional[Tuple[str, ...]]
    # backbone atoms are retyped to protein types; sidechain keeps ligand types
    backbone_types: Tuple[Tuple[str, str], ...]
    # mainchain amide N, (carrying a hydrogen, substituted)
    amide_n_types: Tuple[str, str]
    # (parent mainchain atom, type) for the hydrogens on the backbone
    backbone_h_types: Tuple[Tuple[str, str], ...]
    # named backbone torsions; "conn:n" is the atom n bonds past a connection
    mainchain_torsions: Tuple[Tuple[str, Tuple[str, str, str, str]], ...]
    # icoors taken verbatim from reference_restype; these records are identical
    #    across every canonical residue, so transplanting them keeps the
    #    backbone on database geometry instead of the generated conformer's
    transplant_icoors: Tuple[str, ...]
    # mainchain atoms whose hydrogens must carry canonical names. Only the
    #    amide hydrogen qualifies: cartbonded's wildcard rows reach across the
    #    peptide bond and name it. No wildcard row names an alpha hydrogen, so
    #    the rest keep whatever the ligand pipeline gave them
    renamed_h_parents: Tuple[str, ...] = ()

    @property
    def cap_names(self):
        return tuple(c.name for c in self.caps)

    @property
    def connections(self):
        """(name, atom) of each polymer connection the backbone actually has."""
        return tuple(c for c in (self.down, self.up) if c is not None)

    @property
    def connection_partners(self):
        """(connection name, cap atom standing in for the neighbour) per connection."""
        return tuple(
            (conn[0], partner)
            for conn, partner in (
                (self.down, self.down_partner),
                (self.up, self.up_partner),
            )
            if conn is not None
        )

    def required_atoms(self):
        """Backbone atoms an input residue must carry to match this profile."""
        return frozenset(self.mainchain_atoms) | {a for _, a in self.connections}


def _canonical_alpha_residues(chemdb):
    """The database's own alpha amino acids, which the profile is read from."""
    return [
        r
        for r in chemdb.residues
        if r.name == r.base_name
        and r.properties.is_canonical
        and r.properties.polymer.is_polymer
        and r.properties.polymer.backbone_type == "alpha"
    ]


def _commonest(counter):
    return counter.most_common(1)[0][0] if counter else None


def _stub_from_icoor(
    name, element, refs, icoor, dihedral, bond_to, order="SINGLE", distance=None
):
    """A cap stub placed on a neighbouring residue's own internal coordinates.

    The stubs stand in for the residues across the peptide bonds, so their
    geometry is that backbone's: an icoor gives the angle as the supplement of
    its theta, and its length unless one is taken from elsewhere. An alpha
    carbon's own icoor holds the length but not a usable angle, being the
    placement of the atom tree's root. The dihedral is the stub's placement
    rather than the neighbour's, and keeps the amide planar.
    """
    return CapAtom(
        name,
        element,
        refs,
        icoor.d if distance is None else distance,
        180.0 - math.degrees(icoor.theta),
        dihedral,
        bond_to,
        order,
    )


_ALPHA_PROFILE_CACHE: dict = {}


def alpha_profile(chemdb) -> PolymerProfile:
    """The alpha-amino-acid profile, read off the database's own residues.

    Nothing about an alanine backbone is written here: the mainchain, the
    connections, the atom types, the hydrogen types and the peptide-bond
    geometry are all surveyed from the canonical residues that already carry
    them, so a change to the database reaches this without an edit.
    """
    key = id(chemdb)
    if key not in _ALPHA_PROFILE_CACHE:
        _ALPHA_PROFILE_CACHE[key] = _build_alpha_profile(chemdb)
    return _ALPHA_PROFILE_CACHE[key]


def _build_alpha_profile(chemdb) -> PolymerProfile:  # noqa: C901
    residues = _canonical_alpha_residues(chemdb)
    if not residues:
        raise ValueError("the chemical database defines no alpha amino acids")
    element = {a.name: a.element for a in chemdb.atom_types}

    mainchain = _commonest(
        Counter(r.properties.polymer.mainchain_atoms for r in residues)
    )
    connections = _commonest(
        Counter(
            tuple(
                (c.name, c.atom, c.type) for c in r.connections if c.atom in mainchain
            )
            for r in residues
        )
    )
    by_name = {name: (atom, kind) for name, atom, kind in connections}
    down_atom, bond_type = by_name["down"]
    up_atom, _ = by_name["up"]

    # atom types, and the hydrogen types, as the canonical residues assign them
    types = defaultdict(Counter)
    hydrogen_types = defaultdict(Counter)
    for residue in residues:
        atom_types = {a.name: a.atom_type for a in residue.atoms}
        for name, atom_type in atom_types.items():
            types[name][atom_type] += 1
        for bond in residue.bonds:
            for atom, other in (bond[:2], bond[1::-1]):
                if atom in mainchain and element.get(atom_types.get(other, "")) == "H":
                    hydrogen_types[atom][atom_types[other]] += 1

    # the carbonyl oxygen: the one atom off the mainchain every residue shares
    off_mainchain = Counter()
    for residue in residues:
        adjacency = defaultdict(set)
        for a, b, *_ in residue.bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
        for atom in mainchain:
            for other in adjacency[atom]:
                if (
                    other not in mainchain
                    and element.get(
                        {x.name: x.atom_type for x in residue.atoms}.get(other, "")
                    )
                    == "O"
                ):
                    off_mainchain[other] += 1
    carbonyl = _commonest(off_mainchain)

    # the mainchain atom a sidechain hangs off: the one whose non-mainchain
    #    heavy neighbour is not that oxygen
    roots = Counter()
    for residue in residues:
        atom_types = {a.name: a.atom_type for a in residue.atoms}
        adjacency = defaultdict(set)
        for a, b, *_ in residue.bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
        for atom in mainchain:
            for other in adjacency[atom]:
                if (
                    other not in mainchain
                    and other != carbonyl
                    and element.get(atom_types.get(other, "")) != "H"
                ):
                    roots[atom] += 1
    sidechain_root = _commonest(roots)

    # a nitrogen already substituted is typed differently from one carrying a
    #    hydrogen, which is what tells proline from the rest
    n_types = types[down_atom].most_common()
    amide_n = (n_types[0][0], n_types[-1][0])

    icoors = {
        name: {i.name: i for i in r.icoors}
        for name, r in ((r.name, r) for r in residues)
    }
    shared = {}
    for name in (down_atom, up_atom, carbonyl, "down", "up", *mainchain):
        seen = Counter(
            (i[name].d, round(i[name].theta, 6)) for i in icoors.values() if name in i
        )
        if seen:
            shared[name] = next(
                i[name]
                for i in icoors.values()
                if name in i
                and (i[name].d, round(i[name].theta, 6)) == _commonest(seen)
            )

    return PolymerProfile(
        name="alpha",
        polymer_type="amino_acid",
        backbone_type="alpha",
        reference_restype=residues[0].name,
        mainchain_atoms=tuple(mainchain),
        down=("down", down_atom),
        up=("up", up_atom),
        connection_bond_type=bond_type,
        caps=(
            # the residue before, standing in across the down connection
            _stub_from_icoor(
                "CY",
                "C",
                (up_atom, sidechain_root, down_atom),
                shared["down"],
                180.0,
                down_atom,
            ),
            _stub_from_icoor(
                "OY",
                "O",
                (sidechain_root, down_atom, "CY"),
                shared[carbonyl],
                0.0,
                "CY",
                "DOUBLE",
            ),
            _stub_from_icoor(
                "CAY",
                "C",
                (sidechain_root, down_atom, "CY"),
                shared["up"],
                180.0,
                "CY",
                distance=shared[sidechain_root].d,
            ),
            # and the residue after, across the up connection
            _stub_from_icoor(
                "NM",
                "N",
                (down_atom, sidechain_root, up_atom),
                shared["up"],
                180.0,
                up_atom,
            ),
            _stub_from_icoor(
                "CM",
                "C",
                (sidechain_root, up_atom, "NM"),
                shared["down"],
                180.0,
                "NM",
                distance=shared[sidechain_root].d,
            ),
        ),
        down_partner="CY",
        up_partner="NM",
        sidechain_root_atoms=(sidechain_root,),
        backbone_types=tuple(
            (name, _commonest(types[name]))
            for name in (*mainchain, carbonyl)
            if name != down_atom
        ),
        amide_n_types=amide_n,
        backbone_h_types=tuple(
            (name, _commonest(hydrogen_types[name])) for name in sorted(hydrogen_types)
        ),
        renamed_h_parents=(down_atom,),
        mainchain_torsions=_generic_mainchain_torsions(tuple(mainchain)),
        transplant_icoors=(*mainchain, "up", carbonyl, "down"),
    )


def cap_polymer_profile(atom_array, connection_atom: str, chemdb) -> PolymerProfile:
    """A profile for a residue that terminates a chain rather than continuing it.

    Built from the atom the chain is bonded through, so any capping group is
    described the same way. The peptide bond it makes is real, so the atoms
    forming it take the same types the database gives a backbone.
    """
    alpha = alpha_profile(chemdb)
    backbone_types = dict(alpha.backbone_types)
    hydrogen_types = dict(alpha.backbone_h_types)
    _down_name, alpha_n = alpha.down
    _up_name, alpha_c = alpha.up

    adj, double, element = _heavy_adjacency(atom_array)
    attaches_by_nitrogen = element.get(connection_atom) == "N"

    # the carbonyl oxygen makes a poor frame reference, so prefer any other
    #    heavy neighbour; an amide cap has none at all and the stub is placed
    #    against synthetic references instead
    neighbours = sorted(adj.get(connection_atom, ()))
    neighbour = next(
        (n for n in neighbours if n not in double.get(connection_atom, ())),
        next(iter(neighbours), None),
    )
    frame = tuple(x for x in (neighbour, connection_atom) if x is not None)
    stubs = {c.name: c for c in alpha.caps}

    if attaches_by_nitrogen:
        # the residue before it stands in across the bond
        caps = tuple(
            attr.evolve(stubs[name], refs=refs, bond_to=bond_to)
            for name, refs, bond_to in (
                ("CY", frame, connection_atom),
                ("OY", (*frame, "CY"), "CY"),
                ("CAY", (*frame, "CY"), "CY"),
            )
        )
        types = ((connection_atom, alpha.amide_n_types[0]),)
        h_types = ((connection_atom, hydrogen_types.get(alpha_n)),)
    else:
        caps = tuple(
            attr.evolve(stubs[name], refs=refs, bond_to=bond_to)
            for name, refs, bond_to in (
                ("NM", frame, connection_atom),
                ("CM", (*frame, "NM"), "NM"),
            )
        )
        carbonyl = next(
            (
                o
                for o in sorted(double.get(connection_atom, ()))
                if element.get(o) == "O"
            ),
            None,
        )
        carbonyl_type = next(
            (
                t
                for name, t in alpha.backbone_types
                if name not in alpha.mainchain_atoms
            ),
            None,
        )
        types = tuple(
            [(connection_atom, backbone_types[alpha_c])]
            + ([(carbonyl, carbonyl_type)] if carbonyl else [])
        )
        h_types = ()

    return PolymerProfile(
        name="cap",
        polymer_type=alpha.polymer_type,
        backbone_type="nonstandard",
        reference_restype=None,
        # a cap does not continue the chain, so the atom it bonds through is
        #    the whole of its mainchain
        mainchain_atoms=(connection_atom,),
        down=("down", connection_atom) if attaches_by_nitrogen else None,
        up=None if attaches_by_nitrogen else ("up", connection_atom),
        connection_bond_type=alpha.connection_bond_type,
        caps=caps,
        down_partner="CY" if attaches_by_nitrogen else None,
        up_partner=None if attaches_by_nitrogen else "NM",
        sidechain_root_atoms=(),
        backbone_types=types,
        amide_n_types=(
            (alpha.amide_n_types[0], alpha.amide_n_types[0])
            if attaches_by_nitrogen
            else None
        ),
        backbone_h_types=h_types,
        mainchain_torsions=(),
        transplant_icoors=(),
        renamed_h_parents=(connection_atom,) if attaches_by_nitrogen else (),
    )


def place_atom(a, b, c, d, angle, dihedral):
    """Place a point at distance d from c, angle b-c-x, dihedral a-b-c-x."""
    ang = math.radians(angle)
    dih = math.radians(dihedral)
    bc = c - b
    bc = bc / numpy.linalg.norm(bc)
    n = numpy.cross(b - a, bc)
    n = n / numpy.linalg.norm(n)
    m = numpy.cross(n, bc)
    offset = numpy.array(
        [
            -d * math.cos(ang),
            d * math.sin(ang) * math.cos(dih),
            d * math.sin(ang) * math.sin(dih),
        ]
    )
    return c + offset[0] * bc + offset[1] * m + offset[2] * n


def synthetic_reference(b, c=None):
    """A frame point for a residue with too few atoms to supply one.

    Only its direction matters: the caps are scaffolding, stripped again once
    the residue is typed, so any point off the b-c line serves. With a single
    atom to work from there is no line, and any offset from it does.
    """
    if c is None:
        return b + numpy.array([1.0, 0.0, 0.0])
    axis = c - b
    axis = axis / numpy.linalg.norm(axis)
    trial = numpy.array([1.0, 0.0, 0.0])
    if abs(float(numpy.dot(trial, axis))) > 0.9:
        trial = numpy.array([0.0, 1.0, 0.0])
    perp = numpy.cross(axis, trial)
    return b + perp / numpy.linalg.norm(perp)


# a ring closing from the amide nitrogen back onto the mainchain leaves the
# backbone in a class the peptide potentials describe -- proline, the
# thioprolines, azetidine-2-carboxylic and the pipecolic acids all sit here --
# while an exocyclic substituent does not. Above six atoms the ring stops
# pinning phi and the residue is an n-alkylated backbone that happens to be
# macrocyclized. Both the rule and the bound are modelling decisions.
_PROLINE_LIKE_RING_SIZES = (4, 5, 6)


def _heavy_adjacency(atom_array):
    """Heavy-atom adjacency, and the heavy neighbours reached by a double bond."""
    names = {i: str(n) for i, n in enumerate(atom_array.atom_name)}
    element = {str(n): str(e) for n, e in zip(atom_array.atom_name, atom_array.element)}
    adj: dict = {}
    double: dict = {}
    for i, j, order in atom_array.bonds.as_array():
        a, b = names[int(i)], names[int(j)]
        if element.get(a) == "H" or element.get(b) == "H":
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        if int(order) == struc.BondType.DOUBLE:
            double.setdefault(a, set()).add(b)
            double.setdefault(b, set()).add(a)
    return adj, double, element


def _smallest_ring_size(adj, start: str, end: str) -> Optional[int]:
    """Size of the smallest ring containing the ``start``-``end`` bond."""
    queue = deque([(start, [start])])
    while queue:
        current, path = queue.popleft()
        for neighbour in sorted(adj.get(current, ())):
            if current == start and neighbour == end:
                continue  # the bond itself is not a way around the ring
            if neighbour == end:
                return len(path) + 1
            if neighbour not in path:
                queue.append((neighbour, path + [neighbour]))
    return None


def _ring_substituent_on_nitrogen(adj, element, n_atom: str, ca_atom: str):
    """The heavy substituent on the amide N, or None if it carries only H.

    Returns ``(name, ring_size)``; a ring size outside the proline-like range
    means the residue is not an unmodified alpha backbone.
    """
    for substituent in sorted(adj.get(n_atom, ())):
        if substituent == ca_atom or element.get(substituent) == "H":
            continue
        return substituent, _smallest_ring_size(adj, n_atom, ca_atom)
    return None, None


def _nitrogen_is_unmodified(adj, element, n_atom: str, ca_atom: str) -> bool:
    """Whether every heavy substituent on the amide N closes a small ring."""
    substituent, ring = _ring_substituent_on_nitrogen(adj, element, n_atom, ca_atom)
    if substituent is None:
        return True
    return ring in _PROLINE_LIKE_RING_SIZES


def alpha_backbone_atoms(atom_array, connection_atoms=None):
    """``(N, CA, C)`` of an unmodified alpha backbone, or None.

    The backbone is found by its bonds, never by atom names: a beta amino acid
    and an aromatic hydrazide both carry atoms called N, CA and C. When the
    caller knows which atoms bond to neighbouring residues, the path must end
    on them -- a residue linked through a sidechain carbon has an alpha
    fragment but is not an alpha residue.
    """
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return None
    adj, double, element = _heavy_adjacency(atom_array)

    for n_atom in sorted(a for a in adj if element.get(a) == "N"):
        for ca_atom in sorted(adj[n_atom]):
            if element.get(ca_atom) != "C":
                continue
            for c_atom in sorted(adj[ca_atom]):
                if c_atom == n_atom or element.get(c_atom) != "C":
                    continue
                # the up connection sits on a carbonyl carbon
                if not any(element.get(o) == "O" for o in double.get(c_atom, ())):
                    continue
                if connection_atoms and not set(connection_atoms) <= {n_atom, c_atom}:
                    continue
                if not _nitrogen_is_unmodified(adj, element, n_atom, ca_atom):
                    continue
                return (n_atom, ca_atom, c_atom)
    return None


def _hydrogen_counts(atom_array) -> dict:
    names = {i: str(n) for i, n in enumerate(atom_array.atom_name)}
    element = {str(n): str(e) for n, e in zip(atom_array.atom_name, atom_array.element)}
    counts: dict = {}
    for i, j, _order in atom_array.bonds.as_array():
        a, b = names[int(i)], names[int(j)]
        if element.get(a) == "H":
            counts[b] = counts.get(b, 0) + 1
        elif element.get(b) == "H":
            counts[a] = counts.get(a, 0) + 1
    return counts


def _chain_end_candidates(atom_array, known: str) -> list:
    """Atoms that could carry the backbone's other polymer connection.

    None of them means the residue does not continue the chain at all: it is a
    cap, terminating whatever it is bonded to. Exactly one is a chain member
    seen at a terminus. More than one cannot be told apart.
    """
    adj, double, element = _heavy_adjacency(atom_array)
    hydrogens = _hydrogen_counts(atom_array)
    if element.get(known) == "N":
        candidates = [
            c
            for c in sorted(adj)
            if element.get(c) == "C"
            and any(element.get(o) == "O" for o in double.get(c, ()))
            and sum(1 for o in adj[c] if element.get(o) == "O") >= 2
        ]
    else:
        candidates = [
            n for n in sorted(adj) if element.get(n) == "N" and hydrogens.get(n, 0) >= 1
        ]
    return [c for c in candidates if c != known]


def _far_chain_end(atom_array, known: str) -> Optional[str]:
    """The other end of the backbone, inferred from one known connection.

    A residue seen only at a chain terminus bonds on one side, so the far end
    has to come from its chemistry: the amine that would carry the peptide bond
    if the known end is the carbonyl, the acid if it is the nitrogen. More than
    one candidate is not resolved by guessing -- shortest path and the
    conventional name both pick wrong on real residues -- so it returns None
    and the caller reports what it could not tell apart.
    """
    candidates = _chain_end_candidates(atom_array, known)
    return candidates[0] if len(candidates) == 1 else None


def completed_connection_atoms(atom_array, connection_atoms):
    """Both ends of the backbone where the structure only shows one.

    A residue seen only at a chain terminus is bonded on one side, and one
    connection does not say where the backbone stops. The component definition
    does, having flagged the atoms it gives up on polymerizing; failing that
    the residue's own chemistry usually does. Returns the input unchanged when
    neither settles it, leaving the caller to refuse.
    """
    if not connection_atoms or len(connection_atoms) != 1:
        return connection_atoms
    known = next(iter(connection_atoms))
    present = {str(n) for n in atom_array.atom_name}
    res_name = str(atom_array.res_name[0]) if len(atom_array) else ""

    declared = {e for e in ccd_chain_end_atoms(res_name) if e in present}
    if len(declared) == 2 and known in declared:
        return frozenset(declared)

    far = _far_chain_end(atom_array, known)
    if far is not None:
        return frozenset({known, far})
    return connection_atoms


def mainchain_path(atom_array, connection_atoms) -> Optional[Tuple[str, ...]]:
    """Shortest bonded heavy-atom path between the backbone's two ends."""
    if not connection_atoms or len(connection_atoms) != 2:
        return None
    adj, double, element = _heavy_adjacency(atom_array)
    start, end = _orient_connections(sorted(connection_atoms), double, element)
    if start not in adj or end not in adj:
        return None
    queue = deque([(start, (start,))])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for neighbour in sorted(adj.get(current, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, path + (neighbour,)))
    return None


def _orient_connections(ends, double, element):
    """Order the two connection atoms so the chain runs down-to-up.

    A polymer residue is written from its amine end to its carbonyl end, so
    down attaches at the nitrogen and up at the carbon bearing the carbonyl.
    """
    first, second = ends
    nitrogens = [e for e in ends if element.get(e) == "N"]
    if len(nitrogens) == 1:
        other = second if nitrogens[0] == first else first
        return nitrogens[0], other
    carbonyls = [
        e for e in ends if any(element.get(o) == "O" for o in double.get(e, ()))
    ]
    if len(carbonyls) == 1:
        other = second if carbonyls[0] == first else first
        return other, carbonyls[0]
    return first, second


def _generic_mainchain_torsions(path):
    """phi, theta.., psi, omega over a mainchain of any length.

    Named for what people call them rather than for anything that reads them:
    a residue with no rama or omega table is skipped by the backbone-torsion
    term before its torsions are looked at.
    """
    if len(path) < 3:
        return ()
    out = [("phi", ("down:0", path[0], path[1], path[2]))]
    for i, start in enumerate(range(1, len(path) - 3 + 1), start=1):
        window = path[start : start + 3]
        if len(window) < 3:
            break
        out.append((f"theta{i}", (path[start - 1], *window)))
    out.append(("psi", (path[-3], path[-2], path[-1], "up:0")))
    out.append(("omega", (path[-2], path[-1], "up:0", "up:1")))
    return tuple(out)


def generic_polymer_profile(path) -> PolymerProfile:
    """A profile for a backbone tmol has no peptide description of.

    Everything the alpha profile does to make a residue score as protein is
    dropped: no atom retyping, no hydrogen renaming, no transplanted backbone
    geometry. The residue keeps its ligand atom types; its own cartbonded
    entry covers lengths and angles, and gen_bonded covers torsions, the one
    across the bond to its neighbour included.
    """
    first, last = path[0], path[-1]
    return PolymerProfile(
        name="nonstandard",
        polymer_type="amino_acid",
        # anything but "alpha": the terms that only describe a peptide backbone
        #    key on this and skip
        backbone_type="nonstandard",
        reference_restype=None,
        mainchain_atoms=tuple(path),
        down=("down", first),
        up=("up", last),
        connection_bond_type="AROMATIC",
        caps=(
            CapAtom("CY", "C", (path[2], path[1], first), 1.335, 121.0, 180.0, first),
            CapAtom(
                "OY", "O", (path[1], first, "CY"), 1.231, 123.0, 0.0, "CY", "DOUBLE"
            ),
            CapAtom("CAY", "C", (path[1], first, "CY"), 1.508, 116.0, 180.0, "CY"),
            CapAtom("NM", "N", (path[-3], path[-2], last), 1.335, 116.2, 180.0, last),
            CapAtom("CM", "C", (path[-2], last, "NM"), 1.449, 121.7, 180.0, "NM"),
        ),
        down_partner="CY",
        up_partner="NM",
        # no single atom roots the sidechain; every mainchain atom may carry one
        sidechain_root_atoms=None,
        backbone_types=(),
        amide_n_types=None,
        backbone_h_types=(),
        mainchain_torsions=_generic_mainchain_torsions(path),
        transplant_icoors=(),
    )


# the backbone atoms cartbonded's wildcard rows name. Those rows carry the
# terms that span the peptide bond, which no per-residue entry can supply, and
# they are looked up by atom name -- so a residue whose backbone is called
# something else loses them silently. Nothing else needs a canonical name.
_CANONICAL_ALPHA_BACKBONE = ("N", "CA", "C", "O")


def canonical_alpha_renames(atom_array, connection_atoms=None) -> dict:
    """``{actual name: canonical name}`` for an alpha backbone, or empty.

    The input names are kept as aliases, so a structure that spells them
    otherwise still resolves onto the renamed atoms.
    """
    found = alpha_backbone_atoms(atom_array, connection_atoms)
    if found is None:
        return {}
    _adj, double, element = _heavy_adjacency(atom_array)
    n_atom, ca_atom, c_atom = found
    o_atom = next(
        (o for o in sorted(double.get(c_atom, ())) if element.get(o) == "O"), None
    )
    names = {str(n) for n in atom_array.atom_name}
    res_name = str(atom_array.res_name[0]) if len(atom_array) else "residue"

    renames = {}
    for actual, canonical in zip(
        (n_atom, ca_atom, c_atom, o_atom), _CANONICAL_ALPHA_BACKBONE
    ):
        if actual is None or actual == canonical:
            continue
        if canonical in names:
            raise ValueError(
                f"{res_name}: {canonical!r} is a reserved backbone name for "
                f"noncanonical residues; the backbone {canonical} here is "
                f"{actual!r}, so the atom already called {canonical!r} has to be "
                "renamed in the input"
            )
        renames[actual] = canonical
    return renames


def ring_nitrogen_angle(cartbonded_db):
    """``(x0, K)`` cartbonded fits at a proline ring nitrogen, or None.

    Taken from the wildcard rows rather than measured: the ideal-coordinate
    placement of the down connection gives 91 degrees even for proline itself,
    because the icoor that places it is shared with every alpha residue.
    """
    wildcard = cartbonded_db.residue_params.get("wildcard")
    if wildcard is None:
        return None
    target = (_CANONICAL_RING_ATOM, "N", "+C")
    for params in wildcard.angle_parameters:
        atoms = (str(params.atm1), str(params.atm2), str(params.atm3))
        if atoms in (target, target[::-1]):
            return params.x0, params.K
    return None


# the name the wildcard row uses; a ring atom called this already matches
_CANONICAL_N_SUBSTITUENT = "CN"
_CANONICAL_RING_ATOM = "CD"


def ring_nitrogen_angle_atom(atom_array, connection_atoms=None) -> Optional[str]:
    """The ring atom on the amide N whose angle no wildcard row will match.

    Cartbonded reaches across the peptide bond by atom name, so the angle at a
    ring nitrogen is only scored when its ring atom is called CD. Naming it
    anything else drops the term; returning it here lets the residue carry its
    own copy of the row instead. None when the row already matches, or when
    there is no ring.
    """
    found = alpha_backbone_atoms(atom_array, connection_atoms)
    if found is None:
        return None
    adj, _double, element = _heavy_adjacency(atom_array)
    n_atom, ca_atom, _c_atom = found
    substituent, _ring = _ring_substituent_on_nitrogen(adj, element, n_atom, ca_atom)
    if substituent is None or substituent == _CANONICAL_RING_ATOM:
        return None
    return substituent


def cap_backbone_substitution(atom_array, connection_atom: str, chemdb):
    """``(canonical name, the cap's name for it)`` the wildcard rows will miss.

    Cartbonded reaches across the peptide bond by atom name, and a cap does not
    use a backbone's names: an acetyl's alpha-equivalent is its methyl, and a
    methylamide's is the carbon those rows call CN. It is the atom one bond
    further out than the one the peptide bond is made through.

    None where there is nothing to stand in: a formyl cap has only its carbonyl
    oxygen there, and an amide cap has nothing at all.
    """
    adj, double, element = _heavy_adjacency(atom_array)
    canonical = (
        _CANONICAL_N_SUBSTITUENT
        if element.get(connection_atom) == "N"
        else alpha_profile(chemdb).sidechain_root_atoms[0]
    )
    neighbour = next(
        (
            n
            for n in sorted(adj.get(connection_atom, ()))
            if n not in double.get(connection_atom, ())
        ),
        None,
    )
    if neighbour is None or neighbour == canonical:
        return None
    return canonical, neighbour


def substituted_wildcard_rows(cartbonded_db, canonical, replacement, present):
    """Wildcard rows naming ``canonical`` on this residue, under its own name.

    The rows spanning a peptide bond are matched by name, so a residue that
    calls the atom something else is passed over. Copying them with the name it
    does use, values and all, puts the terms back without inventing any.
    """
    wildcard = cartbonded_db.residue_params.get("wildcard")
    if wildcard is None:
        return {}

    def flip(atoms):
        """The same term read from the other residue's side."""
        return tuple(
            atom[1:] if atom.startswith("+") else "+" + atom for atom in reversed(atoms)
        )

    def rename(atoms):
        # a row naming the atom on the partner's side describes this residue
        #    when read the other way round, which is how the kernel matches it
        if f"+{canonical}" in atoms:
            atoms = flip(atoms)
        if canonical not in atoms:
            return None
        out = []
        for atom in atoms:
            if atom.startswith("+"):
                out.append(atom)
            elif atom == canonical:
                out.append(replacement)
            elif atom not in present:
                return None
            else:
                out.append(atom)
        return tuple(out)

    rows: dict = {"length": [], "angle": [], "torsion": []}
    for params in wildcard.length_parameters:
        atoms = rename((params.atm1, params.atm2))
        if atoms:
            rows["length"].append((atoms, params))
    for params in wildcard.angle_parameters:
        atoms = rename((params.atm1, params.atm2, params.atm3))
        if atoms:
            rows["angle"].append((atoms, params))
    for params in wildcard.torsion_parameters:
        atoms = rename((params.atm1, params.atm2, params.atm3, params.atm4))
        if atoms:
            rows["torsion"].append((atoms, params))
    return rows


def profile_for_atom_array(
    atom_array, connection_atoms=None, chemdb=None
) -> Optional[PolymerProfile]:
    """The profile describing this residue's backbone, or None."""
    if chemdb is None:
        from tmol.database import ParameterDatabase

        chemdb = ParameterDatabase.get_default().chemical
    if connection_atoms and len(connection_atoms) == 1:
        known = next(iter(connection_atoms))
        res_name = str(atom_array.res_name[0]) if len(atom_array) else ""
        present = {str(n) for n in atom_array.atom_name}
        declared = {e for e in ccd_chain_end_atoms(res_name) if e in present}
        # nothing to continue the chain with, and nothing declaring otherwise
        if len(declared) < 2 and not _chain_end_candidates(atom_array, known):
            return cap_polymer_profile(atom_array, known, chemdb)
    connection_atoms = completed_connection_atoms(atom_array, connection_atoms)
    if alpha_backbone_atoms(atom_array, connection_atoms) is not None:
        return alpha_profile(chemdb)
    path = mainchain_path(atom_array, connection_atoms)
    if path is not None and len(path) >= 3:
        return generic_polymer_profile(path)
    return None


def resolve_cap_names(profile: PolymerProfile, existing):
    """Map each cap atom to a name the residue does not already use.

    Cap names are only scaffolding, but noncanonicals do use names like CM, so
    they cannot be fixed strings.
    """
    taken = set(existing)
    resolved = {}
    for cap in profile.caps:
        name = cap.name
        suffix = 0
        while name in taken:
            suffix += 1
            name = f"{cap.name}{suffix}"
        resolved[cap.name] = name
        taken.add(name)
    return resolved


def cap_residue(atom_array, profile: PolymerProfile):
    """Return (capped heavy-atom AtomArray, cap name mapping).

    Hydrogens are dropped: the pipeline derives a SMILES from heavy atoms and the
    bond table, then re-protonates.
    """
    names = {str(n) for n in atom_array.atom_name}
    missing = sorted(profile.required_atoms() - names)
    if missing:
        raise ValueError(f"residue is missing backbone atom(s) {missing}")
    if atom_array.bonds is None or (
        atom_array.array_length() > 1 and atom_array.bonds.get_bond_count() == 0
    ):
        raise ValueError(
            "residue carries no bond table; read the structure with "
            "include_bonds=True so its chemistry can be derived"
        )
    cap_names = resolve_cap_names(profile, names)

    keep = numpy.array([str(e) != "H" for e in atom_array.element])
    kept_indices = numpy.nonzero(keep)[0]
    residue = atom_array[kept_indices]

    pos = {str(n): c for n, c in zip(residue.atom_name, residue.coord)}
    for cap in profile.caps:
        frame = [pos[cap_names.get(r, r)] for r in cap.refs]
        while len(frame) < 3:
            frame = [synthetic_reference(*frame[:2])] + frame
        a, b, c = frame
        pos[cap_names[cap.name]] = place_atom(a, b, c, cap.d, cap.angle, cap.dihedral)

    n_residue = residue.array_length()
    out = struc.AtomArray(n_residue + len(profile.caps))
    out.coord[:n_residue] = residue.coord
    out.atom_name[:n_residue] = residue.atom_name
    out.element[:n_residue] = residue.element
    for offset, cap in enumerate(profile.caps):
        out.coord[n_residue + offset] = pos[cap_names[cap.name]]
        out.atom_name[n_residue + offset] = cap_names[cap.name]
        out.element[n_residue + offset] = cap.element
    for field in ("res_name", "chain_id", "res_id", "hetero"):
        if field in atom_array.get_annotation_categories():
            value = getattr(atom_array, field)[0]
            getattr(out, field)[:] = value

    remap = {int(old): new for new, old in enumerate(kept_indices)}
    bonds = struc.BondList(out.array_length())
    if atom_array.bonds is not None:
        for i, j, bond_type in atom_array.bonds.as_array():
            if i in remap and j in remap:
                bonds.add_bond(remap[i], remap[j], bond_type)
    index = {str(n): i for i, n in enumerate(out.atom_name)}
    order = {"SINGLE": struc.BondType.SINGLE, "DOUBLE": struc.BondType.DOUBLE}
    for cap in profile.caps:
        bonds.add_bond(
            index[cap_names[cap.name]],
            index[cap_names.get(cap.bond_to, cap.bond_to)],
            order[cap.bond_order],
        )
    out.bonds = bonds
    return out, cap_names
