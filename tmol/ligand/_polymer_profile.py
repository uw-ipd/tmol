"""Declarative description of each polymer backbone the NCAA path supports.

A noncanonical polymer residue is prepared as a molecule: its polymer connections
are replaced by chemical stubs (caps), the ligand pipeline types and charges the
capped molecule, then the caps are stripped back to connections. A profile says
which atoms form the backbone, what the caps are, and what the resulting residue
type must declare.
"""

import logging
import math
from collections import Counter, defaultdict, deque
from typing import Optional, Tuple

import attr
import numpy
import biotite.structure as struc
from rdkit import Chem
from rdkit.Chem import rdFMCS

logger = logging.getLogger(__name__)


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
    # a torsion the builder derives from the residue rather than declaring it:
    #    the glycosidic one runs into the base, so its atoms differ per residue
    glycosidic_torsion: Optional[str] = None
    # whose termini patches a generated one is modelled on, where this backbone
    #    is not one the database describes
    terminus_template_backbone: str = "alpha_aa"
    # the mainchain atom the icoor tree is rooted at, where that is not its
    #    first: a nucleotide roots at its 5' oxygen so that losing the
    #    phosphate at a 5' terminus does not orphan the rest of the backbone
    icoor_root: Optional[str] = None

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
        and r.properties.polymer.backbone_type == "alpha_aa"
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
        name="alpha_aa",
        polymer_type="amino_acid",
        backbone_type="alpha_aa",
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
        backbone_type="nonstandard_aa",
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
        # the acid the chain would continue through. Counting its oxygens would
        #    find a free sidechain carboxylate ahead of a backbone carbonyl
        #    whose terminal hydroxyl was never modeled, so what separates them
        #    is the nitrogen an amide sidechain carbon carries and this does not
        candidates = [
            c
            for c in sorted(adj)
            if element.get(c) == "C"
            and any(element.get(o) == "O" for o in double.get(c, ()))
            and not any(element.get(n) == "N" for n in adj[c])
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
    connection does not say where the backbone stops; its own chemistry does.
    Where more than one atom could carry the other connection the conventional
    backbone wins, since a residue deposited under a code that says nothing
    else is far likelier to be linked conventionally than through a sidechain.
    Returns the input unchanged when nothing settles it, leaving the caller to
    refuse.
    """
    if not connection_atoms or len(connection_atoms) != 1:
        return connection_atoms
    known = next(iter(connection_atoms))
    res_name = str(atom_array.res_name[0]) if len(atom_array) else ""

    far = _far_chain_end(atom_array, known)
    if far is not None:
        return frozenset({known, far})

    # more than one candidate: a residue linked through a sidechain carboxyl
    #    looks the same at a terminus as one linked through its own. Prefer the
    #    conventional backbone, which is what a chain of this residue almost
    #    always uses, and say so
    candidates = _chain_end_candidates(atom_array, known)
    conventional = [
        c
        for c in candidates
        if alpha_backbone_atoms(atom_array, frozenset({known, c})) is not None
    ]
    if len(conventional) == 1:
        logger.warning(
            "%s is seen only at a chain terminus and %s could each carry its "
            "other connection; it is read as the conventional backbone through "
            "%s. A copy of the residue in a chain would settle it.",
            res_name or "this residue",
            ", ".join(sorted(candidates)),
            conventional[0],
        )
        return frozenset({known, conventional[0]})
    if len(candidates) > 1:
        logger.warning(
            "%s is seen only at a chain terminus and %s could each carry its "
            "other connection, none of them a conventional backbone; it cannot "
            "be told apart. A copy of the residue in a chain would settle it.",
            res_name or "this residue",
            ", ".join(sorted(candidates)),
        )
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


def generic_polymer_profile(path, element=None, chemdb=None) -> PolymerProfile:
    """A profile for a backbone tmol has no standard description of.

    Everything the standard profiles do to make a residue score as protein or
    as a nucleotide is dropped: no atom retyping, no hydrogen renaming, no
    transplanted backbone geometry. The residue keeps its ligand atom types;
    its own cartbonded entry covers lengths and angles, and gen_bonded covers
    torsions, the one across the bond to its neighbour included.

    A backbone running through a phosphorus is capped as a nucleotide's rather
    than as a peptide's -- an amide stub on a phosphate would be chemistry the
    residue does not have.
    """
    first, last = path[0], path[-1]
    if element is not None and any(element.get(a) == "P" for a in path) and chemdb:
        # a nucleotide chain runs 5' to 3', so the phosphorus end goes first
        if element.get(last) == "P" and element.get(first) != "P":
            path = tuple(reversed(path))
        return _generic_na_profile(path, chemdb)
    return PolymerProfile(
        name="nonstandard_aa",
        polymer_type="amino_acid",
        # anything but "alpha_aa": the terms that only describe a peptide backbone
        #    key on this and skip
        backbone_type="nonstandard_aa",
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
        # a nucleotide seen only at a 5' terminus has one connection and no
        #    phosphate, which is a chain member rather than a cap
        completed = completed_connection_atoms(atom_array, connection_atoms)
        if na_backbone_kind(atom_array, completed) is not None:
            return na_profile(chemdb, na_backbone_kind(atom_array, completed))
        known = next(iter(connection_atoms))
        # nothing to continue the chain with: this residue terminates it
        if not _chain_end_candidates(atom_array, known):
            return cap_polymer_profile(atom_array, known, chemdb)
    connection_atoms = completed_connection_atoms(atom_array, connection_atoms)
    if alpha_backbone_atoms(atom_array, connection_atoms) is not None:
        return alpha_profile(chemdb)
    kind = na_backbone_kind(atom_array, connection_atoms)
    if kind is not None:
        return na_profile(chemdb, kind)
    path = mainchain_path(atom_array, connection_atoms)
    if path is not None and len(path) >= 3:
        _adj, _double, element = _heavy_adjacency(atom_array)
        return generic_polymer_profile(path, element, chemdb)
    return None


def na_sugar_mainchain(adjacency, element):
    """The nucleotide mainchain, found from the sugar ring alone.

    Returns (path, ring) with the path running 5' to 3' -- P-O5'-C5'-C4'-C3'-O3',
    or the same without the phosphate, which a residue seen only at a 5'
    terminus does not have. Derived from the ring rather than from the
    connections, since a residue at a chain end has only one of those and the
    missing one is exactly what has to be worked out.
    """
    for ring in _five_rings(adjacency, element):
        hetero = next(a for a in ring if element.get(a) != "C")
        for c3 in ring:
            if c3 == hetero:
                continue
            exocyclic_o = [
                n
                for n in sorted(adjacency[c3])
                if n not in ring and element.get(n) == "O"
            ]
            for c4 in sorted(adjacency[c3] & ring):
                if c4 == hetero or hetero not in adjacency[c4]:
                    continue
                for c5 in sorted(adjacency[c4] - ring):
                    if element.get(c5) != "C":
                        continue
                    for o5 in sorted(adjacency[c5]):
                        if element.get(o5) != "O" or o5 in ring:
                            continue
                        for o3 in exocyclic_o:
                            path = [o5, c5, c4, c3, o3]
                            if len(set(path)) != 5:
                                continue
                            phosphate = [
                                n
                                for n in sorted(adjacency[o5])
                                if element.get(n) == "P"
                            ]
                            return (
                                tuple(phosphate[:1] + path),
                                tuple(ring),
                            )
    return None, None


def _five_rings(adjacency, element):
    """Five-membered rings carrying exactly one non-carbon, in name order."""
    rings = []
    for hetero in sorted(adjacency):
        if element.get(hetero) in ("C", "H", None):
            continue
        neighbours = sorted(n for n in adjacency[hetero] if element.get(n) == "C")
        if len(neighbours) != 2:
            continue
        first, second = neighbours
        path = _shortest_path(
            {a: bs - {hetero} for a, bs in adjacency.items()}, first, second
        )
        if path is None or len(path) != 4:
            continue
        ring = {hetero, *path}
        if len(ring) == 5 and sum(element.get(a) != "C" for a in ring) == 1:
            rings.append(ring)
    return rings


def na_backbone_kind(atom_array, connection_atoms) -> Optional[str]:
    """ "dna" or "rna" if this residue has a standard nucleotide backbone.

    Standard means a five-membered sugar with the phosphate-to-3'-oxygen path
    closed on it. What hangs off the sugar is not looked at, so a modified base
    or a substituted 2' position is still standard; RNA is told from DNA by an
    oxygen on the sugar, which is where a ribose differs from a deoxyribose.
    """
    if not connection_atoms:
        return None
    adjacency, _double, element = _heavy_adjacency(atom_array)
    path, ring = na_sugar_mainchain(adjacency, element)
    if path is None:
        return None
    # the connections have to be this backbone's ends, not a sidechain's
    if not set(connection_atoms) <= {path[0], path[-1]}:
        return None

    exocyclic_oxygen = any(
        element.get(other_atom) == "O"
        for atom in ring
        if atom not in path
        for other_atom in adjacency[atom]
        if other_atom not in ring and other_atom not in path
    )
    return "rna" if exocyclic_oxygen else "dna"


def _shortest_path(adjacency, start, end):
    """Shortest bonded path between two atoms, or None."""
    queue = deque([(start, (start,))])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for neighbour in sorted(adjacency.get(current, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, path + (neighbour,)))
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


# --------------------------------------------------------------------------- #
# nucleic acids
# --------------------------------------------------------------------------- #

_NA_PROFILE_CACHE: dict = {}


def na_profile(chemdb, kind: str) -> Optional[PolymerProfile]:
    """The DNA or RNA profile, read off the database's own nucleotides.

    Nothing about a nucleotide is written here: the mainchain, the connections,
    the atom types and the torsions are surveyed from the canonical residues
    that already carry them. The backbone is the phosphate and the whole sugar
    -- everything on the near side of the bond from the sugar to the base --
    so only the base is left to the ligand typer.
    """
    key = (id(chemdb), kind)
    if key not in _NA_PROFILE_CACHE:
        _NA_PROFILE_CACHE[key] = _build_na_profile(chemdb, kind)
    return _NA_PROFILE_CACHE[key]


def _canonical_na_residues(chemdb, kind):
    return [
        r
        for r in chemdb.residues
        if r.name == r.base_name
        and r.properties.polymer.backbone_type == kind
        and r.properties.polymer.mainchain_atoms
    ]


def sugar_backbone(residue, mainchain, element):
    """(backbone atom names, sugar ring, glycosidic bond) of one nucleotide.

    The backbone is everything reachable from the mainchain without crossing
    the bond from the sugar to the base, so what counts as backbone is decided
    by the topology rather than by a list of names.
    """
    adjacency = defaultdict(set)
    for a, b, *_ in residue.bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    types = {a.name: a.atom_type for a in residue.atoms}

    ring = _smallest_ring_through(adjacency, mainchain[-2], mainchain[-3])
    if ring is None or len(ring) != 5:
        return None
    # the base hangs off the ring by a heavy atom outside it -- through carbon
    #    in a C-glycoside such as pseudouridine, so the element is not fixed
    substituents = [
        (atom, other)
        for atom in ring
        for other in sorted(adjacency[atom])
        if other not in ring
        and other not in mainchain
        and element.get(types.get(other, "")) not in ("H", None)
    ]
    # a sugar carries substituents of its own -- a 2' hydroxyl, or the methyl
    #    on it -- and they are told from the base by the base being cyclic
    cyclic = [
        bond for bond in substituents if _reaches_a_ring(adjacency, bond[1], bond[0])
    ]
    if len(cyclic) > 1:
        return None
    if not cyclic:
        # no base at all: an abasic site is all backbone
        backbone = {name for name in adjacency if name in types}
        return backbone, ring, None

    anchor, base_atom = cyclic[0]
    backbone, stack = set(), [mainchain[0]]
    while stack:
        name = stack.pop()
        if name in backbone:
            continue
        backbone.add(name)
        for other in adjacency[name]:
            if (name, other) == (anchor, base_atom):
                continue
            stack.append(other)
    return backbone, ring, (anchor, base_atom)


def _reaches_a_ring(adjacency, start, blocked):
    """Whether a ring lies on the far side of the ``blocked``-``start`` bond."""
    seen, stack, order = {blocked}, [start], []
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        order.append(name)
        stack.extend(sorted(adjacency[name]))
    edges = (
        sum(1 for a in order for b in adjacency[a] if b in seen and b != blocked) // 2
    )
    return edges >= len(order)


def _smallest_ring_through(adjacency, first, second):
    """The shortest cycle containing the bond ``first``-``second``."""
    queue = deque([[second, first]])
    seen = {(second, first)}
    while queue:
        path = queue.popleft()
        for other in sorted(adjacency[path[-1]]):
            if other == second and len(path) > 2:
                return path
            if other in path or (path[-1], other) in seen:
                continue
            seen.add((path[-1], other))
            queue.append([*path, other])
    return None


def _build_na_profile(chemdb, kind: str) -> Optional[PolymerProfile]:
    residues = _canonical_na_residues(chemdb, kind)
    if not residues:
        return None
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
    by_name = {name: (atom, kind_) for name, atom, kind_ in connections}
    down_atom, bond_type = by_name["down"]
    up_atom, _ = by_name["up"]

    # the backbone the canonical residues agree on, and the types they give it
    backbones, rings = [], []
    anchor_counts = Counter()
    types = defaultdict(Counter)
    hydrogen_types = defaultdict(Counter)
    for residue in residues:
        found = sugar_backbone(residue, mainchain, element)
        if found is None:
            continue
        backbone, ring, bond = found
        backbones.append(backbone)
        if bond is not None:
            anchor_counts[bond[0]] += 1
        rings.append(tuple(ring))
        atom_types = {a.name: a.atom_type for a in residue.atoms}
        for name in backbone:
            if element.get(atom_types.get(name, "")) == "H":
                continue
            types[name][atom_types[name]] += 1
        for a, b, *_ in residue.bonds:
            for atom, other in ((a, b), (b, a)):
                if atom in backbone and element.get(atom_types.get(other, "")) == "H":
                    hydrogen_types[atom][atom_types[other]] += 1
    if not backbones:
        return None
    shared = set.intersection(*backbones)
    ring = _commonest(Counter(rings))
    anchors = (_commonest(anchor_counts),) if anchor_counts else tuple(sorted(ring))

    torsions = _na_backbone_torsions(residues, shared, mainchain)
    icoors = {r.name: {i.name: i for i in r.icoors} for r in residues}
    reference = residues[0]
    ref_icoors = icoors[reference.name]

    return PolymerProfile(
        name=kind,
        polymer_type=reference.properties.polymer.polymer_type,
        backbone_type=kind,
        reference_restype=reference.name,
        mainchain_atoms=tuple(mainchain),
        down=("down", down_atom),
        up=("up", up_atom),
        connection_bond_type=bond_type,
        caps=_na_caps(ref_icoors, mainchain, down_atom, up_atom),
        down_partner="OY",
        up_partner="PY",
        # the base is the only thing off the backbone, and it hangs off the sugar
        sidechain_root_atoms=anchors,
        backbone_types=tuple(
            (name, _commonest(counter)) for name, counter in sorted(types.items())
        ),
        amide_n_types=None,
        backbone_h_types=tuple(
            (name, _commonest(counter))
            for name, counter in sorted(hydrogen_types.items())
        ),
        mainchain_torsions=torsions,
        transplant_icoors=tuple(sorted(shared) + ["down", "up"]),
        renamed_h_parents=(),
        glycosidic_torsion="chi1",
        icoor_root=_reference_icoor_root(reference, mainchain),
    )


def _na_caps(icoors, mainchain, down_atom, up_atom, reference=None):
    """Stubs standing in for the nucleotides either side, on their own geometry.

    Across the down connection sits the previous residue's O3' and the carbon
    it hangs off, which makes this residue's phosphate the diester it is in a
    chain. Across the up connection sits the next residue's phosphate, so the
    3' oxygen sees an ester rather than a bare hydroxyl.
    """
    phosphate, five_prime = mainchain[0], mainchain[1]
    c5, c4, c3 = mainchain[2], mainchain[3], mainchain[4]
    # the frame is this residue's atoms; the geometry is the canonical one's,
    #    looked up by the role each atom plays rather than by what it is called
    ref_phosphate, ref_up = reference or (phosphate, up_atom)
    return (
        _stub_from_icoor(
            "OY", "O", (c5, five_prime, down_atom), icoors["down"], -64.0, down_atom
        ),
        _stub_from_icoor(
            "CY", "C", (five_prime, phosphate, "OY"), icoors[ref_up], 180.0, "OY"
        ),
        _stub_from_icoor("PY", "P", (c4, c3, up_atom), icoors["up"], 180.0, up_atom),
        _stub_from_icoor(
            "OY1", "O", (c3, up_atom, "PY"), icoors["OP1"], -130.0, "PY", "DOUBLE"
        ),
        _stub_from_icoor("OY2", "O", (c3, up_atom, "PY"), icoors["OP2"], 114.0, "PY"),
        _stub_from_icoor("OY3", "O", (c3, up_atom, "PY"), icoors[phosphate], 0.0, "PY"),
    )


def _na_backbone_torsions(residues, backbone, mainchain):
    """The canonical torsions that name only backbone atoms and connections.

    Everything but the glycosidic one: alpha through zeta and the sugar
    puckers are the same four atoms in every nucleotide, while chi runs into
    the base and is named per residue.
    """

    def spec(atom):
        if atom.atom is not None:
            return atom.atom
        return f"{atom.connection}:{atom.bond_sep_from_conn}"

    out, seen = [], set()
    for residue in residues:
        for torsion in residue.torsions:
            atoms = tuple(spec(a) for a in (torsion.a, torsion.b, torsion.c, torsion.d))
            if torsion.name in seen:
                continue
            if not all(a in backbone or ":" in a for a in atoms):
                continue
            seen.add(torsion.name)
            out.append((torsion.name, atoms))
    return tuple(out)


def glycosidic_torsion_atoms(profile, adjacency, element):
    """The four atoms of a nucleotide's glycosidic torsion, or None.

    Rosetta measures chi from the sugar's ring oxygen through the bond to the
    base: O4'-C1'-N9-C4 in a purine, O4'-C1'-N1-C2 in a pyrimidine. The tables
    are calibrated on that, so the atoms cannot be whichever four a rotatable
    bond search happened to pick.

    The last atom is the base neighbour lying in the most rings, and the lowest
    name among equals -- which reproduces the convention for purines (the fused
    carbon), pyrimidines and C-glycosides alike.
    """
    mainchain = profile.mainchain_atoms
    if len(mainchain) < 5:
        return None
    ring = _smallest_ring_through(adjacency, mainchain[3], mainchain[4])
    if ring is None or len(ring) != 5:
        return None
    hetero = [a for a in ring if element.get(a) not in ("C", None)]
    if len(hetero) != 1:
        return None

    attachment = [
        (atom, other)
        for atom in ring
        for other in sorted(adjacency[atom])
        if other not in ring
        and other not in mainchain
        and element.get(other) not in ("H", None)
        and _reaches_a_ring(adjacency, other, atom)
    ]
    if len(attachment) != 1:
        return None
    anchor, base_atom = attachment[0]

    def rings_through(atom):
        return sum(
            1
            for other in adjacency[atom]
            if other != base_atom and _reaches_a_ring(adjacency, other, atom)
        )

    outward = [
        n
        for n in sorted(adjacency[base_atom])
        if n != anchor and element.get(n) not in ("H", None)
    ]
    if not outward:
        return None
    last = max(outward, key=lambda n: (rings_through(n), [-ord(c) for c in n]))
    return (hetero[0], anchor, base_atom, last)


def _generic_na_profile(path, chemdb) -> PolymerProfile:
    """A nucleotide backbone with no standard description: capped as one.

    Reached by a component the standard profiles cannot map -- a dinucleotide
    fused into one residue, say -- so nothing is retyped and no nucleic acid
    torsion is declared. Only the stubs standing in for its neighbours are a
    nucleotide's, since that is what its two ends actually bond to.
    """
    reference = na_profile(chemdb, "dna")
    icoors = (
        {}
        if reference is None
        else {
            i.name: i
            for r in chemdb.residues
            if r.name == reference.reference_restype
            for i in r.icoors
        }
    )
    return PolymerProfile(
        name="nonstandard_na",
        polymer_type="nucleic_acid",
        backbone_type="nonstandard_na",
        reference_restype=None,
        mainchain_atoms=tuple(path),
        down=("down", path[0]),
        up=("up", path[-1]),
        connection_bond_type="SINGLE",
        caps=_na_caps(
            icoors, (path[0], path[1], path[2], path[-3], path[-2]), path[0], path[-1]
        ),
        down_partner="OY",
        up_partner="PY",
        sidechain_root_atoms=None,
        backbone_types=(),
        amide_n_types=None,
        backbone_h_types=(),
        mainchain_torsions=_generic_mainchain_torsions(path),
        transplant_icoors=(),
        terminus_template_backbone="dna",
        # the 5' patch takes the phosphate away, so the tree cannot be rooted
        #    there; the canonical nucleotide roots at the oxygen after it for
        #    the same reason
        icoor_root=path[1] if len(path) > 1 else None,
    )


def complete_backbone_from_reference(atom_array, profile, param_db):
    """Backbone atoms the residue is missing, donated by a canonical one.

    A residue seen only at a chain end lacks whatever that end does not carry:
    a nucleotide at a 5' terminus has no phosphate at all. The residue type has
    to describe it in a chain, so the missing atoms come from the canonical
    residue of its own class, placed by superimposing that residue's backbone
    on this one's.

    The donor is a canonical residue of the same class, so the graft needs
    no component definition for this residue and works under a code nothing
    defines.
    """
    if profile.reference_restype is None or not profile.backbone_types:
        return atom_array
    present = {str(n) for n in atom_array.atom_name}
    wanted = {name for name, _t in profile.backbone_types} - present
    if not wanted:
        return atom_array

    donor_type = next(
        (r for r in param_db.chemical.residues if r.name == profile.reference_restype),
        None,
    )
    if donor_type is None:
        return atom_array
    coords = _ideal_coords_for(donor_type)
    shared = sorted(present & set(coords))
    if len(shared) < 3 or not wanted <= set(coords):
        return atom_array

    observed = {str(n): c for n, c in zip(atom_array.atom_name, atom_array.coord)}
    rotation, offset = _superposition(
        numpy.array([coords[n] for n in shared]),
        numpy.array([observed[n] for n in shared]),
    )
    added = struc.AtomArray(len(wanted))
    for i, name in enumerate(sorted(wanted)):
        added.coord[i] = coords[name] @ rotation.T + offset
        added.atom_name[i] = name
        added.element[i] = _element_of(donor_type, name)
    for field in ("res_name", "chain_id", "res_id", "hetero"):
        if field in atom_array.get_annotation_categories():
            getattr(added, field)[:] = getattr(atom_array, field)[0]

    combined = atom_array + added
    index = {str(n): i for i, n in enumerate(combined.atom_name)}
    bonds = struc.BondList(combined.array_length())
    if atom_array.bonds is not None:
        for i, j, order in atom_array.bonds.as_array():
            bonds.add_bond(int(i), int(j), int(order))
    for a, b, bond_order, *_ in _localized_bonds(donor_type):
        if a in index and b in index and (a in wanted or b in wanted):
            bonds.add_bond(index[a], index[b], bond_order)
    combined.bonds = bonds
    return combined


def _localized_bonds(residue_type):
    """The residue's bonds as a structure writes them, not as tmol stores them.

    A delocalized group -- a phosphate's two free oxygens, a carboxylate's --
    is stored as a pair of equivalent bonds, which is not a bond order a
    molecule can be built from. One of each such group becomes the double bond
    and the rest single, which is how a structure file carries it.
    """
    delocalized = defaultdict(list)
    localized = []
    for a, b, bond_order, *_ in residue_type.bonds:
        if bond_order == "AROMATIC":
            delocalized[a].append((a, b))
            delocalized[b].append((a, b))
        else:
            localized.append((a, b, 2 if bond_order == "DOUBLE" else 1))

    doubled = set()
    for _centre, group in sorted(delocalized.items()):
        # a ring's bonds are delocalized too and are shared by two atoms each;
        #    only a group all on one centre is a resonance pair to localize
        if len(group) < 2 or any(len(delocalized[b]) > 2 for _a, b in group):
            continue
        doubled.add(group[0])
    seen = set()
    for centre_bonds in delocalized.values():
        for bond in centre_bonds:
            if bond in seen:
                continue
            seen.add(bond)
            localized.append((bond[0], bond[1], 2 if bond in doubled else 1))
    return localized


def _ideal_coords_for(residue_type):
    """A canonical residue's ideal coordinates, by atom name."""
    import cattr

    from tmol.chemical._restypes import RefinedResidueType

    refined = cattr.structure(cattr.unstructure(residue_type), RefinedResidueType)
    xyz = refined.compute_ideal_coords()
    return {ic.name: numpy.asarray(xyz[i]) for i, ic in enumerate(refined.icoors)}


def _element_of(residue_type, name):
    """The element of one of a canonical residue's atoms, from its type."""
    from tmol.database import ParameterDatabase

    types = {a.name: a.atom_type for a in residue_type.atoms}
    elements = {
        a.name: a.element for a in ParameterDatabase.get_default().chemical.atom_types
    }
    return elements.get(types.get(name, ""), name[0])


def _superposition(source, target):
    """Rotation and offset carrying ``source`` onto ``target`` (Kabsch)."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _s, vt = numpy.linalg.svd((source - source_mean).T @ (target - target_mean))
    d = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rotation = vt.T @ numpy.diag([1.0, 1.0, d]) @ u.T
    return rotation, target_mean - source_mean @ rotation.T


def _reference_icoor_root(residue_type, mainchain):
    """The mainchain atom the canonical residue roots its icoor tree at.

    An atom tree's root is its own parent. Which atom that is decides what a
    patch can safely remove: everything is placed against it, directly or not.
    """
    for icoor in residue_type.icoors:
        if icoor.name == icoor.parent and icoor.name in mainchain:
            return icoor.name
    return None


def _base_skeleton(residue_type, mainchain, element_of):
    """An element-and-connectivity graph of one nucleotide's base, or None.

    Bond orders are deliberately dropped: an input's orders are not trusted,
    and the base is identified by its skeleton and heteroatom placement.
    """
    parts = sugar_backbone(residue_type, mainchain, element_of)
    if parts is None:
        return None
    backbone, _ring, glycosidic = parts
    if glycosidic is None:
        return None
    types = {a.name: a.atom_type for a in residue_type.atoms}
    names = sorted(
        n
        for n in types
        if n not in backbone and element_of.get(types[n], "") not in ("H", "")
    )
    if not names:
        return None
    index = {n: i for i, n in enumerate(names)}
    mol = Chem.RWMol()
    for n in names:
        mol.AddAtom(Chem.Atom(element_of[types[n]]))
    for a, b, *_ in residue_type.bonds:
        if a in index and b in index:
            mol.AddBond(index[a], index[b], Chem.BondType.SINGLE)
    return mol.GetMol()


def _base_similarity(query, reference):
    """Jaccard overlap of the largest common substructure of two skeletons.

    Runs to completion: the reference is one of the canonical bases, a dozen
    atoms at most, which bounds the search however large the query is. A
    timeout would return a partial match that scores like a poor one.
    """
    result = rdFMCS.FindMCS(
        [query, reference],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
    )
    shared = result.numAtoms
    if not shared:
        return 0.0
    return shared / (query.GetNumAtoms() + reference.GetNumAtoms() - shared)


def na_base_reference(residue_type, profile, chemdb, minimum=0.5):
    """The canonical nucleotide whose base tables this one is scored on.

    Chosen by matching the base's skeleton against the canonical bases of the
    same polymer, so a modification is scored on the base it modifies without
    anything being declared: 8-oxoguanine keeps guanine's tables because
    guanine is still the closest of the four. A base too unlike any of them --
    or none at all, as at an abasic site -- falls back to the polymer's
    averaged base, so the residue is still scored rather than dropped.
    """
    from tmol.score.na_torsion import BASE_FOR_NAME3, UNKNOWN_BASE_REFERENCE

    kind = profile.backbone_type
    fallback = UNKNOWN_BASE_REFERENCE[kind]
    element_of = {at.name: at.element for at in chemdb.atom_types}
    query = _base_skeleton(residue_type, profile.mainchain_atoms, element_of)
    if query is None:
        return fallback

    # the table is keyed by PDB code, the database by its own residue name;
    #    the io equivalence class is what carries one to the other
    references = sorted(
        (r.io_equiv_class, r)
        for r in chemdb.residues
        if r.name == r.base_name
        and r.properties.polymer.backbone_type == kind
        and r.io_equiv_class in BASE_FOR_NAME3
    )
    best, score = fallback, minimum
    for name, reference in references:
        skeleton = _base_skeleton(
            reference, reference.properties.polymer.mainchain_atoms, element_of
        )
        if skeleton is None:
            continue
        similarity = _base_similarity(query, skeleton)
        if similarity > score:
            best, score = name, similarity
    return best
