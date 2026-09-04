"""Pick the canonical rotamer library a noncanonical residue samples from.

The reference supplies chi geometry only: nothing here reaches the scoring
term, which resolves its library from the residue's own name. A candidate is
adopted only when the residue has a chain of rotatable bonds corresponding to
the reference's chi, so the borrowed numbers name the torsions the library
measured -- including which atom each is measured to, which fixes the value.
"""

import itertools
from typing import Mapping, Optional, Sequence

import attr
import cattr
import numpy

from tmol.chemical import RefinedResidueType

# every atom is weighted by how far it sits from the sidechain root, halving
#    per bond: an error at chi1 swings the whole sidechain where one at the far
#    end moves a single atom. One curve for every candidate, so that the costs
#    of competing references are on the same scale and can be compared
FADE = 0.5
# the two charges add: a corresponding atom of a different element pays for the
#    element, and pays again if it also changes what the atom does to a torsion
SUBSTITUTE_POLAR = 0.5
SUBSTITUTE_ELEMENT = 1.5
# a residue reads every chi its reference defines: a shorter reference of the
#    same shape always exists and is always cheaper, so nothing is gained by
#    reading a longer library's leading chi and leaving the rest unsampled
# a sidechain reaching well past the library's last chi is a different residue
MAX_EXTRA_CHI = 2
# above this the reference is refused
MAX_COST = 1.0


class ReferenceRefused(Exception):
    """A candidate reference cannot supply this residue's chi."""


# a three-coordinate atom whose bond angles sum to more than this is planar
PLANAR_ANGLE_SUM = 350.0


def planar_atoms(adjacency, coords) -> frozenset:
    """Heavy atoms that are planar, from the angles at them.

    Read from geometry rather than bond orders: which bond of a delocalized
    group a file writes as double differs between sources, and a guanidinium
    nitrogen is planar however its tautomer was drawn. ``adjacency`` must
    include hydrogens, since they carry the angles.
    """
    planar = set()
    for atom, neighbours in adjacency.items():
        neighbours = sorted(neighbours)
        if len(neighbours) != 3 or atom not in coords:
            continue
        try:
            vectors = [coords[n] - coords[atom] for n in neighbours]
        except KeyError:
            continue
        norms = [numpy.linalg.norm(v) for v in vectors]
        if min(norms) < 1e-6:
            continue
        total = 0.0
        for i, j in ((0, 1), (0, 2), (1, 2)):
            cosine = numpy.dot(vectors[i], vectors[j]) / (norms[i] * norms[j])
            total += numpy.degrees(numpy.arccos(numpy.clip(cosine, -1.0, 1.0)))
        if total > PLANAR_ANGLE_SUM:
            planar.add(atom)
    return frozenset(planar)


@attr.s(auto_attribs=True, frozen=True)
class SidechainGraph:
    """A residue's heavy-atom graph, with the backbone marked off.

    ``mainchain`` is in sequence order, so the atom before an attachment is the
    one toward the residue's own N terminus -- the atom chi1 is measured from.
    """

    name: str
    element: Mapping[str, str]
    # atoms their resolved atom type marks as hydrogen bond donors / acceptors
    donor: frozenset
    acceptor: frozenset
    adjacency: Mapping[str, frozenset]
    rigid: frozenset
    # heavy atoms carrying a bond that is not a plain single: their torsions
    #    are two-fold and planar where an sp3 centre's are three-fold
    planar: frozenset
    mainchain: tuple
    backbone_type: str
    chirality: str
    coords: Optional[Mapping[str, numpy.ndarray]] = None

    @property
    def mainchain_set(self):
        return frozenset(self.mainchain)

    def sidechain_neighbours(self, atom):
        return frozenset(
            n for n in self.adjacency.get(atom, ()) if n not in self.mainchain_set
        )

    def attachments(self):
        """Mainchain atoms carrying a sidechain, in sequence order."""
        return tuple(a for a in self.mainchain if self.sidechain_neighbours(a))

    def anchor(self, attachment):
        """The mainchain atom chi1 is measured from: the one toward N."""
        position = self.mainchain.index(attachment)
        if position > 0:
            return self.mainchain[position - 1]
        return self.mainchain[1] if len(self.mainchain) > 1 else None

    def bridges_backbone(self):
        """Whether a sidechain runs from one backbone atom back to another.

        Proline's ring does; a pendant ring does not, however saturated.
        """
        seen = set()
        for attachment in self.attachments():
            for start in self.sidechain_neighbours(attachment):
                if start in seen:
                    continue
                component, stack = {start}, [start]
                touching = {attachment}
                while stack:
                    node = stack.pop()
                    for nbr in self.adjacency.get(node, ()):
                        if nbr in self.mainchain_set:
                            touching.add(nbr)
                        elif nbr not in component:
                            component.add(nbr)
                            stack.append(nbr)
                seen |= component
                if len(touching) > 1:
                    return True
        return False


def _degenerate_turn(graph: "SidechainGraph", previous: str, atom: str) -> bool:
    """Whether turning previous-atom moves nothing distinguishable.

    A tetrahedral centre carrying three interchangeable terminal substituents
    -- a deprotonated phosphate or sulfonate, a tert-butyl -- has no torsion:
    every third of a turn returns the same structure. A planar pair, as a
    carboxylate's two oxygens, is not this: its plane still has an orientation,
    which is why glutamate's last chi is a chi.
    """
    substituents = [n for n in graph.adjacency.get(atom, ()) if n != previous]
    if len(substituents) < 3 or atom in graph.planar:
        return False
    if len({graph.element[n] for n in substituents}) != 1:
        return False
    return all(not (graph.adjacency.get(n, frozenset()) - {atom}) for n in substituents)


def chi_paths(graph: SidechainGraph, n_chi: int) -> list:
    """Every way to read n_chi chi off this residue, as paths p0..pN+1.

    Chi i turns the bond (p[i-1], p[i]) and is measured to p[i+1]. Only the
    turned bonds must be rotatable; the atom a chi is measured to may sit
    across a rigid bond, as tyrosine's chi2 is measured into its ring.
    """
    if n_chi < 1:
        return []
    mainchain = graph.mainchain_set
    paths = []

    def extend(path):
        depth = len(path) - 1
        if depth == n_chi + 1:
            paths.append(tuple(path))
            return
        if depth == n_chi and _degenerate_turn(graph, path[-2], path[-1]):
            return
        for nbr in sorted(graph.adjacency.get(path[-1], ())):
            if nbr in path:
                continue
            if depth < n_chi:
                # p1..pN are sidechain, and their bonds are the ones turned
                if nbr in mainchain:
                    continue
                if frozenset((path[-1], nbr)) in graph.rigid:
                    continue
            extend(path + [nbr])

    for attachment in graph.attachments():
        if graph.anchor(attachment) is None:
            continue
        extend([attachment])
    return paths


def longest_chi(graph: SidechainGraph) -> int:
    """The most chi this residue offers along one chain.

    A rigid bond is crossed rather than turned, so chi past a ring or a double
    bond still count: phosphotyrosine turns its phosphate ester beyond the
    ring. The last bond of a chain is never a chi, having nothing beyond it to
    measure to.
    """
    mainchain = graph.mainchain_set
    best = 0

    def turns_along(path):
        return sum(
            1
            for i in range(len(path) - 2)
            if frozenset((path[i], path[i + 1])) not in graph.rigid
            and not _degenerate_turn(graph, path[i], path[i + 1])
        )

    def walk(path):
        nonlocal best
        best = max(best, turns_along(path))
        if len(path) > 2 and path[-1] in mainchain:
            return
        for nbr in sorted(graph.adjacency.get(path[-1], ())):
            if nbr in path:
                continue
            # a ring closing on the backbone ends the chain at the atom it
            #    closes on, which is what the last chi is measured to
            walk(path + [nbr])

    for attachment in graph.attachments():
        for start in sorted(graph.sidechain_neighbours(attachment)):
            walk([attachment, start])
    return best


@attr.s(auto_attribs=True, frozen=True)
class ReferenceProfile:
    """A canonical residue's rotamer library, as chi read off its own graph."""

    name: str
    graph: SidechainGraph
    path: tuple
    # the chi it declares, as (a, b, c, d) atom names
    chi: tuple = ()

    @property
    def n_chi(self):
        return len(self.path) - 2


def _character(graph, atom) -> tuple:
    """(donates, accepts, element) for an atom, from its resolved atom type."""
    return (atom in graph.donor, atom in graph.acceptor, graph.element[atom])


def _distances(graph, path) -> dict:
    """Bonds from the sidechain root to every sidechain atom, shortest first."""
    root = path[1] if len(path) > 1 else path[0]
    distance = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in sorted(graph.adjacency.get(node, ())):
            if nbr in distance or nbr in graph.mainchain_set:
                continue
            distance[nbr] = distance[node] + 1
            queue.append(nbr)
    return distance


def _parity(graph, atom, neighbours) -> Optional[float]:
    """Signed volume at an atom, in the frame of three of its neighbours.

    None where the sign says nothing: at a planar centre it is a cis/trans
    choice, and with fewer than three neighbours placed there is no frame.
    """
    if graph.coords is None or atom in graph.planar or len(neighbours) < 3:
        return None
    try:
        origin = graph.coords[atom]
        a, b, c = (graph.coords[x] - origin for x in neighbours[:3])
    except KeyError:
        return None
    return float(numpy.dot(numpy.cross(a, b), c))


def _embed(graph, path, reference, horizon):
    """Map the reference's sidechain onto this residue's, or None.

    The chi path seeds the correspondence; the rest of the reference is grown
    onto the residue by its bonds, so a five-membered ring cannot be read as a
    six-membered one. Only connectivity constrains the map, never bond order,
    which the dictionary and the residue database do not agree on for amides.
    Handedness is checked as the map is built, and a centre
    of the wrong handedness rejects that correspondence rather than the whole
    residue: another way of laying the two on each other may still work.
    """
    forward = {reference.path[i]: path[i] for i in range(horizon + 1)}
    used = set(forward.values())
    ref_graph = reference.graph

    order = []
    seen = set(forward)
    queue = list(forward)
    while queue:
        node = queue.pop(0)
        for nbr in sorted(ref_graph.adjacency.get(node, ())):
            if nbr in seen or nbr in ref_graph.mainchain_set:
                continue
            seen.add(nbr)
            order.append((nbr, node))
            queue.append(nbr)

    def consistent(ref_atom, atom):
        for nbr in ref_graph.adjacency.get(ref_atom, ()):
            image = forward.get(nbr)
            if image is None:
                continue
            if image not in graph.adjacency.get(atom, ()):
                return False
        return True

    def handed(ref_atom, atom):
        placed = [
            n for n in sorted(ref_graph.adjacency.get(ref_atom, ())) if n in forward
        ]
        mine = _parity(graph, atom, [forward[n] for n in placed])
        theirs = _parity(ref_graph, ref_atom, placed)
        return mine is None or theirs is None or mine * theirs > 0

    def extend(index):
        if index == len(order):
            return dict(forward)
        ref_atom, ref_from = order[index]
        anchor = forward.get(ref_from)
        candidates = (
            sorted(n for n in graph.adjacency.get(anchor, ()) if n not in used)
            if anchor is not None
            else []
        )
        for atom in candidates:
            if not consistent(ref_atom, atom):
                continue
            forward[ref_atom] = atom
            used.add(atom)
            if handed(ref_atom, atom) and all(
                handed(r, forward[r]) for r in forward if r in ref_graph.adjacency
            ):
                found = extend(index + 1)
                if found is not None:
                    del forward[ref_atom]
                    used.discard(atom)
                    return found
            del forward[ref_atom]
            used.discard(atom)
        # the reference reaches somewhere this residue does not; leave it unmapped
        return extend(index + 1)

    for position in range(horizon + 1):
        if not handed(reference.path[position], path[position]):
            return None
    return extend(0)


def _substitution_cost(graph, mine, reference, theirs) -> float:
    """What it costs that two corresponding atoms are not the same atom.

    The two charges add: swapping a carbon for a hydroxyl oxygen changes both
    the element and what the atom does to a torsion, and costs both, where
    swapping it for a thiol sulfur changes only the element.
    """
    a = _character(graph, mine)
    b = _character(reference, theirs)
    cost = 0.0
    if any(a[:2]) != any(b[:2]):
        cost += SUBSTITUTE_POLAR
    if a[2] != b[2]:
        cost += SUBSTITUTE_ELEMENT
    return cost


def path_cost(graph: SidechainGraph, path: tuple, reference: ReferenceProfile) -> float:
    """How poorly the reference's chi stand in for this reading of the residue.

    The reference's sidechain is laid onto the residue's and the two compared
    atom by atom: an atom with no counterpart costs its weight, one whose
    counterpart is the wrong kind of atom costs a fraction of it.
    """
    horizon = len(path) - 1
    for position in range(horizon):
        # the atoms a chi turns between decide its periodicity; only the atom a
        #    chi is measured to may differ, and that is the horizon
        if (path[position] in graph.planar) != (
            reference.path[position] in reference.graph.planar
        ):
            raise ReferenceRefused(
                f"{path[position]} and {reference.path[position]} differ in "
                "hybridization on the chi path"
            )

    mapping = _embed(graph, path, reference, horizon)
    if mapping is None:
        raise ReferenceRefused("no correspondence of the right handedness")

    def fade(distance):
        return FADE**distance

    ours = _distances(graph, path)
    theirs = _distances(reference.graph, reference.path)
    image = set(mapping.values())

    cost = 0.0
    for ref_atom, atom in mapping.items():
        if atom in ours:
            cost += fade(ours[atom]) * _substitution_cost(
                graph, atom, reference.graph, ref_atom
            )
    for ref_atom, distance in theirs.items():
        if ref_atom not in mapping:
            cost += fade(distance)
    for atom, distance in ours.items():
        if atom not in image:
            cost += fade(distance)
    return cost


def reference_cost(graph: SidechainGraph, reference: ReferenceProfile):
    """The cost of this reference and the chi path that earns it.

    Raises ReferenceRefused where the residue cannot read the reference's chi
    at all, so that no cost could make the borrowed numbers mean anything.
    """
    if reference.n_chi == 0:
        raise ReferenceRefused("reference defines no chi")
    if graph.chirality != reference.graph.chirality:
        raise ReferenceRefused(
            f"chirality {graph.chirality} against {reference.graph.chirality}"
        )
    if graph.bridges_backbone() != reference.graph.bridges_backbone():
        raise ReferenceRefused(
            "sidechain rejoins the backbone "
            + ("only here" if graph.bridges_backbone() else "only in the reference")
        )

    declared = longest_chi(graph)
    if declared == 0:
        raise ReferenceRefused("no rotatable sidechain bond")
    if declared - reference.n_chi > MAX_EXTRA_CHI:
        raise ReferenceRefused(
            f"{declared - reference.n_chi} chi past the reference's last"
        )

    if declared < reference.n_chi:
        raise ReferenceRefused(
            f"{reference.n_chi - declared} chi short of the reference"
        )
    paths = chi_paths(graph, reference.n_chi)
    if not paths:
        raise ReferenceRefused("no chi path corresponds")
    scored = []
    for path in paths:
        try:
            scored.append((path_cost(graph, path, reference), path))
        except ReferenceRefused:
            continue
    if not scored:
        raise ReferenceRefused("every reading has the wrong handedness")
    return min(scored, key=lambda entry: (entry[0], entry[1]))


@attr.s(auto_attribs=True, frozen=True)
class ReferenceChoice:
    """The reference chosen for one residue, and what it beat."""

    name: Optional[str]
    cost: Optional[float]
    path: Optional[tuple]
    runner_up: Optional[str]
    gap: Optional[float]
    refusals: Mapping[str, str]


def choose_reference(
    graph: SidechainGraph,
    references: Sequence[ReferenceProfile],
    *,
    max_cost: float = MAX_COST,
) -> ReferenceChoice:
    """The cheapest usable reference for this residue, or none under max_cost."""
    scored, refusals = [], {}
    for candidate in references:
        try:
            cost, path = reference_cost(graph, candidate)
        except ReferenceRefused as refused:
            refusals[candidate.name] = str(refused)
            continue
        scored.append((cost, candidate.name, path))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    if not scored or scored[0][0] > max_cost:
        best = scored[0] if scored else (None, None, None)
        return ReferenceChoice(None, best[0], None, best[1], None, refusals)
    cost, name, path = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else None
    gap = scored[1][0] - cost if len(scored) > 1 else None
    return ReferenceChoice(name, cost, path, runner_up, gap, refusals)


def _declared_chi(residue_type, element) -> tuple:
    """The chi a residue declares, up to its first proton chi, as atom tuples."""
    by_name = {t.name: t for t in residue_type.torsions}
    path = _declared_chi_path(residue_type, element)
    out = []
    for index in range(1, len(path) - 1):
        torsion = by_name["chi%d" % index]
        out.append((torsion.a.atom, torsion.b.atom, torsion.c.atom, torsion.d.atom))
    return tuple(out)


def _declared_chi_path(residue_type, element) -> tuple:
    """The chi path a residue type declares, up to its first proton chi.

    A chi measured to a hydrogen is optH's to sample and no rotamer library
    counts one among its chi; chi past a branch start a second chain the
    reference cannot reach.
    """
    by_name = {t.name: t for t in residue_type.torsions}
    path = []
    for index in itertools.count(1):
        torsion = by_name.get("chi%d" % index)
        if torsion is None:
            break
        atoms = (torsion.b.atom, torsion.c.atom, torsion.d.atom)
        if any(a is None for a in atoms):
            break
        if element[atoms[2]].upper() == "H":
            break
        if not path:
            path = list(atoms)
            continue
        if atoms[0] != path[-2] or atoms[1] != path[-1]:
            break
        path.append(atoms[2])
    return tuple(path)


def _element_map(atoms, atom_type_index, owner: str) -> dict:
    """Element of every atom, read from its atom type's database record.

    An atom type the index does not describe is an error rather than a guess:
    substituting an element silently gives the atom another one's chemistry.
    """
    elements = {}
    for atom in atoms:
        record = atom_type_index.get(atom.atom_type)
        if record is None:
            raise ValueError(
                f"{owner}: atom {atom.name} has atom type {atom.atom_type}, "
                "which the atom type index does not describe"
            )
        elements[atom.name] = record.element
    return elements


def graph_from_residue_type(residue_type, atom_type_index, coords=None):
    """A SidechainGraph for a residue type.

    atom_type_index maps an atom type name to its database record, which is
    where an atom's element and its hydrogen bonding roles both come from.
    """

    def described(name):
        return atom_type_index.get(name)

    element = _element_map(residue_type.atoms, atom_type_index, residue_type.name)
    donor = frozenset(
        a.name
        for a in residue_type.atoms
        if described(a.atom_type) and described(a.atom_type).is_donor
    )
    acceptor = frozenset(
        a.name
        for a in residue_type.atoms
        if described(a.atom_type) and described(a.atom_type).is_acceptor
    )
    heavy = {name for name, symbol in element.items() if symbol.upper() != "H"}
    adjacency = {name: set() for name in element}
    rigid = set()
    for bond in residue_type.bonds:
        adjacency[bond[0]].add(bond[1])
        adjacency[bond[1]].add(bond[0])
        order = bond[2] if len(bond) > 2 else "SINGLE"
        if order != "SINGLE" and bond[0] in heavy and bond[1] in heavy:
            rigid.add(frozenset((bond[0], bond[1])))
    planar = planar_atoms(adjacency, coords or {})
    polymer = residue_type.properties.polymer
    return SidechainGraph(
        name=residue_type.name,
        element=element,
        donor=donor,
        acceptor=acceptor,
        adjacency={
            k: frozenset(n for n in v if n in heavy)
            for k, v in adjacency.items()
            if k in heavy
        },
        rigid=frozenset(rigid),
        planar=planar,
        mainchain=tuple(polymer.mainchain_atoms),
        backbone_type=polymer.backbone_type,
        chirality=polymer.sidechain_chirality,
        coords=coords,
    )


def reference_profiles(chemical_database, library_names) -> list:
    """ReferenceProfiles for the canonical residues that own a rotamer library."""
    atom_type_index = {a.name: a for a in chemical_database.atom_types}
    wanted = {name.upper() for name in library_names}
    profiles = []
    for residue in chemical_database.residues:
        if residue.name.upper() not in wanted:
            continue
        element = _element_map(residue.atoms, atom_type_index, residue.name)
        path = _declared_chi_path(residue, element)
        if len(path) < 3:
            continue
        refined = cattr.structure(cattr.unstructure(residue), RefinedResidueType)
        # ideal_coords is indexed by icoor order, not atom order
        coords = {
            icoor.name: refined.ideal_coords[index]
            for index, icoor in enumerate(refined.icoors)
        }
        graph = graph_from_residue_type(residue, atom_type_index, coords=coords)
        profiles.append(
            ReferenceProfile(residue.name, graph, path, _declared_chi(residue, element))
        )
    return profiles


_PROFILE_CACHE = {}


def _cached_profiles(chemical_database, library_names) -> list:
    """reference_profiles memoized, since every residue asks for the same set."""
    key = (id(chemical_database), tuple(sorted(library_names)))
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = reference_profiles(chemical_database, library_names)
    return _PROFILE_CACHE[key]


def library_chi_count(name, chemical_database, library_names) -> int:
    """How many chi the named rotamer library defines, or 0 if it has none."""
    if not name:
        return 0
    for profile in _cached_profiles(chemical_database, library_names):
        if profile.name == name:
            return profile.n_chi
    return 0


def reference_for_residue_type(
    residue_type,
    chemical_database,
    library_names,
    *,
    extra_atom_types: Sequence = (),
    max_cost: float = MAX_COST,
) -> Optional[str]:
    """The canonical rotamer library this residue type samples chi from.

    extra_atom_types describes the types the residue introduced, which the
    database does not yet hold; without them its own atoms read as carbon.

    None when no library corresponds closely enough, in which case the residue
    is packed from whatever chi samples it declares itself.
    """
    polymer = residue_type.properties.polymer
    if not polymer.is_polymer or polymer.polymer_type != "amino_acid":
        return None
    atom_type_index = {a.name: a for a in chemical_database.atom_types}
    atom_type_index.update({a.name: a for a in extra_atom_types})
    refined = cattr.structure(cattr.unstructure(residue_type), RefinedResidueType)
    # ideal_coords is indexed by icoor order, not atom order
    coords = {
        icoor.name: refined.ideal_coords[index]
        for index, icoor in enumerate(refined.icoors)
    }
    graph = graph_from_residue_type(residue_type, atom_type_index, coords=coords)
    profiles = _cached_profiles(chemical_database, library_names)
    return choose_reference(graph, profiles, max_cost=max_cost).name


def anchor_atom(mainchain, attachment) -> Optional[str]:
    """The mainchain atom on the N-terminal side of a sidechain's attachment.

    Chi1 is measured from it, so a residue and the reference it borrows from
    have to pick the corresponding one: always the neighbour towards the
    residue's own N terminus, whatever the backbone's length or naming.
    """
    mainchain = tuple(mainchain)
    if attachment not in mainchain:
        return None
    index = mainchain.index(attachment)
    return mainchain[index - 1] if index > 0 else None


def transferred_chi(reference: ReferenceProfile, path, anchor) -> Optional[tuple]:
    """The reference's chi, read onto this residue through the path alignment.

    Every atom of a declared chi sits on the reference's chi path except the
    first of chi1, which is the mainchain atom it is measured from. Taking the
    same positions off the residue's path gives chi that correspond atom for
    atom, so a borrowed torsion means here what it meant there. None where the
    correspondence cannot be made, which is a reason to refuse the reference
    rather than to guess at it.
    """
    if len(path) != len(reference.path):
        return None
    position = {atom: index for index, atom in enumerate(reference.path)}
    reference_anchor = anchor_atom(reference.graph.mainchain, reference.path[0])
    out = []
    for atoms in reference.chi:
        mapped = []
        for atom in atoms:
            if atom in position:
                mapped.append(path[position[atom]])
            elif atom == reference_anchor and anchor is not None:
                mapped.append(anchor)
            else:
                return None
        out.append(tuple(mapped))
    return tuple(out)


def reference_chi_for_graph(
    graph: SidechainGraph, references, *, max_cost: float = MAX_COST
):
    """The cheapest reference this residue can take its chi from, and those chi.

    A reference whose chi cannot be transferred is passed over rather than
    borrowed from: numbers read off a library mean nothing unless the torsions
    they name are the ones the library measured.
    """
    scored = []
    for candidate in references:
        try:
            cost, path = reference_cost(graph, candidate)
        except ReferenceRefused:
            continue
        if cost <= max_cost:
            scored.append((cost, candidate.name, candidate, path))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    for _cost, name, candidate, path in scored:
        chi = transferred_chi(candidate, path, anchor_atom(graph.mainchain, path[0]))
        if chi is not None:
            return name, chi
    return None, ()
