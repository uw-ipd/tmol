"""Patches giving a residue a connection where a covalent bond attaches.

A glycan's link to its protein, or a ligand's to a sidechain, is a bond the
residue types know nothing about: serine declares no connection at OG, and a
sugar declares none at its hydroxyls. Each attachment site becomes a patch that
removes displaced atoms and adds a connection, so a residue linked at several
sites is the combination of its patches. A site with available valence can also
attach without losing hydrogen.

Patches name atoms outright rather than matching a pattern: a pattern cannot
tell one hydroxyl of a sugar from another. They are therefore scoped to the one
residue type they were generated for.
"""

import attr
import biotite.structure as struc
import networkx
import numpy as np
from atomworks.io.utils.leaving_atoms import get_leaving_atom_groups

from tmol.database.chemical import (
    Atom,
    ChiSamples,
    Connection,
    Icoor,
    IcoorVariant,
    Torsion,
    UnresolvedAtom,
    VariantScope,
    VariantType,
)

CONNECTION_PREFIX = "conj_"

# Generic staggered attachment grid; this is not a fitted distribution for
# every glycosidic, amide or other conjugated bond.
LINKAGE_SAMPLES = (60.0, -60.0, 180.0)


def connection_name(atom: str) -> str:
    """The name a conjugation connection takes at an atom."""
    return CONNECTION_PREFIX + atom


def _element_for_atom(residue_type, chemdb):
    element = {atom_type.name: atom_type.element for atom_type in chemdb.atom_types}
    return {atom.name: element.get(atom.atom_type) for atom in residue_type.atoms}


def _icoor_leaves(residue_type):
    """Atoms no other atom is built from."""
    referenced = {
        ref
        for icoor in residue_type.icoors
        for ref in (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)
    }
    return {atom.name for atom in residue_type.atoms} - referenced


def _hydrogens_on(residue_type, atom: str, chemdb):
    """The hydrogens bonded to ``atom``, by name."""
    element = _element_for_atom(residue_type, chemdb)
    return sorted(
        other
        for first, second, *_ in residue_type.bonds
        for this, other in ((first, second), (second, first))
        if this == atom and element.get(other) == "H"
    )


def _leaves_excluding(residue_type, removed):
    """Atoms nothing still present is built from."""
    referenced = {
        ref
        for icoor in residue_type.icoors
        if icoor.name not in removed
        for ref in (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)
    }
    return {a.name for a in residue_type.atoms if a.name not in removed} - referenced


def leaving_atoms(residue_type, atom: str, chemdb, n_leaving: int = 1):
    """The hydrogens the bond displaces, or None where the site has none.

    Preparation fills the free valence a covalent bond left behind, so whatever
    the site is -- a hydroxyl oxygen, an anomeric carbon -- the atom in the
    bond's way is a hydrogen. How many go depends on what the bond makes of the
    site: acylating a lysine gives an amide, which keeps one of the amine's
    three hydrogens, where etherifying a hydroxyl displaces its only one.

    They are peeled from the tip: each hydrogen taken is one nothing still
    present is built from, since lysine's three are placed one against the next
    and taking the first would leave the others with no frame.
    """
    candidates = _hydrogens_on(residue_type, atom, chemdb)
    if not candidates:
        return None
    removed = []
    for _ in range(max(min(n_leaving, len(candidates)), 0)):
        remaining = [h for h in candidates if h not in removed]
        if not remaining:
            break
        leaves = _leaves_excluding(residue_type, set(removed))
        removed.append(next((h for h in remaining if h in leaves), remaining[0]))
    return tuple(removed)


def leaving_atom(residue_type, atom: str, chemdb) -> str:
    """The single hydrogen standing where the bond attaches, or None."""
    leaving = leaving_atoms(residue_type, atom, chemdb, 1)
    return leaving[0] if leaving else None


def declared_heavy_leaving_groups(atom_array):
    """Absent template-declared groups at connected sites, scoped by residue.

    An unresolved atom is still present and cannot be removed here. Different
    copies using the same attachment patch must agree on their leaving groups.
    """
    templates = getattr(atom_array, "_custom_ccd_registry", {})
    if not templates or atom_array.bonds is None:
        return {}
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    bonds = atom_array.bonds.as_array()[:, :2]
    residues = np.searchsorted(starts, bonds, side="right") - 1
    endpoints = np.unique(bonds[residues[:, 0] != residues[:, 1]])
    groups, present, result = {}, {}, {}
    for index in endpoints:
        name, atom = str(atom_array.res_name[index]), str(atom_array.atom_name[index])
        if name not in templates:
            continue
        if name not in groups:
            template = templates[name]
            heavy = set(template.atom_name[~np.isin(template.element, ("H", "D"))])
            groups[name] = {
                site: tuple(frozenset(group) & heavy for group in leaving)
                for site, leaving in get_leaving_atom_groups(template).items()
            }
        ri = int(np.searchsorted(starts, index, side="right") - 1)
        if ri not in present:
            present[ri] = set(atom_array.atom_name[starts[ri] : starts[ri + 1]])
        removed = frozenset(
            missing
            for group in groups[name].get(atom, ())
            if group.isdisjoint(present[ri])
            for missing in group
        )
        key = (name, atom)
        if result.setdefault(key, removed) != removed:
            raise ValueError(f"Incompatible declared leaving groups at {key}")
    return result


def _displaced_atoms(residue_type, atom, chemdb, n_hydrogens, heavy_leaving=()):
    present = _hydrogens_on(residue_type, atom, chemdb)
    n_leaving = 1 if n_hydrogens is None else len(present) - n_hydrogens
    if n_leaving < 0:
        raise ValueError(f"Attachment adds hydrogens at {residue_type.name}.{atom}")
    removed = list(leaving_atoms(residue_type, atom, chemdb, n_leaving) or ())
    heavy = set(heavy_leaving) & {a.name for a in residue_type.atoms}
    for name in sorted(heavy):
        removed.append(name)
        removed.extend(_hydrogens_on(residue_type, name, chemdb))
    return tuple(dict.fromkeys(removed))


def _icoor_for(residue_type, name):
    return next((ic for ic in residue_type.icoors if ic.name == name), None)


def _open_valence_frame(residue_type, atom, chemdb, distance):
    """A connection direction from the prepared geometry when no H leaves.

    Two neighbors or a pyramidal three-neighbor center define an open vertex.
    A planar three-neighbor center has no unique side to attach to. The local
    conjugate correction installs the generator's bonded-state length and
    angle targets, just as it does for connections inherited from hydrogen.
    """
    from tmol.ligand._fragmentation import _full_ideal_coords, _angle, _dihedral

    element = _element_for_atom(residue_type, chemdb)
    neighbors = sorted(
        {
            other
            for first, second, *_ in residue_type.bonds
            for this, other in ((first, second), (second, first))
            if this == atom
        },
        key=lambda name: (element[name] == "H", name),
    )
    if len(neighbors) not in (2, 3):
        raise ValueError(
            f"No unambiguous open attachment frame at {residue_type.name}.{atom}"
        )
    xyz = _full_ideal_coords(residue_type)
    vectors = np.stack([xyz[n] - xyz[atom] for n in neighbors])
    lengths = np.linalg.norm(vectors, axis=-1)
    if not np.isfinite(lengths).all() or (lengths < 1e-8).any():
        raise ValueError(f"Degenerate attachment frame at {residue_type.name}.{atom}")
    unit = vectors / lengths[:, None]
    direction = -unit.sum(axis=0)
    norm = np.linalg.norm(direction)
    if (
        norm < 1e-6
        or np.linalg.norm(np.cross(unit[0], unit[1])) < 1e-6
        or (len(neighbors) == 3 and abs(np.linalg.det(unit)) < 1e-4)
    ):
        raise ValueError(
            f"No unoccupied attachment direction at {residue_type.name}.{atom}"
        )
    gp, ggp = neighbors[:2]
    remote = xyz[atom] + direction / norm
    return Icoor(
        name=connection_name(atom),
        parent=atom,
        grand_parent=gp,
        great_grand_parent=ggp,
        phi=-_dihedral(remote, xyz[atom], xyz[gp], xyz[ggp]),
        theta=np.pi - _angle(remote, xyz[atom], xyz[gp]),
        d=float(distance if distance is not None else lengths.mean()),
    )


def _linkage_torsion(name, frame, atom, chi_name, across_connection=False):
    """A bonded attachment torsion, with a connection-spanning axis if needed.

    A ring bond cannot turn independently to reposition its attachment. At
    such sites, sample the new bond instead. The group packer owns this chi.
    """
    if across_connection:
        # At an anomeric/ring carbon the departing H's axis is a ring bond.
        # Move the attached group about the new bond, preserving the ring.
        torsion = Torsion(
            name=chi_name,
            a=UnresolvedAtom(atom=frame.grand_parent),
            b=UnresolvedAtom(atom=atom),
            c=UnresolvedAtom(connection=name, bond_sep_from_conn=0),
            d=UnresolvedAtom(connection=name, bond_sep_from_conn=1),
        )
    else:
        torsion = Torsion(
            name=chi_name,
            a=UnresolvedAtom(atom=frame.great_grand_parent),
            b=UnresolvedAtom(atom=frame.grand_parent),
            c=UnresolvedAtom(atom=atom),
            d=UnresolvedAtom(connection=name, bond_sep_from_conn=0),
        )
    return (
        torsion,
        ChiSamples(
            chi_dihedral=chi_name,
            samples=LINKAGE_SAMPLES,
            expansions=(),
            is_proton=False,
        ),
    )


def conjugation_patch(
    residue_type,
    atom: str,
    chemdb,
    distance=None,
    chi_name=None,
    n_hydrogens=None,
    bond_type="SINGLE",
    heavy_leaving=(),
):
    """Add an attachment, removing displaced H and declared heavy leaving groups.

    The connection inherits a departing atom's internal coordinates, which
    already point along the bond; only the length differs, and the caller
    supplies it where the input measured one.

    ``n_hydrogens`` is how many the site keeps once bonded, as the conjugated
    molecule itself reports it. Without it one hydrogen goes, which is right
    wherever the bond does not change what the site is; acylating an amine
    does change it, and leaving the spare hydrogen behind puts an atom where
    the partner already is.
    """
    leaving_all = _displaced_atoms(
        residue_type, atom, chemdb, n_hydrogens, heavy_leaving
    )
    if not leaving_all and n_hydrogens is None:
        return None
    # Inherit a departing atom's frame whose references remain after removal.
    # Hydrogens may themselves be framed on other departing hydrogens.
    gone = set(leaving_all)

    def _survives(name):
        ic = _icoor_for(residue_type, name)
        return ic is not None and not (
            {ic.parent, ic.grand_parent, ic.great_grand_parent} & gone
        )

    leaving = next(
        (name for name in reversed(leaving_all) if _survives(name)),
        leaving_all[-1] if leaving_all else None,
    )
    frame = (
        _icoor_for(residue_type, leaving)
        if leaving is not None
        else _open_valence_frame(residue_type, atom, chemdb, distance)
    )
    if frame is None:
        return None

    name = connection_name(atom)
    # a hydroxyl that loses its proton to a bond is an ether; the database
    #    says what each type becomes, or says nothing and it is left alone
    conjugated = {t.name: t.conjugated_type for t in chemdb.atom_types}
    current = next(a.atom_type for a in residue_type.atoms if a.name == atom)
    becomes = conjugated.get(current)
    modify_atoms = (Atom(name=f"<{atom}>", atom_type=becomes),) if becomes else ()
    # Inherit a departing atom's frame, or supply an open-valence frame.
    icoors = (
        (
            IcoorVariant(name=name, source=f"<{leaving}>", d=distance)
            if leaving is not None
            else IcoorVariant(
                name=name,
                phi=frame.phi,
                theta=frame.theta,
                d=frame.d,
                parent=f"<{frame.parent}>",
                grand_parent=f"<{frame.grand_parent}>",
                great_grand_parent=f"<{frame.great_grand_parent}>",
            )
        ),
    )

    torsions, chi_samples = (), ()
    if chi_name is not None and frame.grand_parent and frame.great_grand_parent:
        graph = networkx.Graph(
            (a, b, {"order": order}) for a, b, order, *_ in residue_type.bonds
        )
        graph.remove_nodes_from(gone)
        element = _element_for_atom(residue_type, chemdb)
        neighbors = sorted(n for n in graph[atom] if element[n] != "H")
        if neighbors:
            b = frame.grand_parent if frame.grand_parent in neighbors else neighbors[0]
            references = sorted(
                (n for n in graph[b] if n != atom),
                key=lambda n: (element[n] == "H", n),
            )
            bridges = {frozenset(edge) for edge in networkx.bridges(graph)}
            across = (
                graph[b][atom]["order"] != "SINGLE"
                or frozenset((b, atom)) not in bridges
                or not references
            )
            # An icoor may refer to another hydrogen on the same centre.
            # Torsion samples need a bonded four-atom path instead.
            a = (
                frame.great_grand_parent
                if frame.great_grand_parent in references
                else references[0] if references else b
            )
            torsion_frame = attr.evolve(frame, grand_parent=b, great_grand_parent=a)
            # Multiple attachment bonds are not freely rotatable. A chi on
            # the local single bond remains valid when its frame is available.
            if not across or bond_type == "SINGLE":
                torsion, sample = _linkage_torsion(
                    name, torsion_frame, atom, chi_name, across_connection=across
                )
                torsions, chi_samples = (torsion,), (sample,)

    return VariantType(
        name=f"{CONNECTION_PREFIX}{residue_type.base_name}_{atom}",
        display_name=name,
        pattern="",
        remove_atoms=tuple(f"<{name}>" for name in leaving_all),
        add_atoms=(),
        add_atom_aliases=(),
        modify_atoms=modify_atoms,
        add_connections=(Connection(name=name, atom=f"<{atom}>", type=bond_type),),
        add_bonds=(),
        icoors=icoors,
        add_torsions=torsions,
        add_chi_samples=chi_samples,
        applies_to=VariantScope(base_names=(residue_type.base_name,)),
    )


def conjugation_patches(
    residue_type,
    atoms,
    chemdb,
    distances=None,
    hydrogens=None,
    bond_types=None,
    heavy_leaving=None,
):
    """One patch per attachment site, using the supplied bonded hydrogen count.

    Each site's torsion gets a chi number past every one the residue already
    uses, and past the other sites': two patches can apply at once, and a
    number claimed twice would name one torsion twice.
    """
    distances = distances or {}
    taken = {torsion.name for torsion in residue_type.torsions}
    patches = []
    for atom in sorted(atoms):
        number = 1
        while f"chi{number}" in taken:
            number += 1
        chi_name = f"chi{number}"
        taken.add(chi_name)
        patch = conjugation_patch(
            residue_type,
            atom,
            chemdb,
            distances.get(atom),
            chi_name,
            (hydrogens or {}).get(atom),
            (bond_types or {}).get(atom, "SINGLE"),
            (heavy_leaving or {}).get(atom, ()),
        )
        if patch is not None:
            patches.append(patch)
    return tuple(patches)


def with_conjugation_patches(chemdb, residue_type, atoms, distances=None):
    """``chemdb`` extended with the patches ``residue_type`` needs."""
    patches = conjugation_patches(residue_type, atoms, chemdb, distances)
    if not patches:
        return chemdb
    return attr.evolve(chemdb, variants=(*chemdb.variants, *patches))


def conjugation_charge_entries(
    residue_type, atoms, chemdb, base_charges, hydrogens=None, heavy_leaving=None
):
    """``{variant name: {atom: charge}}`` for a residue's conjugation patches.

    The attachment atom takes the charge of every atom that left, so the
    residue keeps the net charge it had. Nothing else moves: every other atom
    falls back to the unpatched entry, and only one atom per patch needs one.

    A generated component's charges do not sum to an integer to begin with --
    they are MMFF94's -- and conserving the total keeps it that way rather than
    imposing a round number the chemistry does not have.
    """
    entries = {}
    for atom in sorted(atoms):
        leaving = _displaced_atoms(
            residue_type,
            atom,
            chemdb,
            (hydrogens or {}).get(atom),
            (heavy_leaving or {}).get(atom, ()),
        )
        if not leaving:
            continue
        if atom not in base_charges or any(
            name not in base_charges for name in leaving
        ):
            continue
        variant = f"{residue_type.name}:{connection_name(atom)}"
        entries[variant] = {
            atom: base_charges[atom] + sum(base_charges[n] for n in leaving)
        }
    return entries


def charges_for_database_residue(param_db, res_name):
    """The unpatched partial charges of a residue already in the database."""
    return {
        entry.atom: entry.charge
        for entry in param_db.scoring.elec.atom_charge_parameters
        if entry.res == res_name
    }
