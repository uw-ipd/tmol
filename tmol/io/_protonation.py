"""Hydrogens for structure inputs, placed by AtomWorks."""

from typing import Collection, Mapping, Sequence

import biotite.structure as struc
import numpy
from atomworks.experimental.protonation import ensure_hydrogens
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.ccd import add_annotations_from_ccd, custom_ccd_residues
from rdkit import Chem

from tmol.io._cif import _FORMAL_CHARGE_SPECIFIED

SOURCE_INDEX = "tmol_source_index"
_HYDROGEN = ("H", "D")
_WATER = ("HOH", "DOD", "WAT")
_CARRIES_HYDROGEN = ("C", "N", "O", "S", "P", "B", "SE")


def hydrogen_names_by_parent(
    parents: list[tuple[str, str, int]], taken: set[str]
) -> list[list[str]]:
    """Names for the hydrogens of each ``(parent name, parent element, count)``.

    A lone hydrogen on X is HX and several are HX1, HX2, ..., so a hydroxyl
    hydrogen carries its oxygen's name (HO2 on O2). Names are kept to four
    characters: past that the parent's element is dropped (H101 for a hydrogen
    of C10), and a name still too long or already in ``taken`` falls back to
    H<element><count>. ``taken`` gains every name given.
    """

    def fallback(element):
        count = 1
        while f"H{element}{count}" in taken:
            count += 1
        return f"H{element}{count}"

    out = []
    for name, element, count in parents:
        element = element.upper()
        stems = [name]
        if name.upper().startswith(element) and len(name) > len(element):
            stems.append(name[len(element) :])
        for stem in stems:
            names = (
                [f"H{stem}"]
                if count == 1
                else [f"H{stem}{i}" for i in range(1, count + 1)]
            )
            if all(len(n) <= 4 and n not in taken for n in names):
                break
        else:
            names = []
            for _ in range(count):
                names.append(fallback(element))
                taken.add(names[-1])
        taken.update(names)
        out.append(names)
    return out


def _names_outside_database(template, parent, names, residue_of):
    """``names`` with the hydrogens of residues the default database lacks renamed.

    Those residues get generated types, whose hydrogens are named after their
    parents; AtomWorks names them after its dictionary, when it has the residue.
    """
    from tmol.io._pose_stack_from_biotite import canonical_ordering_for_biotite

    known = set(canonical_ordering_for_biotite().restype_io_equiv_classes)
    names = names.astype(object)
    outside = ~numpy.isin(template.res_name[parent], list(known))
    for residue in numpy.unique(residue_of[parent[outside]]):
        mine = numpy.flatnonzero(outside & (residue_of[parent] == residue))
        taken = {
            str(n)
            for n, e in zip(
                template.atom_name[residue_of == residue],
                numpy.char.upper(template.element[residue_of == residue].astype(str)),
            )
            if e not in _HYDROGEN
        }
        parents = list(dict.fromkeys(parent[mine].tolist()))
        given = hydrogen_names_by_parent(
            [
                (
                    str(template.atom_name[p]),
                    str(template.element[p]),
                    int((parent[mine] == p).sum()),
                )
                for p in parents
            ],
            taken,
        )
        for p, new in zip(parents, given):
            names[mine[parent[mine] == p]] = new
    return names.astype(str)


Handedness = tuple[str, str, str, str, str, str]
_SENSES = ("+", "-", "cis", "trans")


def _named_by_handedness(template, residue_of, parent, names, coords, handedness):
    """``names`` with the hydrogens swapped that sit on the other side of a centre.

    Each ``(centre, a, b, hydrogen, other, sense)`` of a residue's name says
    where ``hydrogen`` sits: for ``sense`` "+" or "-", the way ``a``, ``b`` and
    it turn about ``centre``; for "cis" or "trans", its side of the ``a`` to
    ``centre`` bond relative to ``b``, bonded to ``a``. A placed ``hydrogen``
    that sits elsewhere trades names with its ``other``.
    """
    res_name = template.res_name[numpy.unique(residue_of, return_index=True)[1]]
    h_residue = residue_of[parent]
    records = [
        (r, record)
        for r in numpy.unique(h_residue).tolist()
        for record in handedness.get(str(res_name[r]), ())
    ]
    if not records:
        return names
    n_atoms = len(template)
    heavy = numpy.flatnonzero(
        numpy.isin(residue_of, [r for r, _ in records])
        & ~numpy.isin(numpy.char.upper(template.element.astype(str)), _HYDROGEN)
    )
    index = dict(
        zip(zip(residue_of[heavy].tolist(), template.atom_name[heavy].tolist()), heavy)
    )
    index.update(
        zip(zip(h_residue.tolist(), names.tolist()), n_atoms + numpy.arange(len(names)))
    )
    rows = []
    for r, (*atoms, sense) in records:
        found = [int(index.get((r, a), -1)) for a in atoms]
        if min(found[:3]) >= 0 and min(found[3:]) >= n_atoms:
            rows.append((*found, _SENSES.index(sense)))
    if not rows:
        return names
    rows = numpy.array(rows, dtype=numpy.int64)
    xyz = numpy.concatenate([template.coord, coords]).astype(numpy.float64)
    x, a, b, h = (xyz[rows[:, k]] for k in range(4))
    turn = numpy.linalg.det(numpy.stack([a - x, b - x, h - x], axis=1))
    axis = (a - x) / numpy.linalg.norm(a - x, axis=-1, keepdims=True)

    def across(v):
        return v - (v * axis).sum(-1, keepdims=True) * axis

    side = (across(h - x) * across(b - a)).sum(-1)
    measured = numpy.where(rows[:, 5] < 2, turn, side) > 0
    swap = rows[measured != (rows[:, 5] % 2 == 0)]
    names = names.copy()
    h, other = swap[:, 3] - n_atoms, swap[:, 4] - n_atoms
    names[h], names[other] = names[other], names[h].copy()
    return names


def _template(structure):
    return structure[0] if isinstance(structure, struc.AtomArrayStack) else structure


def residues_lacking_hydrogens(
    structure: struc.AtomArray | struc.AtomArrayStack,
    residue_names: Collection[str] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Residue starts and which residues have no resolved hydrogen to read.

    A residue counts when it is not water, has an element that bonds to
    hydrogen, is named in ``residue_names`` (every residue when ``None``), and
    has no hydrogen at finite coordinates.
    """
    template = _template(structure)
    starts = struc.get_residue_starts(template, add_exclusive_stop=True)
    if len(template) == 0:
        return starts, numpy.zeros(0, dtype=bool)
    element = numpy.char.upper(template.element.astype(str))
    is_h = numpy.isin(element, _HYDROGEN)
    resolved_h = is_h & numpy.isfinite(template.coord).all(axis=-1)
    carrier = numpy.isin(element, _CARRIES_HYDROGEN)
    first = starts[:-1]
    has_h = numpy.logical_or.reduceat(resolved_h, first)
    has_carrier = numpy.logical_or.reduceat(carrier, first)
    names = template.res_name[first]
    lacking = ~has_h & has_carrier & ~numpy.isin(names, _WATER)
    if residue_names is not None:
        lacking &= numpy.isin(names, list(residue_names))
    return starts, lacking


def with_atomworks_hydrogens(
    structure: struc.AtomArray | struc.AtomArrayStack,
    *,
    ph: float = 7.4,
    residue_names: Collection[str] | None = None,
    coordination: numpy.ndarray | None = None,
    backbone: Mapping[str, tuple[str | None, str | None]] | None = None,
    handedness: Mapping[str, Sequence[Handedness]] | None = None,
) -> struc.AtomArray | struc.AtomArrayStack:
    """``structure`` with AtomWorks hydrogens on every residue that has none.

    AtomWorks ``ensure_hydrogens`` protonates those residues, and the residues
    they bond to for context, once per model at ``ph``. Their heavy atoms take
    its formal charges; the hydrogens it places follow each residue's heavy
    atoms. Residues with resolved hydrogens keep theirs. Unresolved heavy atoms
    get no hydrogens; whatever later rebuilds them builds their hydrogens.
    A structure without a bond table is bonded by residue name for AtomWorks
    and returned without one.

    Args:
        structure: Input AtomArray or AtomArrayStack.
        ph: pH AtomWorks protonates at.
        residue_names: Only residues with these names are protonated; ``None``
            protonates every residue lacking hydrogens.
        coordination: ``[n, 2]`` (metal, donor) atom indices AtomWorks is to
            treat as coordination bonds besides those the structure declares.
        backbone: ``(upper, lower)`` polymer connection atoms by residue name,
            through which consecutive residues of a chain are bonded for
            AtomWorks where the structure leaves them unbonded.
        handedness: ``(centre, a, b, hydrogen, other, sense)`` by residue
            name, where each placed ``hydrogen`` of a centre sits, so that the
            names of hydrogens of one parent follow the residue's types rather
            than AtomWorks' dictionary.

    Returns:
        ``structure`` itself when nothing lacks hydrogens, else a new structure.

    Raises:
        ValueError: If the models of a stack are protonated differently.
    """
    starts, lacking = residues_lacking_hydrogens(structure, residue_names)
    if not lacking.any():
        return structure
    bonds_given = _template(structure).bonds is not None
    if not bonds_given:
        structure = structure.copy()
        structure.bonds = struc.connect_via_residue_names(_template(structure))
    template = _template(structure)
    n_atoms = len(template)
    residue_of = numpy.repeat(numpy.arange(len(lacking)), numpy.diff(starts))
    element = numpy.char.upper(template.element.astype(str))
    is_h = numpy.isin(element, _HYDROGEN)

    bonds = template.bonds.as_array()
    if coordination is None:
        coordination = numpy.zeros((0, 2), dtype=numpy.int64)
    extra = numpy.concatenate(
        [
            numpy.column_stack(
                [
                    coordination,
                    numpy.full(len(coordination), struc.BondType.COORDINATION),
                ]
            ),
            _polymer_gap_links(template, starts, backbone or {}),
        ]
    ).astype(numpy.int64)
    pairs = residue_of[numpy.concatenate([bonds[:, :2], extra[:, :2]])]
    selected = lacking.copy()
    for a, b in (pairs.T, pairs.T[::-1]):
        selected[b[lacking[a]]] = True
    selected &= ~numpy.isin(template.res_name[starts[:-1]], _WATER)
    heavy_mask = selected[residue_of] & ~is_h

    models = (
        [structure[i] for i in range(structure.stack_depth())]
        if isinstance(structure, struc.AtomArrayStack)
        else [structure]
    )
    placed = [_placed_hydrogens(model, heavy_mask, ph, extra) for model in models]
    parent, names, charge = placed[0][:3]
    if any(
        not numpy.array_equal(p[0], parent) or not numpy.array_equal(p[1], names)
        for p in placed[1:]
    ):
        raise ValueError("AtomWorks protonates the models of this stack differently")
    keep_h = lacking[residue_of[parent]]
    parent, names = parent[keep_h], names[keep_h]
    names = _names_outside_database(template, parent, names, residue_of)
    coords = numpy.stack([p[3][keep_h] for p in placed])
    names = _named_by_handedness(
        template, residue_of, parent, names, coords[0], handedness or {}
    )

    categories = template.get_annotation_categories()
    charges = (
        template.charge.copy()
        if "charge" in categories
        else numpy.zeros(n_atoms, dtype=numpy.int8)
    )
    updated = lacking[residue_of[charge[0]]]
    charges[charge[0][updated]] = charge[1][updated]

    registry = getattr(structure, "_custom_ccd_registry", None)
    # the lacking residues' own unresolved hydrogens are replaced
    kept = ~(is_h & lacking[residue_of])
    base = structure.copy()
    base.set_annotation("charge", charges.astype(numpy.int8))
    if "charge" not in categories and _FORMAL_CHARGE_SPECIFIED not in categories:
        base.set_annotation(_FORMAL_CHARGE_SPECIFIED, lacking[residue_of])
    # a bond table cannot be indexed with repeated atoms
    base.bonds = None
    if isinstance(base, struc.AtomArrayStack):
        hydrogens = base[:, parent]
        hydrogens.coord = coords
    else:
        hydrogens = base[parent]
        hydrogens.coord = coords[0]
    hydrogens.bonds = struc.BondList(len(parent))
    base.bonds = struc.BondList(n_atoms)
    hydrogens.atom_name = names
    hydrogens.element[:] = "H"
    hydrogens.charge[:] = 0
    merged = (
        struc.concatenate([base, hydrogens])
        if isinstance(base, struc.AtomArrayStack)
        else concatenate_any([base, hydrogens])
    )
    total = n_atoms + len(parent)
    merged.bonds = struc.BondList(
        total,
        numpy.concatenate(
            [
                bonds,
                numpy.stack(
                    [
                        parent,
                        n_atoms + numpy.arange(len(parent)),
                        numpy.full(len(parent), int(struc.BondType.SINGLE)),
                    ],
                    axis=1,
                ),
            ]
        ).astype(numpy.uint32),
    )
    residue_key = numpy.concatenate([residue_of, residue_of[parent]])
    is_new = numpy.concatenate([numpy.zeros(n_atoms), numpy.ones(len(parent))])
    keep = numpy.concatenate([kept, numpy.ones(len(parent), dtype=bool)])
    order = numpy.lexsort((numpy.arange(total), is_new, residue_key))
    order = order[keep[order]]
    merged = (
        merged[:, order] if isinstance(merged, struc.AtomArrayStack) else merged[order]
    )
    if not bonds_given:
        merged.bonds = None
    if registry is not None:
        merged._custom_ccd_registry = registry
    return merged


_CONVENTIONAL_BACKBONES = (("CA", "C", "N"), ("C4'", "O3'", "P"))


def _polymer_gap_links(
    template: struc.AtomArray,
    starts: numpy.ndarray,
    backbone: Mapping[str, tuple[str | None, str | None]],
):
    """``[n, 3]`` polymer bonds across the gaps of each chain, as single bonds.

    tmol connects consecutive residues of a chain through the upper atom of
    the first and the lower atom of the second, gap or not; AtomWorks sees the
    chemistry that gives them only when they are bonded. ``backbone`` gives
    those ``(upper, lower)`` atoms by residue name; a residue it does not name
    uses C/N when it has a CA and O3'/P when it has a C4'. Atoms already bonded
    to another residue get none.
    """
    n_res = len(starts) - 1
    if n_res < 2:
        return numpy.zeros((0, 3), dtype=numpy.int64)
    residue_of = numpy.repeat(numpy.arange(n_res), numpy.diff(starts))
    names = template.atom_name
    res_name = template.res_name[starts[:-1]]
    chain = template.chain_id[starts[:-1]]
    bonds = template.bonds.as_array()[:, :2].astype(numpy.int64)
    linked = numpy.zeros(len(template), dtype=bool)
    across = residue_of[bonds[:, 0]] != residue_of[bonds[:, 1]]
    linked[bonds[across].ravel()] = True

    upper = numpy.full(n_res, "", dtype=object)
    lower = numpy.full(n_res, "", dtype=object)
    for marker, up, down in _CONVENTIONAL_BACKBONES[::-1]:
        has = numpy.zeros(n_res, dtype=bool)
        has[residue_of[names == marker]] = True
        upper[has], lower[has] = up, down
    for name in set(res_name.tolist()) & set(backbone):
        up, down = backbone[name]
        upper[res_name == name], lower[res_name == name] = up or "", down or ""

    def atom_by_residue(wanted):
        out = numpy.full(n_res, -1, dtype=numpy.int64)
        found = numpy.flatnonzero(names == wanted.astype(str)[residue_of])
        out[residue_of[found[::-1]]] = found[::-1]
        return out

    u, d = atom_by_residue(upper)[:-1], atom_by_residue(lower)[1:]
    gap = (chain[:-1] == chain[1:]) & (u >= 0) & (d >= 0)
    gap &= ~linked[u] & ~linked[d]
    return numpy.column_stack(
        [u[gap], d[gap], numpy.full(int(gap.sum()), int(struc.BondType.SINGLE))]
    ).astype(numpy.int64)


def _with_chelates_covalent(
    sub: struc.AtomArray,
) -> tuple[struc.AtomArray, numpy.ndarray]:
    """``sub`` with each split chelated metal covalently bonded to its component.

    AtomWorks keeps the dictionary's hydrogen count on atoms covalently bonded to
    a metal, which is how a component draws its chelate (a heme's bare pyrroles).
    Also returns the ``[n, 2]`` (atom, atom) pairs of those bonds.
    """
    from tmol.io._pose_stack_from_biotite import METAL_ORIGIN

    none = numpy.zeros((0, 2), dtype=numpy.int64)
    if METAL_ORIGIN not in sub.get_annotation_categories():
        return sub, none
    origin = sub.get_annotation(METAL_ORIGIN).astype(str)
    if not numpy.char.str_len(origin).any():
        return sub, none
    ins = (
        sub.ins_code.astype(str)
        if "ins_code" in sub.get_annotation_categories()
        else numpy.full(len(sub), "")
    )
    residue = numpy.char.add(
        numpy.char.add(numpy.char.add(sub.res_id.astype(str), "\t"), ins),
        numpy.char.add("\t", sub.res_name.astype(str)),
    )
    home = numpy.array(["\t".join(o.split("\t")[:3]) for o in origin])
    bonds = sub.bonds.as_array()
    a, b = bonds[:, 0].astype(int), bonds[:, 1].astype(int)
    same_chain = sub.chain_id[a] == sub.chain_id[b]
    chelate = same_chain & (
        ((numpy.char.str_len(origin[a]) > 0) & (home[a] == residue[b]))
        | ((numpy.char.str_len(origin[b]) > 0) & (home[b] == residue[a]))
    )
    if not chelate.any():
        return sub, none
    bonds[chelate, 2] = struc.BondType.SINGLE
    sub = sub.copy()
    sub.bonds = struc.BondList(len(sub), bonds)
    return sub, bonds[chelate, :2].astype(numpy.int64)


_BOND_ORDER = {
    struc.BondType.SINGLE: 1,
    struc.BondType.DOUBLE: 2,
    struc.BondType.TRIPLE: 3,
    struc.BondType.AROMATIC_SINGLE: 1,
    struc.BondType.AROMATIC_DOUBLE: 2,
    struc.BondType.AROMATIC_TRIPLE: 3,
}


def _charges_without_chelates(out: struc.AtomArray, chelates: numpy.ndarray):
    """Formal charges of ``out`` once its chelate bonds are coordination again.

    A donor keeps the electrons of the bond: one left short of its default
    valence by its remaining bonds and hydrogens is an anion by that much.
    """
    charge = out.charge.copy()
    if not len(chelates):
        return charge
    table = Chem.GetPeriodicTable()
    bonds = out.bonds.as_array()
    chelate = numpy.isin(
        numpy.minimum(bonds[:, 0], bonds[:, 1]) * len(out)
        + numpy.maximum(bonds[:, 0], bonds[:, 1]),
        chelates.min(axis=1) * len(out) + chelates.max(axis=1),
    )
    for donor in numpy.unique(chelates):
        rows = bonds[((bonds[:, 0] == donor) | (bonds[:, 1] == donor)) & ~chelate]
        if not len(rows) or not all(int(t) in _BOND_ORDER for t in rows[:, 2]):
            continue
        element = str(out.element[donor]).capitalize()
        try:
            valence = table.GetDefaultValence(element)
        except RuntimeError:
            continue
        if valence <= 0 or element in ("C", "H"):
            continue
        short = valence - sum(_BOND_ORDER[int(t)] for t in rows[:, 2])
        if short > 0 and charge[donor] == 0:
            charge[donor] = -short
    return charge


def _pn_units(sub: struc.AtomArray) -> numpy.ndarray:
    """Polymer/non-polymer unit of each atom: a chain's polymer, or bonded ligands.

    AtomWorks reads a polymer residue bonded to another unit as covalently
    modified, so a ligand sharing its chain's label must not share its unit.
    """
    chain = sub.chain_id.astype(str)
    if "is_polymer" not in sub.get_annotation_categories():
        return chain.copy()
    starts = struc.get_residue_starts(sub, add_exclusive_stop=True)
    residue_of = numpy.repeat(numpy.arange(len(starts) - 1), numpy.diff(starts))
    ligand = ~sub.is_polymer.astype(bool)
    root = numpy.arange(len(starts) - 1)

    def find(r):
        while root[r] != r:
            root[r] = root[root[r]]
            r = root[r]
        return r

    pairs = sub.bonds.as_array()[:, :2].astype(int)
    a, b = pairs[:, 0], pairs[:, 1]
    linked = ligand[a] & ligand[b] & (chain[a] == chain[b])
    for ra, rb in zip(residue_of[a[linked]], residue_of[b[linked]]):
        root[find(ra)] = find(rb)
    unit = numpy.array([find(r) for r in range(len(root))])[residue_of]
    return numpy.where(
        ligand, numpy.char.add(numpy.char.add(chain, "/"), unit.astype(str)), chain
    )


def _in_chain_order(model: struc.AtomArray, atoms: numpy.ndarray) -> numpy.ndarray:
    """``atoms`` with residues ordered by chain, then residue number and code.

    AtomWorks builds a backbone hydrogen against the bonded residue earlier in
    the array, so residues must run along their chains.
    """
    starts = struc.get_residue_starts(model, add_exclusive_stop=True)
    residue = numpy.repeat(numpy.arange(len(starts) - 1), numpy.diff(starts))[atoms]
    order = numpy.lexsort(
        (
            atoms,
            residue,
            model.ins_code[atoms],
            model.res_id[atoms],
            model.chain_id[atoms],
        )
    )
    return atoms[order]


def _placed_hydrogens(
    model: struc.AtomArray,
    heavy_mask: numpy.ndarray,
    ph: float,
    extra_bonds: numpy.ndarray,
):
    """AtomWorks protonation of the ``heavy_mask`` atoms of one model.

    Returns each placed hydrogen's parent (as an index into ``model``), name and
    coordinates, and the ``(index, charge)`` of every protonated heavy atom.
    """
    from tmol.io._pose_stack_from_biotite import _with_metal_coordination_typed

    source = _in_chain_order(model, numpy.flatnonzero(heavy_mask))
    sub = model[source]
    position = numpy.full(len(model), -1)
    position[source] = numpy.arange(len(source))
    for a, b, bond_type in extra_bonds:
        if position[a] >= 0 and position[b] >= 0:
            sub.bonds.add_bond(position[a], position[b], struc.BondType(bond_type))
    sub, chelates = _with_chelates_covalent(_with_metal_coordination_typed(sub))
    sub.set_annotation(SOURCE_INDEX, source)
    categories = sub.get_annotation_categories()
    if "charge" not in categories:
        sub.set_annotation("charge", numpy.zeros(len(sub), dtype=numpy.int8))
    if "pn_unit_iid" not in categories:
        sub.set_annotation("pn_unit_iid", _pn_units(sub))
    if "pn_unit_id" not in categories:
        sub.set_annotation("pn_unit_id", sub.pn_unit_iid.copy())
    registry = getattr(model, "_custom_ccd_registry", None) or {}
    with custom_ccd_residues(registry):
        sub = add_annotations_from_ccd(
            sub,
            annotations=["nhyd"],
            overwrite=True,
            hydrogen_policy="remove",
            ccd_mirror_path=None,
        )
        out = ensure_hydrogens(
            sub,
            ph=ph,
            silence_rdkit_warnings=True,
            silence_lost_annotation_warnings=True,
        )
    index = out.get_annotation(SOURCE_INDEX).astype(numpy.int64)
    is_h = numpy.isin(numpy.char.upper(out.element.astype(str)), _HYDROGEN)
    out_of_source = numpy.full(len(model), -1)
    out_of_source[index[~is_h]] = numpy.flatnonzero(~is_h)
    charge = _charges_without_chelates(out, out_of_source[source[chelates]])
    return (
        index[is_h],
        out.atom_name[is_h].astype(str),
        (index[~is_h], charge[~is_h]),
        out.coord[is_h],
    )
