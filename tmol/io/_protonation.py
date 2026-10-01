"""Protonation of structure inputs: AtomWorks decides it, tmol builds the hydrogens."""

from collections import Counter, defaultdict
from typing import Collection, Mapping

import biotite.structure as struc
import numpy
from atomworks.constants import METAL_ELEMENTS
from atomworks.experimental.protonation import (
    assign_hydrogens,
    find_disulfides,
    hydrogen_plan,
    place_hydrogens,
)
from atomworks.experimental.protonation.geometry import names_from_parent
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.ccd import add_annotations_from_ccd, custom_ccd_residues
from rdkit import Chem

from tmol.database.chemical import (
    DEPROTONATED_VAR_IND,
    HIS_UNRESOLVED_VAR_IND,
    l_base_name,
    special_case_variant_index,
)
from tmol.io._cif import _FORMAL_CHARGE_SPECIFIED

# per atom, the res_type_variant its residue's protonation state selects; -1 for none
PROTONATION_VARIANT = "tmol_protonation_variant"
_HYDROGEN = ("H", "D")
_WATER = ("HOH", "DOD", "WAT")
_CARRIES_HYDROGEN = ("C", "N", "O", "S", "P", "B", "SE")

# (heavy atoms whose hydrogen counts tell a class's forms apart,
#    {(their counts, anionic): res_type_variant})
Forms = tuple[tuple[str, ...], dict[tuple[tuple[int, ...], bool], int]]
_FORMS: dict[int, tuple] = {}
# AtomWorks' (hydrogen count, charge) of each heavy atom, by residue context
_STATES: dict[str, dict[str, tuple[int, int]]] = {}


def hydrogen_names_by_parent(
    parents: list[tuple[str, str, int]], taken: set[str]
) -> list[list[str]]:
    """AtomWorks' names for the hydrogens of each ``(parent name, element, count)``.

    Names avoid ``taken``, which gains every name given."""
    out = []
    for name, element, count in parents:
        out.append(names_from_parent(name, element.upper(), count, taken))
        taken.update(out[-1])
    return out


def database_forms(chemdb) -> dict[str, Forms]:
    """The protonation ``Forms`` of each default-database class in ``chemdb``.

    Only plain, histidine and deprotonated forms count; tmol's disulfide search
    decides CYD."""
    cached = _FORMS.get(id(chemdb))
    if cached is not None and cached[0] is chemdb:
        return cached[1]
    from tmol.io._pose_stack_from_biotite import canonical_ordering_for_biotite

    known = set(canonical_ordering_for_biotite().restype_io_equiv_classes)
    element = {t.name: t.element.upper() for t in chemdb.atom_types}
    counts = defaultdict(dict)
    for res in chemdb.residues:
        variant = special_case_variant_index(res)
        if (
            res.io_equiv_class not in known
            or variant == HIS_UNRESOLVED_VAR_IND
            or variant > DEPROTONATED_VAR_IND
            or l_base_name(res) == "CYD"
        ):
            continue
        on = {a.name: 0 for a in res.atoms if element[a.atom_type] not in _HYDROGEN}
        for a, b, *_ in res.bonds:
            for heavy, h in ((a, b), (b, a)):
                if heavy in on and h not in on:
                    on[heavy] += 1
        counts[res.io_equiv_class].setdefault(variant, on)
    out = {}
    for name, by_variant in counts.items():
        atoms = tuple(
            sorted(
                a
                for a in set().union(*by_variant.values())
                if len({on.get(a) for on in by_variant.values()}) > 1
            )
        )
        table = {}
        for variant, on in sorted(by_variant.items()):
            key = (tuple(on.get(a, -1) for a in atoms), variant == DEPROTONATED_VAR_IND)
            table.setdefault(key, variant)
        out[name] = (atoms, table)
    _FORMS[id(chemdb)] = (chemdb, out)
    return out


def _template(structure):
    return structure[0] if isinstance(structure, struc.AtomArrayStack) else structure


def residues_lacking_hydrogens(
    structure: struc.AtomArray | struc.AtomArrayStack,
    residue_names: Collection[str] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Residue starts, and which non-water residues with a hydrogen-bearing element
    (named in ``residue_names``, if given) have no hydrogen at finite coordinates."""
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
    forms: Mapping[str, Forms] | None = None,
) -> struc.AtomArray | struc.AtomArrayStack:
    """``structure`` with AtomWorks' protonation of every residue that has no hydrogen.

    Residues in ``forms`` are only marked ``PROTONATION_VARIANT`` for tmol to build;
    others get AtomWorks' hydrogens and formal charges.

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
        forms: ``database_forms`` of the residues tmol builds itself.

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
    stack = isinstance(structure, struc.AtomArrayStack)
    template = _template(structure)
    n_atoms = len(template)
    residue_of = struc.get_all_residue_positions(template)
    res_name = template.res_name[starts[:-1]]
    is_h = numpy.isin(numpy.char.upper(template.element.astype(str)), _HYDROGEN)
    bonds = template.bonds.as_array().astype(numpy.int64)
    single = int(struc.BondType.SINGLE)
    coordination = numpy.zeros((0, 2)) if coordination is None else coordination
    disulfides = find_disulfides(template)
    extra = numpy.concatenate(
        [
            numpy.c_[
                coordination,
                numpy.full(len(coordination), int(struc.BondType.COORDINATION)),
            ],
            _polymer_gap_links(template, starts, backbone or {}),
            numpy.c_[disulfides, numpy.full(len(disulfides), single)],
        ]
    ).astype(numpy.int64)

    forms = forms or {}
    placing = lacking & ~numpy.isin(res_name, list(forms))
    asked = [r for r in numpy.flatnonzero(lacking & ~placing) if forms[res_name[r]][0]]
    keys = _contexts(template, starts, residue_of, numpy.r_[bonds, extra], asked, ph)
    new = {}
    for r, key in zip(asked, keys):
        if key not in _STATES:
            new.setdefault(key, r)
    call = placing.copy()
    call[list(new.values())] = True
    if call.any():
        pairs = residue_of[numpy.r_[bonds[:, :2], extra[:, :2]]]
        selected = call.copy()
        for a, b in (pairs.T, pairs.T[::-1]):
            selected[b[call[a]]] = True
        selected &= ~numpy.isin(res_name, _WATER)
        models = (
            [structure[i] for i in range(structure.stack_depth())]
            if stack
            else [structure]
        )
        placed = [
            _placed_hydrogens(model, selected[residue_of] & ~is_h, ph, extra)
            for model in models
        ]
        parent, names, charge, _, free = placed[0]
        count = numpy.bincount(parent, minlength=n_atoms)
        count[free] = -1
        state_charge = numpy.zeros(n_atoms, dtype=numpy.int64)
        state_charge[charge[0]] = charge[1]
        for key, r in new.items():
            _STATES[key] = {
                str(template.atom_name[i]): (int(count[i]), int(state_charge[i]))
                for i in range(starts[r], starts[r + 1])
            }

    variant = numpy.full(n_atoms, -1, dtype=numpy.int8)
    for r, key in zip(asked, keys):
        variant[starts[r] : starts[r + 1]] = _variant(forms[res_name[r]], _STATES[key])
    if not placing.any():
        out = structure.copy()
        out.set_annotation(PROTONATION_VARIANT, variant)
        return out if bonds_given else _unbonded(out)

    if any(
        not numpy.array_equal(p[0], parent) or not numpy.array_equal(p[1], names)
        for p in placed[1:]
    ):
        raise ValueError("AtomWorks protonates the models of this stack differently")
    keep_h = placing[residue_of[parent]]
    parent, names = parent[keep_h], names[keep_h]
    names = _names_by_parent(template, parent, names, residue_of)
    coords = numpy.stack([p[3][keep_h] for p in placed])

    categories = template.get_annotation_categories()
    charges = (
        template.charge.copy()
        if "charge" in categories
        else numpy.zeros(n_atoms, dtype=numpy.int8)
    )
    updated = placing[residue_of[charge[0]]]
    charges[charge[0][updated]] = charge[1][updated]

    registry = getattr(structure, "_custom_ccd_registry", None)
    # the placed residues' own unresolved hydrogens are replaced
    kept = ~(is_h & placing[residue_of])
    base = structure.copy()
    base.set_annotation("charge", charges.astype(numpy.int8))
    base.set_annotation(PROTONATION_VARIANT, variant)
    if "charge" not in categories and _FORMAL_CHARGE_SPECIFIED not in categories:
        base.set_annotation(_FORMAL_CHARGE_SPECIFIED, placing[residue_of])
    # a bond table cannot be indexed with repeated atoms
    base.bonds = None
    hydrogens = base[:, parent] if stack else base[parent]
    hydrogens.coord = coords if stack else coords[0]
    hydrogens.bonds = struc.BondList(len(parent))
    base.bonds = struc.BondList(n_atoms)
    hydrogens.atom_name = names
    hydrogens.element[:] = "H"
    hydrogens.charge[:] = 0
    merged = (struc.concatenate if stack else concatenate_any)([base, hydrogens])
    total = n_atoms + len(parent)
    h_bonds = numpy.c_[
        parent, n_atoms + numpy.arange(len(parent)), [single] * len(parent)
    ]
    merged.bonds = struc.BondList(total, numpy.r_[bonds, h_bonds].astype(numpy.uint32))
    residue_key = numpy.r_[residue_of, residue_of[parent]]
    is_new = numpy.r_[numpy.zeros(n_atoms), numpy.ones(len(parent))]
    keep = numpy.r_[kept, numpy.ones(len(parent), dtype=bool)]
    order = numpy.lexsort((numpy.arange(total), is_new, residue_key))
    order = order[keep[order]]
    merged = merged[:, order] if stack else merged[order]
    if registry is not None:
        merged._custom_ccd_registry = registry
    return merged if bonds_given else _unbonded(merged)


def _unbonded(structure):
    structure.bonds = None
    return structure


def _contexts(template, starts, residue_of, bonds, residues, ph) -> list[str]:
    """Per residue, a key of what AtomWorks decides its state from: pH, name, resolved
    atoms' charges, and each bond out (with its length to a metal)."""
    mine = numpy.zeros(len(starts) - 1, dtype=bool)
    mine[residues] = True
    name = template.atom_name.tolist()
    element = template.element.tolist()
    metal = numpy.isin(
        numpy.char.upper(template.element.astype(str)), sorted(METAL_ELEMENTS)
    )
    categories = template.get_annotation_categories()
    charge = template.charge.tolist() if "charge" in categories else [0] * len(name)
    tokens = defaultdict(list)
    resolved = numpy.isfinite(template.coord).all(-1)
    for i in numpy.flatnonzero(mine[residue_of] & resolved).tolist():
        tokens[residue_of[i]].append(f"{name[i]}{charge[i]}")
    across = bonds[residue_of[bonds[:, 0]] != residue_of[bonds[:, 1]]]
    for i, j, k in numpy.r_[across, across[:, [1, 0, 2]]].tolist():
        if mine[residue_of[i]]:
            token = f"{name[i]}>{element[j]}{charge[j]}:{k}"
            if metal[j]:
                length = numpy.linalg.norm(template.coord[i] - template.coord[j])
                token += f"@{float(length)!r}"
            tokens[residue_of[i]].append(token)
    return [
        f"{ph}|{template.res_name[starts[r]]}|{'|'.join(sorted(tokens[r]))}"
        for r in residues
    ]


def _variant(forms: Forms, state: Mapping[str, tuple[int, int]]) -> int:
    """The res_type_variant ``state`` selects among ``forms``; -1 where it leaves it open."""
    atoms, table = forms
    got = [state.get(a, (-1, 0)) for a in atoms]
    if min(n for n, _ in got) < 0:
        return -1
    return table.get((tuple(n for n, _ in got), any(c < 0 for _, c in got)), -1)


def _names_by_parent(template, parent, names, residue_of):
    """``names`` with the placed hydrogens named after their parents, as generated
    types name them (AtomWorks names them after its dictionary)."""
    names = names.astype(object)
    heavy = ~numpy.isin(numpy.char.upper(template.element.astype(str)), _HYDROGEN)
    for residue in numpy.unique(residue_of[parent]):
        mine = numpy.flatnonzero(residue_of[parent] == residue)
        taken = set(template.atom_name[heavy & (residue_of == residue)].tolist())
        counts = Counter(parent[mine].tolist())
        given = hydrogen_names_by_parent(
            [
                (str(template.atom_name[p]), str(template.element[p]), n)
                for p, n in counts.items()
            ],
            taken,
        )
        for p, new in zip(counts, given):
            names[mine[parent[mine] == p]] = new
    return names.astype(str)


_CONVENTIONAL_BACKBONES = (("CA", "C", "N"), ("C4'", "O3'", "P"))


def _polymer_gap_links(
    template: struc.AtomArray,
    starts: numpy.ndarray,
    backbone: Mapping[str, tuple[str | None, str | None]],
):
    """``[n, 3]`` single bonds joining consecutive residues of a chain across its gaps,
    as tmol connects them, by ``backbone`` (upper, lower) atoms, else C/N or O3'/P."""
    n_res = len(starts) - 1
    if n_res < 2:
        return numpy.zeros((0, 3), dtype=numpy.int64)
    residue_of = struc.get_all_residue_positions(template)
    names = template.atom_name
    res_name = template.res_name[starts[:-1]]
    same_chain = numpy.ones(n_res - 1, dtype=bool)
    # entity too: 3T14's FAD 500 follows MET 418 in chain A
    for key in ("chain_id", "label_entity_id", "sym_id"):
        if key in template.get_annotation_categories():
            values = template.get_annotation(key)[starts[:-1]]
            same_chain &= values[1:] == values[:-1]
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
    gap = same_chain & (u >= 0) & (d >= 0)
    gap &= ~linked[u] & ~linked[d]
    return numpy.column_stack(
        [u[gap], d[gap], numpy.full(int(gap.sum()), int(struc.BondType.SINGLE))]
    ).astype(numpy.int64)


def _with_chelates_covalent(
    sub: struc.AtomArray,
) -> tuple[struc.AtomArray, numpy.ndarray]:
    """``sub`` with each split chelated metal bonded covalently to its component, as
    the dictionary draws it (155C heme: bare pyrrole N), and ``[n, 2]`` those bonds."""
    from tmol.io._pose_stack_from_biotite import METAL_ORIGIN

    none = numpy.zeros((0, 2), dtype=numpy.int64)
    if METAL_ORIGIN not in sub.get_annotation_categories():
        return sub, none
    origin = sub.get_annotation(METAL_ORIGIN).astype(str)
    if not numpy.char.str_len(origin).any():
        return sub, none
    ins = sub.ins_code.astype(str)
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


def _charges_without_chelates(
    state: struc.AtomArray, hydrogens: numpy.ndarray, chelates: numpy.ndarray
):
    """Formal charges of ``state`` once its chelate bonds are coordination again:
    a donor short of its default valence is an anion by that much."""
    charge = state.charge.copy()
    if not len(chelates):
        return charge
    table = Chem.GetPeriodicTable()
    bonds = state.bonds.as_array()
    chelate = numpy.isin(
        numpy.minimum(bonds[:, 0], bonds[:, 1]) * len(state)
        + numpy.maximum(bonds[:, 0], bonds[:, 1]),
        chelates.min(axis=1) * len(state) + chelates.max(axis=1),
    )
    for donor in numpy.unique(chelates):
        rows = bonds[((bonds[:, 0] == donor) | (bonds[:, 1] == donor)) & ~chelate]
        if not (len(rows) or hydrogens[donor]) or not all(
            int(t) in _BOND_ORDER for t in rows[:, 2]
        ):
            continue
        element = str(state.element[donor]).capitalize()
        try:
            valence = table.GetDefaultValence(element)
        except RuntimeError:
            continue
        if valence <= 0 or element in ("C", "H"):
            continue
        short = valence - hydrogens[donor]
        short -= sum(_BOND_ORDER[int(t)] for t in rows[:, 2])
        if short > 0 and charge[donor] == 0:
            charge[donor] = -short
    return charge


def _placed_hydrogens(
    model: struc.AtomArray,
    heavy_mask: numpy.ndarray,
    ph: float,
    extra_bonds: numpy.ndarray,
):
    """AtomWorks protonation of the ``heavy_mask`` atoms of one model: hydrogen parents,
    names, (index, charge) of heavy atoms, coordinates, and tautomer-free atoms."""
    from tmol.io._pose_stack_from_biotite import _with_metal_coordination_typed

    source = numpy.flatnonzero(heavy_mask)
    sub = model[source]
    position = numpy.full(len(model), -1)
    position[source] = numpy.arange(len(source))
    for a, b, bond_type in extra_bonds:
        if position[a] >= 0 and position[b] >= 0:
            sub.bonds.add_bond(position[a], position[b], struc.BondType(bond_type))
    sub, chelates = _with_chelates_covalent(_with_metal_coordination_typed(sub))
    categories = sub.get_annotation_categories()
    if "charge" not in categories:
        sub.set_annotation("charge", numpy.zeros(len(sub), dtype=numpy.int8))
    if "pn_unit_iid" not in categories:
        sub.set_annotation("pn_unit_iid", sub.chain_id.astype(str))
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
        # sub has no hydrogens, so the state's atoms are sub's, in order
        state = assign_hydrogens(sub, ph=ph)
        plan = hydrogen_plan(state)
    hydrogens = numpy.bincount(plan.parent, minlength=len(state))
    return (
        source[plan.parent],
        plan.atom_name.astype(str),
        (source, _charges_without_chelates(state, hydrogens, chelates)),
        place_hydrogens(state.coord, plan),
        source[state.tautomer_free.astype(bool)],
    )
