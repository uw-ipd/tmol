"""Protonation of structure inputs: AtomWorks decides it, tmol builds the hydrogens."""

from collections import Counter, OrderedDict, defaultdict
from threading import RLock
from typing import Collection, Mapping

import biotite.structure as struc
import numpy
from atomworks.constants import METAL_ELEMENTS
from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens
from atomworks.experimental.protonation.geometry import names_from_parent
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.ccd import (
    add_annotations_from_ccd,
    custom_ccd_residues,
    get_custom_ccd_entries,
)
from rdkit import Chem
from scipy.spatial import cKDTree

from tmol.database.chemical import (
    DEPROTONATED_VAR_IND,
    HIS_UNRESOLVED_VAR_IND,
    NEUTRAL_TERMINUS_VAR_BASE,
    l_base_name,
    special_case_variant_index,
)
from tmol.io._cif import _FORMAL_CHARGE_SPECIFIED
from tmol.io._input_termini import EXPLICIT_TERMINI
from tmol.utility.weak_identity_cache import WeakIdentityLRU

# per atom, the res_type_variant its residue's protonation state selects; -1 for none
PROTONATION_VARIANT = "tmol_protonation_variant"
PROTONATION_ALTERNATIVES = "tmol_protonation_alternatives"
_HYDROGEN = ("H", "D")
_WATER = ("HOH", "DOD", "WAT")
_CARRIES_HYDROGEN = ("C", "N", "O", "S", "P", "B", "SE")

# (heavy atoms whose hydrogen counts tell a class's forms apart,
#    {their counts: res_type_variant}, {other heavy atoms: the count all forms give},
#    the amine a neutral amino terminus leaves two hydrogens on, or None)
Forms = tuple[tuple[str, ...], dict[tuple[int, ...], int], dict[str, int], str | None]
_FORMS = WeakIdentityLRU()
# AtomWorks' (hydrogen count, charge) of each heavy atom, by residue context
_STATES: OrderedDict[str, dict[str, tuple[int, int]]] = OrderedDict()
_STATE_LOCK = RLock()
_STATE_CAPACITY = 8192


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
    decides CYD. Connection atoms have no fixed count."""
    return _FORMS.get_or_create(chemdb, (), lambda: _database_forms(chemdb))


def _database_forms(chemdb) -> dict[str, Forms]:
    from tmol.io._pose_stack_from_biotite import canonical_ordering_for_biotite

    known = set(canonical_ordering_for_biotite().restype_io_equiv_classes)
    element = {t.name: t.element.upper() for t in chemdb.atom_types}
    counts = defaultdict(dict)
    linked = defaultdict(set)
    amine = {}
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
        linked[res.io_equiv_class].update(c.atom for c in res.connections)
        down = {c.name: c.atom for c in res.connections}.get("down")
        if variant == 0 and res.properties.polymer.backbone_type == "alpha_aa":
            amine.setdefault(res.io_equiv_class, down if on.get(down) == 1 else None)
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
            table.setdefault(tuple(on.get(a, -1) for a in atoms), variant)
        fixed = {
            a: n
            for a, n in next(iter(by_variant.values())).items()
            if a not in atoms and a not in linked[name]
        }
        out[name] = (atoms, table, fixed, amine.get(name))
    return out


def _template(structure):
    return structure[0] if isinstance(structure, struc.AtomArrayStack) else structure


def residues_lacking_hydrogens(
    structure: struc.AtomArray | struc.AtomArrayStack,
    residue_names: Collection[str] | None = None,
    forms: Mapping[str, Forms] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Residue starts, and which non-water residues with a hydrogen-bearing element
    (named in ``residue_names``, if given) have no hydrogen at finite coordinates, or
    are named in ``forms`` with a heavy atom carrying other hydrogens than its forms."""
    template = _template(structure)
    starts = struc.get_residue_starts(template, add_exclusive_stop=True)
    if len(template) == 0:
        return starts, numpy.zeros(0, dtype=bool)
    element = numpy.char.upper(template.element.astype(str))
    resolved = numpy.isfinite(template.coord).all(axis=-1)
    carrier = numpy.isin(element, _CARRIES_HYDROGEN)
    first = starts[:-1]
    has_h = numpy.logical_or.reduceat(numpy.isin(element, _HYDROGEN) & resolved, first)
    has_carrier = numpy.logical_or.reduceat(carrier, first)
    names = template.res_name[first]
    lacking = ~has_h & has_carrier & ~numpy.isin(names, _WATER)
    if forms and (has_h & numpy.isin(names, list(forms))).any():
        # PDBbind 10GS draws polar hydrogens only: a missing thiol H says nothing
        count = hydrogens_by_parent(template)
        expected = numpy.full(len(template), -1)
        residue_of = struc.get_all_residue_positions(template)
        for name in set(names[has_h].tolist()) & set(forms):
            mine = names[residue_of] == name
            fixed = forms[name][2]
            expected[mine] = [fixed.get(a, -1) for a in template.atom_name[mine]]
        off = (expected >= 0) & resolved & (count != expected)
        lacking |= has_h & numpy.logical_or.reduceat(off, first)
    if residue_names is not None:
        lacking &= numpy.isin(names, list(residue_names))
    return starts, lacking


def hydrogens_by_parent(template: struc.AtomArray) -> numpy.ndarray:
    """Resolved hydrogens bonded to each atom; bonded by residue name without bonds."""
    element = numpy.char.upper(template.element.astype(str))
    is_h = numpy.isin(element, _HYDROGEN) & numpy.isfinite(template.coord).all(-1)
    bonds = template.bonds
    if bonds is None:
        bonds = struc.connect_via_residue_names(template)
    a, b = bonds.as_array()[:, :2].T.astype(numpy.int64)
    parent = numpy.r_[a[is_h[b] & ~is_h[a]], b[is_h[a] & ~is_h[b]]]
    return numpy.bincount(parent, minlength=len(template))


def _cache_protonation_states(keys, states):
    """Update the bounded state cache under its lock."""
    with _STATE_LOCK:
        for key in keys:
            _STATES[key] = states[key]
            _STATES.move_to_end(key)
        while len(_STATES) > _STATE_CAPACITY:
            _STATES.popitem(last=False)


def _stated_terminal_variants(template, starts, lacking, forms):
    """Database variants for complete, explicitly supplied free amines."""
    variant = numpy.full(len(template), -1, dtype=numpy.int8)
    if EXPLICIT_TERMINI in template.get_annotation_categories():
        stated = template.get_annotation(EXPLICIT_TERMINI)[starts[:-1]]
        count = hydrogens_by_parent(template)
        # Complete input states need no protonation call, including their zero
        # hydrogens on the unprotonated atom of a histidine tautomer.
        for r in numpy.flatnonzero(~lacking & ((stated & 1) != 0)):
            begin, end = starts[r : r + 2]
            name = template.res_name[begin]
            if name in (forms or {}):
                state = {
                    str(template.atom_name[i]): (int(count[i]), 0)
                    for i in range(begin, end)
                }
                chosen = _variant(forms[name], state, terminal=True)
                if chosen >= NEUTRAL_TERMINUS_VAR_BASE:
                    variant[begin:end] = chosen
    return variant


def _protonate_stack(structure, **kwargs):
    """Keep an ensemble only when independently assigned models share chemistry."""
    models = list(structure)
    registry = getattr(structure, "_custom_ccd_registry", None)
    if registry is not None:
        for model in models:
            model._custom_ccd_registry = registry
    marked = [with_atomworks_hydrogens(model, **kwargs) for model in models]
    if all(result is model for result, model in zip(marked, models)):
        return structure
    if any(
        not model.equal_annotations(marked[0]) or model.bonds != marked[0].bonds
        for model in marked[1:]
    ):
        raise ValueError("AtomWorks protonates the models of this stack differently")
    result = struc.stack(marked)
    if registry is not None:
        result._custom_ccd_registry = registry
    return result


def with_atomworks_hydrogens(
    structure: struc.AtomArray | struc.AtomArrayStack,
    *,
    ph: float = 7.4,
    residue_names: Collection[str] | None = None,
    coordination: numpy.ndarray | None = None,
    backbone: Mapping[str, tuple[str | None, str | None]] | None = None,
    forms: Mapping[str, Forms] | None = None,
    hydrogens: numpy.ndarray | None = None,
    alternative_types: Collection = (),
) -> struc.AtomArray | struc.AtomArrayStack:
    """``structure`` with AtomWorks' protonation of every residue lacking hydrogens.

    AtomWorks keeps the counts ``hydrogens`` and the resolved hydrogens of residues
    in ``forms`` declare. Residues in ``forms`` are only marked
    ``PROTONATION_VARIANT`` for tmol to build; others get AtomWorks' hydrogens and
    formal charges.

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
        hydrogens: Hydrogen count per atom AtomWorks is to keep, -1 where none.

    Returns:
        ``structure`` itself when nothing lacks hydrogens, else a new structure.

    Raises:
        ValueError: If the models of a stack are protonated differently.
    """
    if isinstance(structure, struc.AtomArrayStack):
        return _protonate_stack(
            structure,
            ph=ph,
            residue_names=residue_names,
            coordination=coordination,
            backbone=backbone,
            forms=forms,
            hydrogens=hydrogens,
            alternative_types=alternative_types,
        )

    starts, lacking = residues_lacking_hydrogens(structure, residue_names, forms)
    template = _template(structure)
    variant = _stated_terminal_variants(template, starts, lacking, forms)
    if not lacking.any():
        if (variant >= 0).any():
            structure = structure.copy()
            structure.set_annotation(PROTONATION_VARIANT, variant)
        return structure
    bonds_given = _template(structure).bonds is not None
    if not bonds_given:
        structure = structure.copy()
        structure.bonds = struc.connect_via_residue_names(_template(structure))
    template = _template(structure)
    n_atoms = len(template)
    residue_of = struc.get_all_residue_positions(template)
    res_name = template.res_name[starts[:-1]]
    is_h = numpy.isin(numpy.char.upper(template.element.astype(str)), _HYDROGEN)
    bonds = template.bonds.as_array().astype(numpy.int64)
    single = int(struc.BondType.SINGLE)
    coordination = numpy.zeros((0, 2)) if coordination is None else coordination
    disulfides = _disulfides(template)
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

    declared = numpy.where(lacking[residue_of], hydrogens_by_parent(template), 0)
    declared[declared == 0] = -1
    if hydrogens is not None:
        declared = numpy.where(hydrogens >= 0, hydrogens, declared)
    every = numpy.r_[bonds, extra]
    forms = forms or {}
    placing = lacking & ~numpy.isin(res_name, list(forms))
    # a heavy bond to another residue, not a state, replaces a hydrogen there
    varying = numpy.zeros(n_atoms, dtype=bool)
    amine = numpy.zeros(n_atoms, dtype=bool)
    for name in set(res_name[lacking].tolist()) & set(forms):
        mine = template.res_name == name
        varying |= mine & numpy.isin(template.atom_name, forms[name][0])
        amine |= mine & (template.atom_name == forms[name][3])
    bonded = _bonded_out(template, residue_of, every)
    replaced = numpy.logical_or.reduceat(varying & bonded, starts[:-1])
    terminal = numpy.logical_or.reduceat(amine & ~bonded, starts[:-1])
    asked = [
        r
        for r in numpy.flatnonzero(lacking & ~placing & ~replaced)
        if forms[res_name[r]][0] or terminal[r]
    ]
    keys = _contexts(template, starts, residue_of, every, asked, ph, declared)
    cacheable = not (
        getattr(structure, "_custom_ccd_registry", None) or get_custom_ccd_entries()
    )
    with _STATE_LOCK:
        states = {key: _STATES[key] for key in keys if cacheable and key in _STATES}
        for key in states:
            _STATES.move_to_end(key)
    new = {
        key: r for r, key in zip(reversed(asked), reversed(keys)) if key not in states
    }
    call = placing.copy()
    call[list(new.values())] = True
    if call.any():
        pairs = residue_of[numpy.r_[bonds[:, :2], extra[:, :2]]]
        selected = call.copy()
        for a, b in (pairs.T, pairs.T[::-1]):
            selected[b[call[a]]] = True
        selected &= ~numpy.isin(res_name, _WATER)
        placed = _placed_hydrogens(
            structure, selected[residue_of] & ~is_h, ph, extra, declared
        )
        parent, names, charge, _, free = placed
        count = numpy.bincount(parent, minlength=n_atoms)
        # a free tautomer and an unresolved atom (4NDZ B:171 TYR ring) say nothing
        count[free] = -1
        count[~numpy.isfinite(template.coord).all(axis=-1)] = -1
        state_charge = numpy.zeros(n_atoms, dtype=numpy.int64)
        state_charge[charge[0]] = charge[1]
        for key, r in new.items():
            states[key] = {
                str(template.atom_name[i]): (int(count[i]), int(state_charge[i]))
                for i in range(starts[r], starts[r + 1])
            }
        if cacheable:
            _cache_protonation_states(new, states)

    for r, key in zip(asked, keys):
        variant[starts[r] : starts[r + 1]] = _variant(
            forms[res_name[r]], states[key], terminal[r]
        )
    if alternative_types:
        from tmol.io._protonation_alternatives import encode_protonation_alternatives

        structure = structure.copy()
        structure.set_annotation(
            PROTONATION_ALTERNATIVES,
            encode_protonation_alternatives(
                template,
                starts,
                residue_of,
                extra,
                declared,
                asked,
                forms,
                alternative_types,
                ph,
            ),
        )
    if not placing.any():
        marked = structure.copy()
        marked.set_annotation(PROTONATION_VARIANT, variant)
        return marked if bonds_given else _unbonded(marked)

    keep_h = placing[residue_of[parent]]
    parent, names = parent[keep_h], names[keep_h]
    names = _names_by_parent(template, parent, names, residue_of)
    coords = placed[3][keep_h]

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
    hydrogens = base[parent]
    hydrogens.coord = coords
    hydrogens.bonds = struc.BondList(len(parent))
    base.bonds = struc.BondList(n_atoms)
    hydrogens.atom_name = names
    hydrogens.element[:] = "H"
    hydrogens.charge[:] = 0
    merged = concatenate_any([base, hydrogens])
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
    merged = merged[order]
    if registry is not None:
        merged._custom_ccd_registry = registry
    return merged if bonds_given else _unbonded(merged)


def _disulfides(template) -> numpy.ndarray:
    """The SG pairs the pose builder makes: closest unpaired SG within 2.5 A."""
    sg = numpy.flatnonzero(
        numpy.isin(template.res_name, ("CYS", "DCY"))
        & (template.atom_name == "SG")
        & numpy.isfinite(template.coord).all(axis=-1)
    )
    bonds = template.bonds.as_array()[:, :2]
    bonds = bonds[numpy.isin(bonds, sg).any(axis=1)]
    to_s = numpy.char.upper(template.element[bonds[:, ::-1]].astype(str)) == "S"
    paired = numpy.isin(sg, bonds[to_s])
    coord = template.coord[sg]
    close = cKDTree(coord).query_pairs(2.5, output_type="ndarray")
    d = numpy.linalg.norm(coord[close[:, 0]] - coord[close[:, 1]], axis=-1)
    pairs = []
    for i, j in close[numpy.lexsort((d, close[:, 0]))]:
        if not paired[i] and not paired[j]:
            paired[[i, j]] = True
            pairs.append((sg[i], sg[j]))
    return numpy.array(pairs, dtype=numpy.int64).reshape(-1, 2)


def _unbonded(structure):
    structure.bonds = None
    return structure


def _contexts(template, starts, residue_of, bonds, residues, ph, declared) -> list[str]:
    """Per residue, a key of what AtomWorks decides its state from: pH, name, resolved
    atoms' charges and ``declared`` hydrogen counts, and each bond out (with its
    length to a metal)."""
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
        tokens[residue_of[i]].append(f"{name[i]}{charge[i]}h{declared[i]}")
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


def _variant(
    forms: Forms, state: Mapping[str, tuple[int, int]], terminal: bool = False
) -> int:
    """The res_type_variant ``state`` selects among ``forms``; -1 where it leaves it open.

    A ``terminal`` residue whose amine keeps two hydrogens is a neutral amino terminus.
    """
    got = tuple(state.get(a, (-1, 0))[0] for a in forms[0])
    variant = -1 if any(n < 0 for n in got) else forms[1].get(got, -1)
    if variant >= 0 and terminal and state.get(forms[3], (0, 0))[0] == 2:
        variant += NEUTRAL_TERMINUS_VAR_BASE
    return variant


def _bonded_out(template, residue_of, bonds) -> numpy.ndarray:
    """Atoms with a covalent bond to a heavy atom of another residue, metals aside."""
    element = numpy.char.upper(template.element.astype(str))
    other = numpy.isin(element, [*_HYDROGEN, *METAL_ELEMENTS])
    covalent = bonds[bonds[:, 2] != int(struc.BondType.COORDINATION)]
    a, b = covalent[:, 0], covalent[:, 1]
    across = (residue_of[a] != residue_of[b]) & ~other[a] & ~other[b]
    out = numpy.zeros(len(template), dtype=bool)
    out[numpy.r_[a[across], b[across]]] = True
    return out


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
    """``[n, 3]`` single bonds joining consecutive polymer residues of a chain across its
    gaps by ``backbone`` (upper, lower) atoms, else C/N or O3'/P; free atoms only."""
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
    if EXPLICIT_TERMINI in template.get_annotation_categories():
        stated = template.get_annotation(EXPLICIT_TERMINI)[starts[:-1]]
        gap &= ((stated[:-1] & 2) == 0) & ((stated[1:] & 1) == 0)
    if "is_polymer" in template.get_annotation_categories():
        # 8GPB AMP A930/A940: free nucleotides of one chain are no neighbours
        polymer = template.is_polymer[starts[:-1]].astype(bool)
        gap &= polymer[:-1] & polymer[1:]
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
    declared: numpy.ndarray | None = None,
):
    """AtomWorks protonation of the ``heavy_mask`` atoms of one model, keeping the
    ``declared`` counts (-1 where none): hydrogen parents, names, (index, charge) of
    heavy atoms, coordinates, and tautomer-free atoms."""
    source, sub, chelates, registry = _protonation_input(model, heavy_mask, extra_bonds)
    with custom_ccd_residues(registry):
        atom_array = assign_hydrogens(
            sub, ph=ph, hydrogens=None if declared is None else declared[source]
        )
        protonated = place_hydrogens(atom_array)
    # each placed hydrogen has one bond, to its parent
    bonds = protonated.bonds.as_array()[:, :2]
    is_new = protonated.atom_id[bonds] >= len(atom_array)
    hydrogen = bonds[is_new]
    parent = protonated.atom_id[bonds[is_new[:, ::-1]]]
    order = numpy.lexsort((protonated.atom_id[hydrogen], parent))
    hydrogen, parent = hydrogen[order], parent[order]
    hydrogens = numpy.bincount(parent, minlength=len(atom_array))
    return (
        source[parent],
        protonated.atom_name[hydrogen].astype(str),
        (source, _charges_without_chelates(atom_array, hydrogens, chelates)),
        protonated.coord[hydrogen],
        source[atom_array.tautomer_free.astype(bool)],
    )


def _protonation_input(model, heavy_mask, extra_bonds):
    """Select heavy atoms and supply the bonds/CCD context used by AtomWorks."""
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
        sub.set_annotation("atom_id", numpy.arange(len(sub)))
    return source, sub, chelates, registry
