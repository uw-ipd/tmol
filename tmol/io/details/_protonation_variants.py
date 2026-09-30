"""Read each residue's protonation state from the hydrogens it presents."""

import logging
from collections import defaultdict

import attr
import torch

from tmol.database.chemical import (
    DEPROTONATED_VAR_IND,
    NEUTRAL_TERMINUS,
    NEUTRAL_TERMINUS_VAR_BASE,
    special_case_variant_index,
)

logger = logging.getLogger(__name__)


def select_protonation_variants(
    canonical_ordering,
    chemical_db,
    res_types,
    res_type_variants,
    atom_is_present,
    metal_assignments,
    covalent_bonds=None,
):
    """The protonation variant of each residue, read from its hydrogens.

    A titratable hydrogen is one some form of the class has and no
    deprotonated form does: a thiol's HG, a phenol's HH, either imidazole NH.
    A residue presenting every hydrogen all forms share, in a variant expecting
    a titratable one, is deprotonated when it presents none; one presenting
    fewer (polar hydrogens only) is not. Variants that expect none,
    such as a disulfide, never change, and neither does a residue whose
    titratable hydrogen's parent is bonded to another residue in
    ``covalent_bonds`` (``[pose, res1, atom1, res2, atom2]`` rows): the bond
    replaces that hydrogen.

    A residue presenting no hydrogen at all, as from coordinates without
    hydrogens, says nothing about its state. It keeps its variant unless it
    coordinates a metal, which takes the first form whose atoms donate.
    """
    classes = canonical_ordering.restype_io_equiv_classes
    hydrogens, donates, parent, linked = _variant_atoms(
        canonical_ordering, chemical_db, res_types
    )
    device = atom_is_present.device
    bonded = torch.zeros(atom_is_present.shape, dtype=torch.bool, device=device)
    if covalent_bonds is not None and len(covalent_bonds):
        rows = covalent_bonds.to(device)
        bonded[rows[:, 0], rows[:, 1], rows[:, 2]] = True
        bonded[rows[:, 0], rows[:, 3], rows[:, 4]] = True
    out = res_type_variants
    has_h = torch.zeros(res_types.shape, dtype=torch.bool, device=device)
    for c, by_variant in hydrogens.items():
        is_class = res_types == c
        all_h = torch.tensor(sorted(set().union(*by_variant.values())), device=device)
        has_h |= is_class & atom_is_present[:, :, all_h].any(dim=-1)
        if DEPROTONATED_VAR_IND not in by_variant:
            continue
        titratable = (
            set().union(
                *(h for v, h in by_variant.items() if v != DEPROTONATED_VAR_IND)
            )
            - by_variant[DEPROTONATED_VAR_IND]
        )
        expecting = [
            v
            for v, h in by_variant.items()
            if v != DEPROTONATED_VAR_IND and h & titratable
        ]
        if not titratable or not expecting:
            continue
        titr = torch.tensor(sorted(titratable), device=device)
        has_titratable = atom_is_present[:, :, titr].any(dim=-1)
        parents = sorted({parent[c][h] for h in titratable if h in parent[c]})
        if parents:
            has_titratable |= bonded[:, :, parents].any(dim=-1)
        shared = set.intersection(*map(set, by_variant.values()))
        shared = sorted(h for h in shared if parent[c].get(h) not in linked[c])
        complete = atom_is_present[:, :, shared].all(dim=-1)
        expects = torch.isin(out, torch.tensor(expecting, device=out.device))
        move = is_class & expects & has_h & complete & ~has_titratable
        if bool(move.any()):
            out = torch.where(move, torch.full_like(out, DEPROTONATED_VAR_IND), out)

    return _with_metal_donor_variants(
        out, res_types, has_h, metal_assignments, donates, classes
    )


def _variant_atoms(canonical_ordering, chemical_db, res_types):
    """Per present class and variant: hydrogen and metal-donor canonical atoms.

    Also each hydrogen's heavy parent and the connection atoms, per class.
    """
    classes = canonical_ordering.restype_io_equiv_classes
    present = {int(c) for c in torch.unique(res_types[res_types >= 0]).tolist()}
    wanted = {classes[c]: c for c in present}
    atom_type = {at.name: at for at in chemical_db.atom_types}
    hydrogens = defaultdict(lambda: defaultdict(set))
    donates = defaultdict(lambda: defaultdict(set))
    parent = defaultdict(dict)
    linked = defaultdict(set)
    for res in chemical_db.residues:
        c = wanted.get(res.io_equiv_class)
        if c is None:
            continue
        v = special_case_variant_index(res)
        index = canonical_ordering.restypes_atom_index_mapping[res.io_equiv_class]
        linked[c].update(index[x.atom] for x in res.connections if x.atom in index)
        element = {}
        for a in res.atoms:
            at = atom_type.get(a.atom_type)
            if a.name not in index or at is None:
                continue
            element[a.name] = at.element
            if at.element == "H" and v <= DEPROTONATED_VAR_IND:
                hydrogens[c][v].add(index[a.name])
            if at.is_metal_donor:
                donates[c][v].add(index[a.name])
        for a, b, *_ in res.bonds:
            for h, heavy in ((a, b), (b, a)):
                if element.get(h) == "H" and element.get(heavy, "H") != "H":
                    parent[c][index[h]] = index[heavy]
    return hydrogens, donates, parent, linked


def _with_metal_donor_variants(
    out, res_types, has_h, metal_assignments, donates, classes
):
    """``out`` with each metal-coordinating residue lacking hydrogens in a donor form."""
    required = defaultdict(set)
    for (pose, _), got in metal_assignments:
        for res, atom in got.donor_atoms:
            if not bool(has_h[pose, res]):
                required[(pose, res)].add(atom)
    if not required:
        return out
    out = out.clone()
    for (pose, res), atoms in required.items():
        cls = int(res_types[pose, res])
        if atoms <= donates[cls][int(out[pose, res])]:
            continue
        options = sorted(v for v, can in donates[cls].items() if atoms <= can)
        if not options:
            logger.warning(
                "%s %d coordinates through atoms no form of it can donate",
                classes[cls],
                res,
            )
            continue
        out[pose, res] = options[0]
    return out


def neutral_terminus_patches(canonical_ordering, chemical_db, res_types, variants):
    """The neutral amino terminus, for each base type of a class a residue asks it of.

    The database's patch applies to no type, so its forms exist only where needed.
    """
    asked = torch.unique(res_types[variants >= NEUTRAL_TERMINUS_VAR_BASE]).tolist()
    if not asked:
        return ()
    classes = {canonical_ordering.restype_io_equiv_classes[c] for c in asked}
    patch = next(v for v in chemical_db.variants if v.display_name == NEUTRAL_TERMINUS)
    return tuple(
        attr.evolve(
            patch,
            name=f"{patch.name}_{res.name}",
            applies_to=attr.evolve(patch.applies_to, base_names=(res.name,)),
        )
        for res in chemical_db.residues
        if res.name == res.base_name and res.io_equiv_class in classes
    )
