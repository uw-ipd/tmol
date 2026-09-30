"""Read each residue's protonation state from the hydrogens it presents."""

import logging
from collections import defaultdict

import torch

from tmol.database.chemical import DEPROTONATED_VAR_IND, special_case_variant_index

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
    """Each residue's variant from its hydrogens: one presenting hydrogens but no
    titratable one (unless covalently bonded there) is deprotonated.

    A residue with no hydrogens keeps its variant, unless it coordinates a metal:
    then it takes the first form whose atoms donate.
    """
    classes = canonical_ordering.restype_io_equiv_classes
    hydrogens, donates, parent = _variant_atoms(
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
        expects = torch.isin(out, torch.tensor(expecting, device=out.device))
        move = is_class & expects & has_h & ~has_titratable
        if bool(move.any()):
            out = torch.where(move, torch.full_like(out, DEPROTONATED_VAR_IND), out)

    return _with_metal_donor_variants(
        out, res_types, has_h, metal_assignments, donates, classes
    )


def _variant_atoms(canonical_ordering, chemical_db, res_types):
    """Per present class and variant: hydrogen and metal-donor canonical atoms.

    Also each hydrogen's heavy parent, per class.
    """
    classes = canonical_ordering.restype_io_equiv_classes
    present = {int(c) for c in torch.unique(res_types[res_types >= 0]).tolist()}
    wanted = {classes[c]: c for c in present}
    atom_type = {at.name: at for at in chemical_db.atom_types}
    hydrogens = defaultdict(lambda: defaultdict(set))
    donates = defaultdict(lambda: defaultdict(set))
    parent = defaultdict(dict)
    for res in chemical_db.residues:
        c = wanted.get(res.io_equiv_class)
        if c is None:
            continue
        v = special_case_variant_index(res)
        index = canonical_ordering.restypes_atom_index_mapping[res.io_equiv_class]
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
    return hydrogens, donates, parent


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
