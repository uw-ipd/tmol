"""Database-supported alternatives from AtomWorks' existing titration rules."""

from functools import lru_cache

import numpy
from atomworks.experimental.protonation import assign_hydrogens
from atomworks.experimental.protonation.dimorphite import site_rules
from atomworks.io.utils.ccd import custom_ccd_residues

from tmol.database.chemical import l_base_name, special_case_variant_index

# RT ln(10) at 298 K, kcal/mol per proton and pH unit.
RT_LN10 = 1.364


def parse_protonation_alternatives(value: str) -> dict[str, float]:
    """Decode a residue's comma-separated ``base_name:offset`` annotation."""
    return {
        name: float(offset)
        for entry in value.split(",")
        if entry
        for name, offset in [entry.rsplit(":", 1)]
    }


@lru_cache(maxsize=1)
def _titration_boundaries():
    """Dimorphite-DL's pH transitions and their model pKas."""
    boundaries = sorted(
        {
            (pka - 0.1 * std, pka)
            for _, _, sites in site_rules()
            for _, pka, std in sites
        }
    )
    return tuple(
        (
            pka,
            (boundaries[i - 1][0] + edge) / 2 if i else edge - 1,
            (edge + boundaries[i + 1][0]) / 2 if i + 1 < len(boundaries) else edge + 1,
        )
        for i, (edge, pka) in enumerate(boundaries)
    )


def encode_protonation_alternatives(
    model, starts, residue_of, extra_bonds, declared, asked, forms, residue_types, ph
):
    """Record free, unambiguously supported database states within one pKa unit.

    Input hydrogens and cross-residue bonds fix a site. Assignment at either
    side of an actual AtomWorks titration boundary supplies candidates; database
    templates only name those states. Free tautomers share the same offset.
    """
    from tmol.io._protonation import _protonation_input

    labels = {}
    for residue in residue_types:
        if residue.metal_sites or ":" in residue.name or l_base_name(residue) == "CYD":
            continue
        labels.setdefault(residue.io_equiv_class, {})[
            special_case_variant_index(residue)
        ] = residue.base_name
    asked = [
        r
        for r in asked
        if forms[model.res_name[starts[r]]][0]
        and not (declared[starts[r] : starts[r + 1]] >= 0).any()
    ]
    encoded = numpy.full(len(starts) - 1, "", dtype=object)
    if not asked:
        return encoded[residue_of].astype(str)
    bonds = numpy.concatenate([model.bonds.as_array(), extra_bonds]).astype(int)
    selected = numpy.isin(residue_of, asked)
    neighbours = bonds[numpy.any(selected[bonds[:, :2]], axis=1), :2].ravel()
    selected |= numpy.isin(residue_of, residue_of[neighbours])
    selected &= ~numpy.isin(numpy.char.upper(model.element.astype(str)), ("H", "D"))
    source, sub, _, registry = _protonation_input(model, selected, extra_bonds)
    local = numpy.full(len(model), -1)
    local[source] = numpy.arange(len(source))

    @lru_cache(maxsize=None)
    def at_ph(value):
        return assign_hydrogens(sub, ph=value, hydrogens=declared[source])

    with custom_ccd_residues(registry):
        assigned = at_ph(ph)
        transitions = [
            (pka, at_ph(below), at_ph(above))
            for pka, below, above in _titration_boundaries()
            if abs(ph - pka) <= 1
        ]
    for residue in asked:
        begin, end = starts[residue : residue + 2]
        name = model.res_name[begin]
        varying, variants, *_ = forms[name]
        atoms = {str(model.atom_name[i]): int(local[i]) for i in range(begin, end)}
        if any(atoms.get(atom, -1) < 0 for atom in varying):
            continue
        indices = numpy.array([atoms[atom] for atom in varying])
        if not numpy.isfinite(sub.coord[indices]).all():
            continue

        def state(array):
            return tuple(array.nhyd[indices].tolist()), tuple(
                array.charge[indices].tolist()
            )

        current = state(assigned)
        offsets = {current[0]: 0.0}
        for pka, below, above in transitions:
            for near, far in ((below, above), (above, below)):
                if state(near) == current:
                    counts, _ = state(far)
                    offsets[counts] = (
                        RT_LN10 * (ph - pka) * (sum(counts) - sum(current[0]))
                    )
        if assigned.tautomer_free[indices].all():
            for counts in variants:
                if sum(counts) == sum(current[0]):
                    offsets[counts] = 0.0
        options = {
            labels[name][variants[counts]]: offset
            for counts, offset in offsets.items()
            if counts in variants and variants[counts] in labels.get(name, {})
        }
        if len(options) > 1:
            encoded[residue] = ",".join(
                f"{label}:{offset!r}" for label, offset in options.items()
            )
    return encoded[residue_of].astype(str)
