"""The opt-in achiral GLY model must also build equivalent alpha hydrogens."""

import attr

from .scoring._cartbonded import CartBondedDatabase


def symmetric_gly_geometry(chemical, cartbonded):
    hydrogen_names = {"HA2", "HA3"}

    def symmetric_residue(residue):
        if residue.base_name != "GLY":
            return residue
        pair = [
            ic
            for ic in residue.icoors
            if ic.name in hydrogen_names and ic.parent == "CA"
        ]
        if {ic.name for ic in pair} != hydrogen_names:
            return residue
        length = sum(ic.d for ic in pair) / 2
        return attr.evolve(
            residue,
            icoors=tuple(
                attr.evolve(ic, d=length) if ic in pair else ic for ic in residue.icoors
            ),
        )

    params = dict(cartbonded.residue_params)
    gly_names = {r.name for r in chemical.residues if r.base_name == "GLY"}
    for name in gly_names & params.keys():
        record = params[name]
        pair = [
            p
            for p in record.length_parameters
            if {p.atm1, p.atm2} in ({"CA", "HA2"}, {"CA", "HA3"})
        ]
        if len(pair) != 2:
            continue
        # Average the two harmonic potentials under HA2 <-> HA3. Their
        # coordinate-independent offset is immaterial; weighting the target
        # by K also handles independently supplied force constants.
        stiffness = sum(p.K for p in pair)
        target = (
            sum(p.K * p.x0 for p in pair) / stiffness
            if stiffness
            else sum(p.x0 for p in pair) / 2
        )
        params[name] = attr.evolve(
            record,
            length_parameters=tuple(
                attr.evolve(p, x0=target, K=stiffness / 2) if p in pair else p
                for p in record.length_parameters
            ),
        )
    return (
        attr.evolve(
            chemical, residues=tuple(map(symmetric_residue, chemical.residues))
        ),
        CartBondedDatabase.from_cartres_dict(params, cartbonded.connection_params),
    )
