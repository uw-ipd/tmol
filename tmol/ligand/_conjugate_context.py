"""Parameter variants for equal attachment inventories in different contexts."""

from collections import defaultdict
import hashlib
import json

import attr

from tmol.database import inject_residue_params
from tmol.ligand._conjugate_model import iter_capped_conjugate_models
from tmol.ligand._conjugation_patches import CONNECTION_PREFIX
from tmol.ligand._parameter_replacements import _baseline_charges

CONTEXT_PREFIX = "context_"


def model_attachment_contexts(model):
    """Named partners per source residue, independent of coordinates and IDs."""
    local = {int(s): i for i, s in enumerate(model.source_atom_indices) if s >= 0}
    contexts = defaultdict(list)
    for first, second, _ in model.connections:
        for source, partner in ((first, second), (second, first)):
            i, j = local[source], local[partner]
            contexts[int(model.source_residue_indices[i])].append(
                (
                    str(model.atom_array.atom_name[i]),
                    str(model.atom_array.res_name[j]),
                    str(model.atom_array.atom_name[j]),
                )
            )
    return {res: tuple(sorted(context)) for res, context in contexts.items()}


def contextual_conjugate_types(atom_array, parameter_database, residue_name):
    """Split ambiguous parameter identities without renaming the source atoms.

    Called only after this residue's generated local parameters conflict.
    Its complete chemical inventory must already agree; incompatible source
    identities still fail the normal chemistry checks.
    """
    models = list(iter_capped_conjugate_models(atom_array, parameter_database.chemical))
    contexts = defaultdict(set)
    for model in models:
        local_contexts = model_attachment_contexts(model)
        sites, names = defaultdict(set), {}
        local = {int(s): i for i, s in enumerate(model.source_atom_indices) if s >= 0}
        for bond, connections in zip(model.connections, model.connection_names):
            for source, connection in zip(bond[:2], connections):
                i = local[source]
                residue = int(model.source_residue_indices[i])
                names[residue] = str(model.atom_array.res_name[i])
                if connection.startswith(CONNECTION_PREFIX):
                    sites[residue].add(connection)
        for residue, site_names in sites.items():
            contexts[names[residue], frozenset(site_names)].add(local_contexts[residue])
    original = next(
        rt for rt in parameter_database.chemical.residues if rt.name == residue_name
    )
    sites = frozenset(
        c.name for c in original.connections if c.name.startswith(CONNECTION_PREFIX)
    )
    key = (original.io_equiv_class, sites)
    ambiguous = {key: contexts[key]} if len(contexts[key]) > 1 else {}
    if not ambiguous:
        return parameter_database, models
    chem = parameter_database.chemical
    existing = {rt.name for rt in chem.residues}
    charge_index = {
        (p.res, p.atom): p.charge
        for p in parameter_database.scoring.elec.atom_charge_parameters
    }
    cart = parameter_database.scoring.cartbonded.residue_params
    added, charges, bonded = [], {}, {}
    for rt in chem.residues:
        if rt.conjugation_context:
            continue
        sites = frozenset(
            c.name for c in rt.connections if c.name.startswith(CONNECTION_PREFIX)
        )
        for context in sorted(ambiguous.get((rt.io_equiv_class, sites), ())):
            digest = hashlib.sha256(json.dumps(context).encode()).hexdigest()[:16]
            name = f"{rt.name}:{CONTEXT_PREFIX}{digest}"
            if name in existing:
                continue
            added.append(attr.evolve(rt, name=name, conjugation_context=context))
            charges[name] = _baseline_charges(charge_index, rt)
            bonded[name] = cart.get(rt.name, cart.get(rt.base_name))
            existing.add(name)
    if not added:
        return parameter_database, models
    extended = inject_residue_params(
        parameter_database, [], partial_charges=charges, cartbonded_params=bonded
    )
    return (
        attr.evolve(
            extended,
            chemical=attr.evolve(extended.chemical, residues=(*chem.residues, *added)),
        ),
        models,
    )
