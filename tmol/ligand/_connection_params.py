"""Harmonic attachment parameters derived from complete capped chemistry.

This is a local harmonic approximation to MMFF94 bond/angle terms. It omits
anharmonic and stretch-bend terms; it is not the full MMFF energy function.
"""

from collections import defaultdict
import json
import math

from attr import evolve
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from tmol.database.scoring import ConnectionCartRes, LengthGroup, AngleGroup
from tmol.database.scoring._content_hash import content_hash
from tmol.ligand._conjugate_model import iter_capped_conjugate_models
from tmol.ligand._conjugation_patches import CONNECTION_PREFIX, connection_name
from tmol.ligand._dimorphite_dl import ProtSubstructFuncs, protonate_mol_variants
from tmol.ligand._rdkit_mol import rdkit_mol_from_ligand_atom_array

# RDKit MMFF/Params.h, MDYNE_A_TO_KCAL_MOL. At equilibrium the bond and
# radian-angle Hessians are this factor times MMFF kb and ka, respectively.
MMFF_HARMONIC_CONVERSION = 143.9325


def _parameterized_model(model, ph):
    mol = rdkit_mol_from_ligand_atom_array(model.atom_array)
    if mol.GetNumAtoms() != len(model.atom_array):
        raise ValueError("Conjugate conversion changed the heavy-atom inventory")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    mol.RemoveAllConformers()
    variants = protonate_mol_variants(
        mol, min_ph=ph, max_ph=ph, pka_precision=0.1, max_variants=128, silent=True
    )
    if not variants:
        raise ValueError("Conjugate protonation produced no chemical state")
    mol = Chem.AddHs(variants[0])
    mapping = {
        a.GetAtomMapNum() - 1: a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1
    }
    if set(mapping) != set(range(len(model.atom_array))):
        raise ValueError("Conjugate protonation changed mapped heavy-atom identity")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if props is None:
        raise ValueError("MMFF94 does not cover this capped conjugate")
    return mol, props, mapping


def _neighbors(restype):
    neighbors = {a.name: set() for a in restype.atoms}
    for a, b, *_ in restype.bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return {name: sorted(adjacent) for name, adjacent in neighbors.items()}


def _residue_mapping(model, mol, heavy_map, locals_, restype, elements, neighbors):
    """Map source heavy atoms and equivalent hydrogens by their bonded parent."""
    aliases = {a.alt_name: a.name for a in restype.atom_aliases}
    present = {a.name for a in restype.atoms}
    result = {}
    for local in locals_:
        name = str(model.atom_array.atom_name[local])
        name = name if name in present else aliases.get(name, name)
        if name in present:
            result[name] = heavy_map[local]
    atom_elements = {a.name: elements[a.atom_type] for a in restype.atoms}
    for name, index in list(result.items()):
        hydrogens = [n for n in neighbors[name] if atom_elements[n] == "H"]
        generated = sorted(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(index).GetNeighbors()
            if n.GetAtomicNum() == 1
        )
        # Terminal variants may differ at distant backbone atoms. Do not map
        # incompatible H sets; a requested attachment path must still match.
        if len(hydrogens) == len(generated):
            result.update(zip(hydrogens, generated))
    return result


def _connection_record(mol, props, first, second, maps, names, adjacency):
    connections = [
        next(c for c in rt.connections if c.name == name)
        for rt, name in zip((first, second), names)
    ]
    roots = [c.atom for c in connections]
    try:
        indices = [mapping[root] for mapping, root in zip(maps, roots)]
        neighbors = [adj[root] for adj, root in zip(adjacency, roots)]
        for side in (0, 1):
            expected = {maps[side][n] for n in neighbors[side]} | {indices[1 - side]}
            actual = {
                n.GetIdx() for n in mol.GetAtomWithIdx(indices[side]).GetNeighbors()
            }
            if expected != actual:
                raise ValueError(
                    "Attachment neighborhood differs from its patched residue type, or an angle spans three blocks"
                )
    except KeyError as err:
        raise ValueError(
            "Attachment atom/hydrogen mapping differs from its patched residue type"
        ) from err
    bond_type = str(mol.GetBondBetweenAtoms(*indices).GetBondType())
    if any(c.type != bond_type for c in connections):
        raise ValueError(
            "Attachment bond order differs from its patched connection type"
        )
    bond = props.GetMMFFBondStretchParams(mol, *indices)
    if bond is None:
        raise ValueError("Missing MMFF94 attachment bond parameter")
    lengths = (
        LengthGroup(
            roots[0], "+" + roots[1], x0=bond[2], K=MMFF_HARMONIC_CONVERSION * bond[1]
        ),
    )
    angles = []
    for side in (0, 1):
        for neighbor in neighbors[side]:
            values = props.GetMMFFAngleBendParams(
                mol, maps[side][neighbor], indices[side], indices[1 - side]
            )
            if values is None:
                raise ValueError("Missing MMFF94 attachment angle parameter")
            prefix, far = ("", "+") if side == 0 else ("+", "")
            angles.append(
                AngleGroup(
                    prefix + neighbor,
                    prefix + roots[side],
                    far + roots[1 - side],
                    x0=math.radians(values[2]),
                    K=MMFF_HARMONIC_CONVERSION * values[1],
                )
            )
    return ConnectionCartRes(
        first.name, names[0], second.name, names[1], lengths, tuple(angles)
    )


def _model_members(model, candidates_by_site):
    source_local, locals_by_residue = {}, defaultdict(list)
    for local, source in enumerate(model.source_atom_indices):
        if source >= 0:
            source_local[int(source)] = local
            locals_by_residue[int(model.source_residue_indices[local])].append(local)
    sites, links = defaultdict(set), []
    for a, b, _ in model.connections:
        pair = []
        for source in (a, b):
            local = source_local[source]
            ri = int(model.source_residue_indices[local])
            name = connection_name(str(model.atom_array.atom_name[local]))
            sites[ri].add(name)
            pair.append((ri, name))
        links.append(pair)
    candidates = {
        ri: candidates_by_site.get(
            (
                str(model.atom_array.res_name[locals_by_residue[ri][0]]),
                frozenset(names),
            ),
            (),
        )
        for ri, names in sites.items()
    }
    if any(candidates.values()) and not all(candidates.values()):
        raise ValueError("No compatible patched residue type for a conjugate member")
    return locals_by_residue, sites, links, candidates


def _context_signature(model, mol, props, heavy_map, locals_):
    # Include chemical identity as well as numerical MMFF assignments. Two
    # isomers can have the same MMFF types and charges without being one type.
    named = {heavy_map[i]: str(model.atom_array.atom_name[i]) for i in locals_}
    atoms = []
    for index, name in named.items():
        atom = mol.GetAtomWithIdx(index)
        atoms.append(
            (
                name,
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                props.GetMMFFAtomType(index),
                round(props.GetMMFFPartialCharge(index), 12),
                sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors()),
                tuple(
                    sorted(
                        (named[b.GetOtherAtomIdx(index)], str(b.GetBondType()))
                        for b in atom.GetBonds()
                        if b.GetOtherAtomIdx(index) in named
                    )
                ),
            )
        )
    return tuple(sorted(atoms))


def _model_records(
    model, candidates_by_site, elements, adjacency, contexts, ph, consumer=None
):
    locals_by_residue, sites, links, candidates = _model_members(
        model, candidates_by_site
    )
    if not any(candidates.values()):
        return
    mol, props, heavy_map = _parameterized_model(model, ph)
    mappings = {}
    for ri, types in candidates.items():
        signature = _context_signature(
            model, mol, props, heavy_map, locals_by_residue[ri]
        )
        identity = (types[0].base_name, tuple(sorted(sites[ri])))
        if identity in contexts and contexts[identity] != signature:
            raise ValueError(
                f"Incompatible conjugate chemistry shares residue identity {identity}"
            )
        contexts[identity] = signature
        for rt in types:
            mappings[ri, rt.name] = _residue_mapping(
                model,
                mol,
                heavy_map,
                locals_by_residue[ri],
                rt,
                elements,
                adjacency[rt.name],
            )
    if consumer is not None:
        consumer(model, mol, props, heavy_map, candidates, mappings, adjacency)
    unmapped = Chem.Mol(mol)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)
    chemical_smiles = Chem.MolToSmiles(Chem.RemoveHs(unmapped))
    for (ra, ca), (rb, cb) in links:
        for first in candidates[ra]:
            for second in candidates[rb]:
                rts, names = (first, second), (ca, cb)
                maps = (mappings[ra, first.name], mappings[rb, second.name])
                if (first.name, ca) > (second.name, cb):
                    rts, names, maps = rts[::-1], names[::-1], maps[::-1]
                record = _connection_record(
                    mol,
                    props,
                    *rts,
                    maps,
                    names,
                    [adjacency[rt.name] for rt in rts],
                )
                yield record, chemical_smiles


def _model_identity(model):
    # Include every chemistry annotation consumed by the RDKit converter,
    # plus names and relative residue/cap identity used to map parameter rows.
    # Coordinates are all NaN in these topology-only models. Chain numbering
    # and PDB metadata do not affect this generator's chemical assignments.
    array = model.atom_array
    arrays = {
        name: array.get_annotation(name)
        for name in (
            "res_name",
            "atom_name",
            "element",
            "charge",
            "tmol_aromatic",
            "tmol_source_subtype",
        )
        if name in array.get_annotation_categories()
    }
    arrays["bonds"] = array.bonds.as_array()
    arrays["source"] = model.source_atom_indices >= 0
    arrays["residue"] = np.unique(model.source_residue_indices, return_inverse=True)[1]
    return content_hash(
        tuple(
            (name, str(values.dtype), values.shape, values.tobytes())
            for name, values in arrays.items()
        )
    )


def generate_conjugate_connection_params(
    atom_array, parameter_database, *, ph=7.4, _model_consumer=None
):
    """Generate complete two-block bond/angle records for prepared attachments.

    Exact patched names remain separate, including terminal combinations. A
    repeated residue/site identity with incompatible MMFF chemistry raises;
    this function does not rename input residues or invent context selection.
    Existing non-conjugation connections are not assigned new records. This
    private generator is not yet part of the default preparation pipeline.
    """
    if not math.isfinite(ph):
        raise ValueError("Conjugate protonation pH must be finite")
    chem = parameter_database.chemical
    candidates, adjacency = defaultdict(list), {}
    for rt in chem.residues:
        sites = frozenset(
            c.name for c in rt.connections if c.name.startswith(CONNECTION_PREFIX)
        )
        if sites:
            candidates[rt.base_name, sites].append(rt)
            adjacency[rt.name] = _neighbors(rt)
    for alias in chem.name3_aliases:
        for (base, sites), types in list(candidates.items()):
            if base == alias.read_as:
                candidates.setdefault((alias.name3, sites), types)
    elements = {a.name: a.element for a in chem.atom_types}
    records, contexts, provenance = {}, {}, defaultdict(set)
    seen = set()
    for model in iter_capped_conjugate_models(atom_array, chem):
        identity = _model_identity(model)
        if identity in seen:
            continue
        for record, smiles in _model_records(
            model, candidates, elements, adjacency, contexts, ph, _model_consumer
        ):
            key = (
                record.block_type1,
                record.connection1,
                record.block_type2,
                record.connection2,
            )
            if key in records and records[key] != record:
                raise ValueError(
                    f"Conflicting generated connection parameters for {key}"
                )
            records[key] = record
            provenance[key].add(smiles)
        # Repeated groups need parameterization once per call, without keeping
        # molecules alive or growing a cross-pose/global chemistry cache.
        if len(seen) == 128:
            seen.clear()
        seen.add(identity)

    if not records:
        return ()
    protonation = {
        "engine": "tmol Dimorphite-DL",
        "selection": "first ordered variant",
        "pka_precision": 0.1,
        "max_variants": 128,
        # Hash the actual compiled rule records, not a potentially edited
        # on-disk file or query-object addresses.
        "rules_sha256": content_hash(
            tuple(
                (name, smart, sites)
                for name, smart, _, sites in ProtSubstructFuncs._compiled_substructures()
            )
        ),
    }

    return tuple(
        evolve(
            records[key],
            provenance=json.dumps(
                {
                    "method": "tmol-mmff94-harmonic-v1",
                    "rdkit": rdBase.rdkitVersion,
                    "ph": ph,
                    "protonation": protonation,
                    "model": "complete conjugate with polymer caps",
                    "capped_smiles": sorted(provenance[key]),
                    "approximation": "equilibrium curvature; no anharmonic or stretch-bend terms",
                },
                sort_keys=True,
            ),
        )
        for key in sorted(records)
    )
