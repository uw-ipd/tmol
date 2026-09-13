"""Harmonic attachment parameters derived from complete capped chemistry.

Generated geometry and constants match the ordinary ligand Cartesian generator.
The alternate MMFF harmonic source is retained for private diagnostic consumers.
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
from tmol.ligand._conjugation_patches import CONNECTION_PREFIX
from tmol.ligand._dimorphite_dl import ProtSubstructFuncs, protonate_mol_variants
from tmol.ligand._conjugate_geometry import generated_conjugate_coordinates
from tmol.ligand._registry import GENERATED_LENGTH_K, GENERATED_ANGLE_K

# RDKit MMFF/Params.h, MDYNE_A_TO_KCAL_MOL. At equilibrium the bond and
# radian-angle Hessians are this factor times MMFF kb and ka, respectively.
MMFF_HARMONIC_CONVERSION = 143.9325


def _parameterized_model(model, ph):
    mol = Chem.Mol(model.molecule)
    if mol.GetNumAtoms() != len(model.atom_array):
        raise ValueError("Conjugate conversion changed the heavy-atom inventory")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
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


def _connection_record(mol, props, first, second, maps, names, adjacency, coords=None):
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
    # Polymer ports use AROMATIC for the partial-double peptide convention,
    # whereas RDKit represents both amide and ester attachments as SINGLE.
    if any(
        c.type != bond_type
        and not (
            c.name in ("up", "down") and c.type == "AROMATIC" and bond_type == "SINGLE"
        )
        for c in connections
    ):
        raise ValueError(
            "Attachment bond order differs from its patched connection type"
        )
    if coords is None:
        bond = props.GetMMFFBondStretchParams(mol, *indices)
        if bond is None:
            raise ValueError("Missing MMFF94 attachment bond parameter")
        length, length_k = bond[2], MMFF_HARMONIC_CONVERSION * bond[1]
    else:
        length = float(np.linalg.norm(coords[indices[0]] - coords[indices[1]]))
        length_k = GENERATED_LENGTH_K
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Generated attachment bond has degenerate geometry")
    lengths = (LengthGroup(roots[0], "+" + roots[1], x0=length, K=length_k),)
    angles = []
    for side in (0, 1):
        for neighbor in neighbors[side]:
            if coords is None:
                values = props.GetMMFFAngleBendParams(
                    mol, maps[side][neighbor], indices[side], indices[1 - side]
                )
                if values is None:
                    raise ValueError("Missing MMFF94 attachment angle parameter")
                angle, angle_k = (
                    math.radians(values[2]),
                    MMFF_HARMONIC_CONVERSION * values[1],
                )
            else:
                a = coords[maps[side][neighbor]] - coords[indices[side]]
                b = coords[indices[1 - side]] - coords[indices[side]]
                if min(np.linalg.norm(a), np.linalg.norm(b)) <= 0:
                    raise ValueError(
                        "Generated attachment angle has degenerate geometry"
                    )
                angle, angle_k = (
                    math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)),
                    GENERATED_ANGLE_K,
                )
            prefix, far = ("", "+") if side == 0 else ("+", "")
            angles.append(
                AngleGroup(
                    prefix + neighbor,
                    prefix + roots[side],
                    far + roots[1 - side],
                    x0=angle,
                    K=angle_k,
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
    for (a, b, _), names in zip(model.connections, model.connection_names, strict=True):
        pair = []
        for source, name in zip((a, b), names):
            local = source_local[source]
            ri = int(model.source_residue_indices[local])
            sites[ri].add(name)
            pair.append((ri, name))
        links.append(pair)
    candidates = {
        ri: [
            rt
            for rt in candidates_by_site.get(
                (
                    str(model.atom_array.res_name[locals_by_residue[ri][0]]),
                    frozenset(n for n in names if n.startswith(CONNECTION_PREFIX)),
                ),
                (),
            )
            if names <= {c.name for c in rt.connections}
        ]
        for ri, names in sites.items()
    }
    if any(candidates.values()) and not all(candidates.values()):
        raise ValueError("No compatible patched residue type for a conjugate member")
    return locals_by_residue, sites, links, candidates


def _named_chirality(atom, named):
    """Tetrahedral handedness in residue-name order, independent of CIP priority."""
    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_UNSPECIFIED:
        return 0
    if tag not in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ):
        raise ValueError("Unsupported non-tetrahedral conjugate stereochemistry")
    neighbors = [
        (
            (0, named[n.GetIdx()])
            if n.GetIdx() in named
            else (1, n.GetAtomicNum(), n.GetIsotope())
        )
        for n in atom.GetNeighbors()
    ]
    if len(set(neighbors)) != len(neighbors):
        raise ValueError("Ambiguous external neighbors at a conjugate stereocenter")
    inversions = sum(a > b for i, a in enumerate(neighbors) for b in neighbors[i + 1 :])
    return (-1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 1) * (-1) ** inversions


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
                _named_chirality(atom, named),
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
    model,
    candidates_by_site,
    elements,
    adjacency,
    contexts,
    ph,
    consumer=None,
    *,
    parameter_source="generated-geometry",
    seed=0,
    existing=frozenset(),
):
    locals_by_residue, sites, links, candidates = _model_members(
        model, candidates_by_site
    )
    if not any(candidates.values()):
        return
    pairs = [
        (ra, ca, first, rb, cb, second)
        for (ra, ca), (rb, cb) in links
        for first in candidates[ra]
        for second in candidates[rb]
        if frozenset(((first.name, ca), (second.name, cb))) not in existing
    ]
    if not pairs:
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
            if rt.name not in adjacency:
                adjacency[rt.name] = _neighbors(rt)
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
    coords = (
        generated_conjugate_coordinates(mol, seed)
        if parameter_source == "generated-geometry"
        else None
    )
    unmapped = Chem.Mol(mol)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)
    chemical_smiles = Chem.MolToSmiles(Chem.RemoveHs(unmapped))
    for ra, ca, first, rb, cb, second in pairs:
        rts, names = (first, second), (ca, cb)
        maps = (mappings[ra, first.name], mappings[rb, second.name])
        if (first.name, ca) > (second.name, cb):
            rts, names, maps = rts[::-1], names[::-1], maps[::-1]
        record = _connection_record(
            mol, props, *rts, maps, names, [adjacency[rt.name] for rt in rts], coords
        )
        yield record, chemical_smiles


def _model_identity(model):
    # Include every chemistry annotation consumed by the RDKit converter,
    # plus names and relative residue/cap identity used to map parameter rows.
    # Coordinates are discarded after observing stereochemistry. Chain numbers
    # and PDB metadata do not affect the generator's chemical assignments.
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
        (
            Chem.MolToSmiles(model.molecule),
            tuple(
                (name, str(values.dtype), values.shape, values.tobytes())
                for name, values in arrays.items()
            ),
        )
    )


def generate_conjugate_connection_params(
    atom_array,
    parameter_database,
    *,
    ph=7.4,
    seed=0,
    parameter_source="generated-geometry",
    existing=(),
    _model_consumer=None,
):
    """Generate complete two-block bond/angle records for prepared attachments.

    Exact patched names remain separate, including terminal combinations. A
    repeated residue/site identity with incompatible chemistry raises;
    this function does not rename input residues or invent context selection.
    Existing non-conjugation connections are not assigned new records. Pairs
    covered by ``existing`` records retain those parameters without regeneration.
    The default source uses generated ideal geometry and the ordinary ligand
    Cartesian constants. ``mmff94-harmonic`` is a private diagnostic alternative.
    """
    if not math.isfinite(ph):
        raise ValueError("Conjugate protonation pH must be finite")
    if parameter_source not in ("generated-geometry", "mmff94-harmonic"):
        raise ValueError(f"Unknown conjugate parameter source: {parameter_source}")
    chem = parameter_database.chemical
    candidates, adjacency = defaultdict(list), {}
    for rt in chem.residues:
        sites = frozenset(
            c.name for c in rt.connections if c.name.startswith(CONNECTION_PREFIX)
        )
        candidates[rt.base_name, sites].append(rt)
        if rt.io_equiv_class != rt.base_name:
            candidates[rt.io_equiv_class, sites].append(rt)
    if not any(sites for _, sites in candidates):
        return ()
    for alias in chem.name3_aliases:
        for (base, sites), types in list(candidates.items()):
            if base == alias.read_as:
                candidates.setdefault((alias.name3, sites), types)
    elements = {a.name: a.element for a in chem.atom_types}
    records, contexts, provenance = {}, {}, defaultdict(set)
    existing = frozenset(
        frozenset(((r.block_type1, r.connection1), (r.block_type2, r.connection2)))
        for r in existing
    )
    seen = set()
    for model in iter_capped_conjugate_models(atom_array, chem):
        identity = _model_identity(model)
        if identity in seen:
            continue
        for record, smiles in _model_records(
            model,
            candidates,
            elements,
            adjacency,
            contexts,
            ph,
            _model_consumer,
            parameter_source=parameter_source,
            seed=seed,
            existing=existing,
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

    if parameter_source == "generated-geometry":
        from openbabel import openbabel

        method = {
            "method": "tmol-generated-geometry-cartbonded-v1",
            "openbabel": openbabel.OBReleaseVersion(),
            "seed": seed,
            "length_K": GENERATED_LENGTH_K,
            "angle_K": GENERATED_ANGLE_K,
        }
    else:
        method = {
            "method": "tmol-mmff94-harmonic-v1",
            "approximation": "equilibrium curvature; no anharmonic or stretch-bend terms",
        }
    return tuple(
        evolve(
            records[key],
            provenance=json.dumps(
                {
                    **method,
                    "rdkit": rdBase.rdkitVersion,
                    "ph": ph,
                    "protonation": protonation,
                    "model": "complete conjugate with polymer caps",
                    "capped_smiles": sorted(provenance[key]),
                },
                sort_keys=True,
            ),
        )
        for key in sorted(records)
    )
