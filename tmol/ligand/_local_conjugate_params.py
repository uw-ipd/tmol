"""Local conjugate corrections from connected/disconnected capped models.

This private generator preserves the curated baseline and adds MMFF charge
changes; changed local bond/angle terms use MMFF equilibrium curvatures. The
comparison fragments are valence-completed references, not physical reactants.
It is not yet installed by default preparation.
"""

from dataclasses import dataclass, replace
from itertools import combinations
from graphlib import TopologicalSorter
import json
import math

import attr
import numpy as np
from rdkit import Chem

from tmol.database.chemical import RawResidueType
from tmol.database.scoring import AngleGroup, CartRes, ConnectionCartRes, LengthGroup
from tmol.ligand import _connection_params as connection
from tmol.ligand._atom_typing import assign_tmol_atom_types
from tmol.ligand._conjugation_patches import CONNECTION_PREFIX
from tmol.ligand._parameter_replacements import _baseline_charges, _local_identity


@dataclass(frozen=True)
class LocalConjugateParameters:
    residue_type: RawResidueType
    partial_charges: dict[str, float]
    cartbonded_params: CartRes
    baseline_sha256: str


@dataclass(frozen=True)
class ConjugateParameters:
    residues: tuple[LocalConjugateParameters, ...]
    connections: tuple[ConnectionCartRes, ...]
    charge_model: str = "curated-baseline-plus-mmff94-delta-v1"


def _disconnected_model(model):
    array = model.atom_array.copy()
    molecule = Chem.RWMol(model.molecule)
    mapping = {int(i): j for j, i in enumerate(model.source_atom_indices) if i >= 0}
    for a, b, _ in model.connections:
        array.bonds.remove_bond(mapping[a], mapping[b])
        molecule.RemoveBond(mapping[a], mapping[b])
    molecule.UpdatePropertyCache(strict=False)
    return replace(model, atom_array=array, molecule=molecule.GetMol())


def _hydrogens(mol, index):
    return sorted(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(index).GetNeighbors()
        if n.GetAtomicNum() == 1
    )


def _atomic_state(atom):
    return (
        atom.GetFormalCharge(),
        str(atom.GetHybridization()),
        atom.GetIsAromatic(),
        tuple(sorted(str(b.GetBondType()) for b in atom.GetBonds())),
    )


def _typing(mol):
    # The classifier may update perception; retain the force-field molecule.
    return {a.index: a.atom_type for a in assign_tmol_atom_types(Chem.Mol(mol))}


def _equivalent_hydrogens(mol, props, indices):
    # Generated H atoms have no persistent names. Matching them by a parent
    # is justified only when that parent's H atoms have equivalent parameters.
    states = {
        (props.GetMMFFAtomType(i), round(props.GetMMFFPartialCharge(i), 12))
        for i in indices
    }
    if len(states) > 1:
        raise ValueError("Conjugate reference hydrogens are not parameter-equivalent")


def _reference_mapping(
    mol, props, heavy_map, before, mapping, restype, neighbors, elements
):
    free_mol, free_props, free_heavy = before
    to_source = {v: k for k, v in heavy_map.items()}
    free_mapping, removed = {}, {}
    atom_elements = {a.name: elements[a.atom_type] for a in restype.atoms}
    for name, index in mapping.items():
        if mol.GetAtomWithIdx(index).GetAtomicNum() == 1:
            continue
        other = free_heavy[to_source[index]]
        free_mapping[name] = other
        full_h, free_h = _hydrogens(mol, index), _hydrogens(free_mol, other)
        _equivalent_hydrogens(mol, props, full_h)
        _equivalent_hydrogens(free_mol, free_props, free_h)
        names = sorted(n for n in neighbors[name] if atom_elements[n] == "H")
        if len(free_h) < len(full_h):
            raise ValueError(
                f"Conjugate chemistry adds hydrogens at {restype.name},{name}"
            )
        if len(names) == len(full_h):
            free_mapping.update(zip(names, free_h))
        removed[name] = sum(
            free_props.GetMMFFPartialCharge(i) for i in free_h[len(full_h) :]
        )
    return free_mapping, removed


def install_conjugate_parameters(parameter_database, result):
    """Install private generated parameters with the shared baseline guard."""
    from tmol.ligand._parameter_replacements import install_replacements

    return install_replacements(parameter_database, result.residues, result.connections)


def _correct_atoms(
    mol, free_mol, typing, free_typing, mapping, free_mapping, rt, neighbors, rosetta
):
    roots = {c.atom for c in rt.connections if c.name.startswith(CONNECTION_PREFIX)}
    original_types = {a.name: a.atom_type for a in rt.atoms}
    changed = {
        name
        for name, i in mapping.items()
        if name in free_mapping
        and (
            (
                original_types[name] not in rosetta
                and typing[i] != free_typing[free_mapping[name]]
            )
            or _atomic_state(mol.GetAtomWithIdx(i))
            != _atomic_state(free_mol.GetAtomWithIdx(free_mapping[name]))
        )
    }
    # Every torsion involving a changed central atom has its endpoints within
    # two bonds. Include H neighbors so amide impropers can match HN.
    local = roots | changed
    for _ in range(2):
        local |= {n for name in tuple(local) for n in neighbors[name]}
    atoms = []
    for atom in rt.atoms:
        if atom.name not in local or atom.name not in mapping:
            atoms.append(atom)
            continue
        i, j = mapping[atom.name], free_mapping.get(atom.name)
        physical = atom.atom_type
        if (
            atom.name in roots
            or physical not in rosetta
            or (
                j is not None
                and _atomic_state(mol.GetAtomWithIdx(i))
                != _atomic_state(free_mol.GetAtomWithIdx(j))
            )
        ):
            physical = typing[i]
        atoms.append(
            attr.evolve(
                atom,
                atom_type=physical,
                genbonded_type=typing[i] if typing[i] != physical else None,
            )
        )
    return tuple(atoms)


def _correct_bonded(
    mol, props, before, mapping, free_mapping, rt, atoms, neighbors, baseline, rosetta
):
    free_mol, free_props, _ = before
    conversion = connection.MMFF_HARMONIC_CONVERSION
    changed = {2: {}, 3: {}}
    paths = (
        (name, other)
        for name, adjacent in neighbors.items()
        for other in adjacent
        if name < other
    )
    angles = (
        (a, center, b)
        for center, adjacent in neighbors.items()
        for a, b in combinations(adjacent, 2)
    )
    for size, source, method, rowtype in (
        (2, paths, "GetMMFFBondStretchParams", LengthGroup),
        (3, angles, "GetMMFFAngleBendParams", AngleGroup),
    ):
        for path in source:
            if any(n not in mapping or n not in free_mapping for n in path):
                continue
            full = getattr(props, method)(mol, *(mapping[n] for n in path))
            free = getattr(free_props, method)(
                free_mol, *(free_mapping[n] for n in path)
            )
            if full is None or free is None:
                raise ValueError(f"Missing local MMFF parameter for {rt.name},{path}")
            if full != free:
                changed[size][min(path, path[::-1])] = rowtype(
                    *path,
                    x0=full[2] if size == 2 else math.radians(full[2]),
                    K=conversion * full[1],
                )
    present = {a.name for a in atoms}
    physical = {a.name: a.atom_type for a in atoms}
    old_types = {a.name: a.atom_type for a in rt.atoms}
    transferred = {
        n for n in present if old_types[n] in rosetta and physical[n] not in rosetta
    }
    fields = {}
    for field, size in (
        ("length_parameters", 2),
        ("angle_parameters", 3),
        ("torsion_parameters", 4),
        ("hxltorsion_parameters", 4),
        ("improper_parameters", 4),
    ):
        retained = []
        for row in getattr(baseline, field):
            path = tuple(getattr(row, f"atm{i}") for i in range(1, size + 1))
            if any(not n.startswith("+") and n not in present for n in path):
                continue
            if size < 4 and min(path, path[::-1]) in changed[size]:
                continue
            ownership = path[2:3] if field == "improper_parameters" else path[1:3]
            if size == 4 and transferred.intersection(ownership):
                continue
            retained.append(row)
        if size < 4:
            retained.extend(changed[size][key] for key in sorted(changed[size]))
        fields[field] = tuple(retained)
    return attr.evolve(baseline, **fields)


def _correct_icoors(rt, mol, props, mapping, neighbors, atoms, bonded, baseline):
    """Use the same targets for construction, including planar retained H."""
    previous_lengths = set(baseline.length_parameters)
    previous_angles = set(baseline.angle_parameters)
    lengths = {
        frozenset((r.atm1, r.atm2)): r.x0
        for r in bonded.length_parameters
        if r not in previous_lengths
    }
    angles = {
        min((r.atm1, r.atm2, r.atm3), (r.atm3, r.atm2, r.atm1)): r.x0
        for r in bonded.angle_parameters
        if r not in previous_angles
    }
    icoors = {ic.name: ic for ic in rt.icoors}
    physical = {a.name: a.atom_type for a in atoms}
    for ic in rt.icoors:
        values = {}
        length = lengths.get(frozenset((ic.name, ic.parent)))
        if length is not None:
            values["d"] = length
        key = (ic.name, ic.parent, ic.grand_parent)
        angle = angles.get(min(key, key[::-1]))
        if angle is not None:
            values["theta"] = math.pi - angle
        if values:
            icoors[ic.name] = attr.evolve(ic, **values)
    mapped = set(mapping.values())
    for conn in rt.connections:
        if not conn.name.startswith(CONNECTION_PREFIX):
            continue
        root = mapping[conn.atom]
        remote = [
            a.GetIdx()
            for a in mol.GetAtomWithIdx(root).GetNeighbors()
            if a.GetIdx() not in mapped
        ]
        if len(remote) != 1:
            raise ValueError(
                f"Ambiguous attachment construction frame at {rt.name},{conn.name}"
            )
        remote = remote[0]
        ic = icoors[conn.name]
        heavy = [
            n
            for n in neighbors[conn.atom]
            if mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() > 1
        ]
        grand = ic.grand_parent if ic.grand_parent in heavy else next(iter(heavy), None)
        if grand is None:
            raise ValueError(
                f"No heavy attachment construction frame at {rt.name},{conn.name}"
            )
        hydrogens = [
            n
            for n in neighbors[conn.atom]
            if mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() == 1
        ]
        if grand != ic.grand_parent or (
            physical[conn.atom] == "Nad" and ic.great_grand_parent in hydrogens
        ):
            # Keep the template's attachment direction while replacing an H
            # frame dependency by a stable heavy neighbor. This avoids an
            # H -> connection -> H construction cycle for an acylated amine.
            from tmol.chemical._ideal_coords import (
                build_coords_from_icoors,
                frame_from_coords,
            )

            great = next(
                (
                    n
                    for n in neighbors[grand]
                    if n != conn.atom
                    and n in mapping
                    and mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() > 1
                ),
                None,
            )
            if great is None:
                raise ValueError(
                    f"No heavy attachment plane reference at {rt.name},{conn.name}"
                )
            names = {value.name: i for i, value in enumerate(rt.icoors)}
            ancestors = np.asarray(
                [
                    [
                        names[value.parent],
                        names[value.grand_parent],
                        names[value.great_grand_parent],
                    ]
                    for value in rt.icoors
                ],
                dtype=np.int32,
            )
            geometry = np.asarray(
                [[value.phi, value.theta, value.d] for value in rt.icoors]
            )
            xyz = build_coords_from_icoors(ancestors, geometry)
            frame = frame_from_coords(
                xyz[names[great]], xyz[names[grand]], xyz[names[conn.atom]]
            )
            vector = xyz[names[conn.name]] - xyz[names[conn.atom]]
            phi = math.atan2(
                -float(vector @ frame[:3, 0]), float(vector @ frame[:3, 1])
            )
            ic = attr.evolve(ic, grand_parent=grand, great_grand_parent=great, phi=phi)
        bond = props.GetMMFFBondStretchParams(mol, root, remote)
        angle = props.GetMMFFAngleBendParams(mol, mapping[grand], root, remote)
        icoors[conn.name] = attr.evolve(
            ic, d=bond[2], theta=math.pi - math.radians(angle[2])
        )
        # A one-H amide must use the actual partner connection as its plane
        # reference. A frame on the old amine's sidechain cannot enforce this.
        if physical[conn.atom] == "Nad":
            hydrogens = [
                n
                for n in neighbors[conn.atom]
                if mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() == 1
            ]
            if len(hydrogens) == 1 and len(heavy) == 1:
                name = hydrogens[0]
                bond = props.GetMMFFBondStretchParams(mol, root, mapping[name])
                angle = props.GetMMFFAngleBendParams(
                    mol, mapping[grand], root, mapping[name]
                )
                icoors[name] = attr.evolve(
                    icoors[name],
                    parent=conn.atom,
                    grand_parent=grand,
                    great_grand_parent=conn.name,
                    phi=math.pi,
                    theta=math.pi - math.radians(angle[2]),
                    d=bond[2],
                )
    # The original first three coordinates define the rigid frame. Reorder the
    # remaining construction dependencies so a connection can precede its H.
    anchors = {ic.name for ic in rt.icoors[:3]}
    graph = {
        name: {ic.parent, ic.grand_parent, ic.great_grand_parent} - anchors
        for name, ic in icoors.items()
        if name not in anchors
    }
    order = list(TopologicalSorter(graph).static_order())
    return tuple(icoors[ic.name] for ic in rt.icoors[:3]) + tuple(
        icoors[name] for name in order
    )


def generate_conjugate_parameters(atom_array, parameter_database, *, ph=7.4):
    """Generate consistent local and connection parameters, without installing.

    Charges add the connected-minus-disconnected MMFF correction to the already
    patched baseline, undoing the reference-H charges folded onto its parents.
    Unsupported H additions or nonzero artificial-cap charge changes raise.
    The caller must supply an uncorrected baseline, not a previously corrected database.
    """
    elements = {a.name: a.element for a in parameter_database.chemical.atom_types}
    charge_index = {
        (p.res, p.atom): p.charge
        for p in parameter_database.scoring.elec.atom_charge_parameters
    }
    rosetta = parameter_database.scoring.genbonded.rosetta_typed
    rows = {}
    previously_corrected = set()
    for record in parameter_database.scoring.cartbonded.connection_params:
        try:
            metadata = json.loads(record.provenance)
        except (ValueError, TypeError):
            continue
        if isinstance(metadata, dict) and "local_conjugate" in metadata:
            previously_corrected.update((record.block_type1, record.block_type2))

    def consume(model, mol, props, heavy_map, candidates, mappings, adjacency):
        before = connection._parameterized_model(_disconnected_model(model), ph)
        free_mol, free_props, free_heavy = before
        typing, free_typing = _typing(mol), _typing(free_mol)
        cap_delta = 0.0
        for local, source in enumerate(model.source_atom_indices):
            if source >= 0:
                continue
            i, j = heavy_map[local], free_heavy[local]
            cap_delta += props.GetMMFFPartialCharge(
                i
            ) - free_props.GetMMFFPartialCharge(j)
            cap_delta += sum(props.GetMMFFPartialCharge(h) for h in _hydrogens(mol, i))
            cap_delta -= sum(
                free_props.GetMMFFPartialCharge(h) for h in _hydrogens(free_mol, j)
            )
        if abs(cap_delta) > 1e-8:
            raise ValueError(
                "Local conjugate charge correction changes artificial cap charges"
            )
        for ri, types in candidates.items():
            for rt in types:
                if rt.name in previously_corrected:
                    raise ValueError(
                        f"Conjugate parameters already corrected for {rt.name}; "
                        "generate from the uncorrected baseline"
                    )
                mapping, neighbors = mappings[ri, rt.name], adjacency[rt.name]
                free_mapping, removed = _reference_mapping(
                    mol, props, heavy_map, before, mapping, rt, neighbors, elements
                )
                charges = _baseline_charges(charge_index, rt)
                for name, i in mapping.items():
                    if name not in free_mapping:
                        continue
                    delta = (
                        props.GetMMFFPartialCharge(i)
                        - free_props.GetMMFFPartialCharge(free_mapping[name])
                        - removed.get(name, 0.0)
                    )
                    if abs(delta) > 1e-12:
                        charges[name] += delta
                atoms = _correct_atoms(
                    mol,
                    free_mol,
                    typing,
                    free_typing,
                    mapping,
                    free_mapping,
                    rt,
                    neighbors,
                    rosetta,
                )
                records = parameter_database.scoring.cartbonded.residue_params
                baseline = records.get(rt.name, records.get(rt.base_name))
                if baseline is None:
                    raise ValueError(
                        f"Missing baseline bonded parameters for {rt.name}"
                    )
                bonded = _correct_bonded(
                    mol,
                    props,
                    before,
                    mapping,
                    free_mapping,
                    rt,
                    atoms,
                    neighbors,
                    baseline,
                    rosetta,
                )
                icoors = _correct_icoors(
                    rt, mol, props, mapping, neighbors, atoms, bonded, baseline
                )
                row = LocalConjugateParameters(
                    attr.evolve(rt, atoms=atoms, icoors=icoors),
                    charges,
                    bonded,
                    _local_identity(rt, _baseline_charges(charge_index, rt), baseline),
                )
                if rt.name in rows and rows[rt.name] != row:
                    raise ValueError(
                        f"Conflicting local conjugate parameters for {rt.name}"
                    )
                rows[rt.name] = row

    connections = connection.generate_conjugate_connection_params(
        atom_array, parameter_database, ph=ph, _model_consumer=consume
    )
    result = ConjugateParameters(tuple(rows[name] for name in sorted(rows)), ())
    annotated = []
    for record in connections:
        metadata = json.loads(record.provenance)
        metadata["local_conjugate"] = {
            "charge_model": result.charge_model,
            "baseline_sha256": {
                name: rows[name].baseline_sha256
                for name in (record.block_type1, record.block_type2)
            },
        }
        annotated.append(
            attr.evolve(record, provenance=json.dumps(metadata, sort_keys=True))
        )
    return replace(result, connections=tuple(annotated))
