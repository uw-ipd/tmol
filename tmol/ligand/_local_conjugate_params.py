"""Local conjugate corrections from connected/disconnected capped models.

The generator-ideals source uses ordinary ligand Cartesian constants and keeps
the patched residue charges. The private MMFF harmonic diagnostic adds charge
deltas and uses equilibrium curvatures. Both share typing and construction.
Comparison fragments are valence-completed references, not physical reactants.
Default preparation installs the generator-ideals corrections and exports them
as guarded replacements.
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
    charge_model: str = "conserved-patched-residue-v1"


class ConjugateContextConflict(ValueError):
    def __init__(self, residue_name):
        self.residue_name = residue_name
        super().__init__(f"Conflicting local conjugate parameters for {residue_name}")


def _disconnected_model(model):
    array = model.atom_array.copy()
    molecule = Chem.RWMol(model.molecule)
    # Explicit bond orders let detached aromatic nitrogens recover their H.
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    mapping = {int(i): j for j, i in enumerate(model.source_atom_indices) if i >= 0}
    for a, b, _ in model.connections:
        array.bonds.remove_bond(mapping[a], mapping[b])
        molecule.RemoveBond(mapping[a], mapping[b])
        # Detached comparison fragments complete the endpoints' open valences.
        for index in (mapping[a], mapping[b]):
            molecule.GetAtomWithIdx(index).SetNoImplicit(False)
    Chem.SanitizeMol(molecule)
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


def _correct_bonds(rt, mol, mapping):
    """Localize surviving non-ring bonds after attachment, retaining unmapped caps."""
    bonds, bond_changes = [], set()
    for first, second, order, *flags in rt.bonds:
        if first not in mapping or second not in mapping:
            bonds.append((first, second, order, *flags))
            continue
        bond = mol.GetBondBetweenAtoms(mapping[first], mapping[second])
        target = str(bond.GetBondType()) if not bond.GetIsAromatic() else order
        if target != order:
            bond_changes.update((first, second))
        bonds.append((first, second, target, *flags))
    return tuple(bonds), bond_changes


def _correct_atoms(
    mol,
    free_mol,
    typing,
    free_typing,
    mapping,
    free_mapping,
    rt,
    neighbors,
    rosetta,
    *,
    preserve_conjugate_types,
    bond_changes,
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
    local = roots | changed | bond_changes
    for _ in range(2):
        local |= {n for name in tuple(local) for n in neighbors[name]}
    atoms = []
    for atom in rt.atoms:
        if atom.name not in local or atom.name not in mapping:
            atoms.append(atom)
            continue
        i, j = mapping[atom.name], free_mapping.get(atom.name)
        # Conjugation patches already select the database's physical type for
        # their attachment atom. The connected model supplies its generic
        # torsion reference without replacing that explicit Rosetta choice.
        physical = atom.atom_type
        if (
            atom.name in bond_changes
            or (atom.name in roots and not preserve_conjugate_types)
            or physical not in rosetta
            or (
                atom.name not in roots
                and j is not None
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
    *,
    parameter_source,
    bond_changes,
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
            if full != free or bond_changes.intersection(path):
                changed[size][min(path, path[::-1])] = rowtype(
                    *path,
                    x0=full[2] if size == 2 else math.radians(full[2]),
                    K=(
                        conversion * full[1]
                        if parameter_source == "mmff94-harmonic"
                        else (
                            connection.GENERATED_LENGTH_K
                            if size == 2
                            else connection.GENERATED_ANGLE_K
                        )
                    ),
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
    """Use local atom targets and residue-local connection frames for construction.

    Partner-specific bond geometry belongs to connection-pair records. Group
    packing measures those bonds from the pose, not the virtual port coordinates.
    """
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
    chemical_type = {a.name: a.genbonded_type or a.atom_type for a in atoms}
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
    for conn in rt.connections:
        if not conn.name.startswith(CONNECTION_PREFIX):
            continue
        root = mapping[conn.atom]
        ic = icoors[conn.name]
        heavy = [
            n
            for n in neighbors[conn.atom]
            if mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() > 1
        ]
        grand = ic.grand_parent if ic.grand_parent in heavy else next(iter(heavy), None)
        if grand is None:
            if chemical_type[conn.atom] != "Nad":
                # A terminal one-heavy-atom component gets its hydrogen plane
                # from the partner, like a one-heavy-atom polymer cap.
                if len(neighbors[conn.atom]) == 1:
                    hydrogen = neighbors[conn.atom][0]
                    icoors[hydrogen] = attr.evolve(
                        icoors[hydrogen],
                        grand_parent=conn.name,
                        great_grand_parent=conn.atom,
                        theta=ic.theta,
                    )
                continue
            raise ValueError(
                f"No heavy attachment construction frame at {rt.name},{conn.name}"
            )
        hydrogens = [
            n
            for n in neighbors[conn.atom]
            if mol.GetAtomWithIdx(mapping[n]).GetAtomicNum() == 1
        ]
        if grand != ic.grand_parent or (
            chemical_type[conn.atom] == "Nad" and ic.great_grand_parent in hydrogens
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
        icoors[conn.name] = ic
        # A one-H amide must use the actual partner connection as its plane
        # reference. A frame on the old amine's sidechain cannot enforce this.
        if chemical_type[conn.atom] == "Nad":
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


def _corrected_residue_names(connection_params):
    previously_corrected = set()
    for record in connection_params:
        try:
            metadata = json.loads(record.provenance)
        except (ValueError, TypeError):
            continue
        if isinstance(metadata, dict) and "local_conjugate" in metadata:
            previously_corrected.update((record.block_type1, record.block_type2))
    return previously_corrected


def generate_conjugate_parameters(
    atom_array,
    parameter_database,
    *,
    ph=7.4,
    parameter_source="generator-ideals",
    existing=(),
    _models=None,
):
    """Generate consistent local and connection parameters, without installing.

    ``generator-ideals`` retains the already patched charges and uses the
    ordinary ligand bond/angle constants. The private ``mmff94-harmonic`` source
    adds connected-minus-disconnected charge deltas, undoing reference-H charges
    folded onto parents, and rejects artificial-cap charge changes.
    Connections in ``existing`` retain their supplied parameters and protect
    both endpoint residue types from local changes. Other residue types must
    have an uncorrected baseline.
    """
    if parameter_source not in ("generator-ideals", "mmff94-harmonic"):
        raise ValueError(f"Unknown conjugate parameter source: {parameter_source}")
    charge_delta = parameter_source == "mmff94-harmonic"
    existing = tuple(existing)
    protected = {name for r in existing for name in (r.block_type1, r.block_type2)}
    elements = {a.name: a.element for a in parameter_database.chemical.atom_types}
    charge_index = {
        (p.res, p.atom): p.charge
        for p in parameter_database.scoring.elec.atom_charge_parameters
    }
    rosetta = parameter_database.scoring.genbonded.rosetta_typed
    rows = {}
    previously_corrected = _corrected_residue_names(
        parameter_database.scoring.cartbonded.connection_params
    )

    def consume(model, mol, props, heavy_map, candidates, mappings, adjacency):
        candidates = {
            ri: [rt for rt in types if rt.name not in protected]
            for ri, types in candidates.items()
        }
        if not any(candidates.values()):
            return
        before = connection._parameterized_model(_disconnected_model(model), ph)
        free_mol, free_props, free_heavy = before
        typing, free_typing = _typing(mol), _typing(free_mol)
        cap_delta = 0.0
        for local, source in (
            enumerate(model.source_atom_indices) if charge_delta else ()
        ):
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
                baseline_charges = _baseline_charges(charge_index, rt)
                charges = baseline_charges.copy() if charge_delta else baseline_charges
                for name, i in mapping.items() if charge_delta else ():
                    if name not in free_mapping:
                        continue
                    delta = (
                        props.GetMMFFPartialCharge(i)
                        - free_props.GetMMFFPartialCharge(free_mapping[name])
                        - removed.get(name, 0.0)
                    )
                    if abs(delta) > 1e-12:
                        charges[name] += delta
                bonds, bond_changes = _correct_bonds(rt, mol, mapping)
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
                    preserve_conjugate_types=not charge_delta,
                    bond_changes=bond_changes,
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
                    parameter_source=parameter_source,
                    bond_changes=bond_changes,
                )
                icoors = _correct_icoors(
                    rt, mol, props, mapping, neighbors, atoms, bonded, baseline
                )
                row = LocalConjugateParameters(
                    attr.evolve(rt, atoms=atoms, bonds=bonds, icoors=icoors),
                    charges,
                    bonded,
                    _local_identity(rt, baseline_charges, baseline),
                )
                if rt.name in rows and rows[rt.name] != row:
                    raise ConjugateContextConflict(rt.name)
                rows[rt.name] = row

    connections = connection.generate_conjugate_connection_params(
        atom_array,
        parameter_database,
        ph=ph,
        _model_consumer=consume,
        parameter_source=parameter_source,
        existing=existing,
        _models=_models,
    )
    result = ConjugateParameters(
        tuple(rows[name] for name in sorted(rows)),
        (),
        charge_model=(
            "curated-baseline-plus-mmff94-delta-v1"
            if charge_delta
            else "conserved-patched-residue-v1"
        ),
    )
    annotated = []
    for record in connections:
        baselines = {
            name: rows[name].baseline_sha256
            for name in (record.block_type1, record.block_type2)
            if name in rows
        }
        if not baselines:
            annotated.append(record)
            continue
        metadata = json.loads(record.provenance)
        metadata["local_conjugate"] = {
            "charge_model": result.charge_model,
            "baseline_sha256": baselines,
        }
        annotated.append(
            attr.evolve(record, provenance=json.dumps(metadata, sort_keys=True))
        )
    return replace(result, connections=tuple(annotated))
