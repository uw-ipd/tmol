"""Chemical model, independent RDKit curvature and native attachment checks."""

import json
import math
from dataclasses import replace

import attr
import biotite.structure as struc
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.database.scoring import CartBondedDatabase
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.ligand._connection_params import (
    MMFF_HARMONIC_CONVERSION,
    generate_conjugate_connection_params,
)
from tmol.ligand import _connection_params
from tmol.ligand import prepare_ligands, load_params_file, write_params_file
from tmol.ligand._registry import inject_ligand_preparations
from tmol.tests.ligand import test_conjugate_model
from tmol.tests.score.cartbonded.test_explicit_connection_parameters import (
    scoring_term,
)

conjugate_input = test_conjugate_model.conjugate_input


def _only_mmff_term(mol, term):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    assert props is not None
    for name in ("Bond", "Angle", "StretchBend", "Oop", "Torsion", "VdW", "Ele"):
        getattr(props, f"SetMMFF{name}Term")(name == term)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
    force_field = AllChem.MMFFGetMoleculeForceField(mol, props)
    assert force_field is not None
    return props, force_field


@pytest.mark.parametrize(
    "smiles,term", [("O", "Bond"), ("O", "Angle"), ("C#N", "Angle")]
)
def test_harmonic_units_against_rdkit_energy_and_force_curvature(smiles, term):
    # Water gives one angle and two bonds. Moving one H radially changes just
    # one enabled bond; moving it on a unit circle changes just the angle.
    # HCN additionally exercises MMFF's special linear-angle expression.
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    props, force_field = _only_mmff_term(mol, term)
    if term == "Bond":
        _, constant, target = props.GetMMFFBondStretchParams(mol, 0, 1)
    else:
        _, constant, target = props.GetMMFFAngleBendParams(mol, 1, 0, 2)
        target = math.radians(target)

    def sample(delta):
        value = target + delta
        if term == "Bond":
            xyz = np.array([[0.0, 0.0, 0.0], [value, 0.0, 0.0], [0.0, 1.0, 0.0]])
            direction = np.array([1.0, 0.0, 0.0])
        else:
            xyz = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [math.cos(value), math.sin(value), 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            direction = np.array([-math.sin(value), math.cos(value), 0.0])
        energy = force_field.CalcEnergy(xyz.ravel().tolist())
        gradient = np.array(force_field.CalcGrad(xyz.ravel().tolist())).reshape(-1, 3)
        return energy, gradient[1].dot(direction)

    step = 1e-4
    lower, center, upper = [sample(x) for x in (-step, 0.0, step)]
    expected = MMFF_HARMONIC_CONVERSION * constant
    assert center[1] == pytest.approx(0.0, abs=1e-9)
    assert (upper[0] - 2 * center[0] + lower[0]) / step**2 == pytest.approx(
        expected, rel=3e-7
    )
    assert (upper[1] - lower[1]) / (2 * step) == pytest.approx(expected, rel=3e-7)


def test_generated_records_ignore_coordinates_and_instance_numbering(conjugate_input):
    _, original, database = conjugate_input
    expected = generate_conjugate_connection_params(original, database)
    assert expected
    second = original.copy()
    second.chain_id[:] = "ZZ"
    second.res_id += 1000
    second.coord[:] = np.nan
    # Reverse atom order within each residue as well. Identity must follow
    # names/bonds through conversion and protonation rather than RDKit order.
    starts = struc.get_residue_starts(second, add_exclusive_stop=True)
    indices = np.concatenate(
        [np.arange(b - 1, a - 1, -1) for a, b in zip(starts[:-1], starts[1:])]
    )
    second = second[indices]
    assert generate_conjugate_connection_params(second, database) == expected
    assert generate_conjugate_connection_params(original + second, database) == expected
    for record in expected:
        provenance = json.loads(record.provenance)
        assert provenance["method"] == "tmol-mmff94-harmonic-v1"
        assert provenance["ph"] == 7.4
        assert provenance["protonation"]["selection"] == "first ordered variant"
        assert len(provenance["protonation"]["rules_sha256"]) == 64
        assert provenance["capped_smiles"]
        assert len(record.length_parameters) == 1
        for row in (*record.length_parameters, *record.angle_parameters):
            assert math.isfinite(row.K) and row.K > 0
            assert math.isfinite(row.x0) and row.x0 > 0


def test_reused_residue_name_with_changed_attachment_chemistry_raises(conjugate_input):
    fixture, original, database = conjugate_input
    if fixture != "biotin":
        pytest.skip("The amide-to-hemiaminal case extends biotin")
    second = original.copy()
    second.chain_id[:] = "ZZ"
    carbon = int(
        np.flatnonzero((second.res_name == "BTN") & (second.atom_name == "C11"))[0]
    )
    neighbors, orders = second.bonds.get_bonds(carbon)
    oxygen = next(
        int(i)
        for i, order in zip(neighbors, orders)
        if order == struc.BondType.DOUBLE and second.element[i] == "O"
    )
    second.bonds.remove_bond(carbon, oxygen)
    second.bonds.add_bond(carbon, oxygen, struc.BondType.SINGLE)
    with pytest.raises(
        ValueError, match="Incompatible conjugate chemistry shares residue identity"
    ):
        generate_conjugate_connection_params(original + second, database)


def test_attachment_hydrogen_inventory_must_match_patched_type(conjugate_input):
    fixture, original, database = conjugate_input
    if fixture != "biotin":
        pytest.skip("This hydrogen inventory check uses the biotin amide")
    residues = tuple(
        (
            attr.evolve(
                rt,
                atoms=tuple(a for a in rt.atoms if a.name != "HZ1"),
                bonds=tuple(b for b in rt.bonds if "HZ1" not in b[:2]),
            )
            if rt.base_name == "LYS" and "conj_NZ" in rt.name
            else rt
        )
        for rt in database.chemical.residues
    )
    bad_database = attr.evolve(
        database, chemical=attr.evolve(database.chemical, residues=residues)
    )
    with pytest.raises(ValueError, match="Attachment neighborhood differs"):
        generate_conjugate_connection_params(original, bad_database)


def test_duplicate_groups_parameterized_once_per_call(conjugate_input, monkeypatch):
    fixture, original, database = conjugate_input
    copies = []
    for i in range(3):
        copy = original.copy()
        copy.res_id += 10000 * i
        copies.append(copy)
    combined = struc.concatenate(copies)
    parameterize = _connection_params._parameterized_model
    calls = []

    def counted(model, ph):
        calls.append(ph)
        return parameterize(model, ph)

    monkeypatch.setattr(_connection_params, "_parameterized_model", counted)
    count = 2 if fixture == "nglycan" else 1
    generate_conjugate_connection_params(combined, database)
    assert calls == [7.4] * count
    # No cached molecule/assignment survives the call or a pH change.
    generate_conjugate_connection_params(combined, database, ph=6.4)
    assert calls == [7.4] * count + [6.4] * count


def test_generated_record_bundle_preserves_model_provenance(conjugate_input, tmp_path):
    _, array, database = conjugate_input
    records = generate_conjugate_connection_params(array, database)
    path = tmp_path / "generated.tmol"
    prepare_ligands(array, seed=20250828, params_output=str(path))
    preps = load_params_file(path)
    preps[0] = replace(preps[0], connection_params=records)
    write_params_file(preps, path, format="tmol")
    restored = load_params_file(path)
    assert restored[0].connection_params == records
    for base in (ParameterDatabase.get_default(), database):
        enriched = inject_ligand_preparations(base, restored)
        assert enriched.scoring.cartbonded.connection_params == records
        assert generate_conjugate_connection_params(array, enriched) == records
        assert inject_ligand_preparations(enriched, restored) is enriched


def test_attachment_bond_order_must_match_patched_type(conjugate_input):
    _, array, database = conjugate_input
    residues = tuple(
        attr.evolve(
            rt,
            connections=tuple(
                attr.evolve(c, type="DOUBLE") if c.name.startswith("conj_") else c
                for c in rt.connections
            ),
        )
        for rt in database.chemical.residues
    )
    bad_database = attr.evolve(
        database, chemical=attr.evolve(database.chemical, residues=residues)
    )
    with pytest.raises(ValueError, match="Attachment bond order differs"):
        generate_conjugate_connection_params(array, bad_database)


def _pose_record_paths(pose, records):
    by_key = {
        (r.block_type1, r.connection1, r.block_type2, r.connection2): r for r in records
    }
    connections = pose.inter_residue_connections.cpu().tolist()
    offsets = pose.block_coord_offset.cpu().tolist()
    block_types = pose.packed_block_types.active_block_types
    types = [
        [block_types[t] if t >= 0 else None for t in row]
        for row in pose.block_type_ind.cpu().tolist()
    ]
    paths = []
    for pi, row in enumerate(types):
        for bi, bt in enumerate(row):
            if bt is None:
                continue
            for ci, conn in enumerate(bt.connections):
                bj, cj = connections[pi][bi][ci]
                if bj <= bi:
                    continue
                other = row[bj]
                pair = (bt.name, conn.name, other.name, other.connections[cj].name)
                reverse = (*pair[2:], *pair[:2])
                if pair in by_key:
                    record, blocks = by_key[pair], (bi, bj)
                elif reverse in by_key:
                    record, blocks = by_key[reverse], (bj, bi)
                else:
                    continue
                terms = []
                for kind, rows in enumerate(
                    (record.length_parameters, record.angle_parameters)
                ):
                    for parameter in rows:
                        indices = []
                        for field in ("atm1", "atm2", "atm3")[: kind + 2]:
                            name = getattr(parameter, field)
                            side = name.startswith("+")
                            name = name[1:] if side else name
                            block = blocks[side]
                            indices.append(
                                offsets[pi][block] + row[block].atom_to_idx[name]
                            )
                        # Scoring tables store float32 even with double coords.
                        terms.append(
                            (
                                kind,
                                indices,
                                float(np.float32(parameter.x0)),
                                float(np.float32(parameter.K)),
                            )
                        )
                paths.append((pi, bi, bj, terms))
    return paths


def _reference(coords, paths, pair_weights):
    terms = [coords.new_zeros(()), coords.new_zeros(())]
    for pi, bi, bj, rows in paths:
        for kind, indices, target, constant in rows:
            xyz = coords[pi, indices]
            if kind == 0:
                value = (xyz[0] - xyz[1]).norm()
            else:
                first, second = xyz[0] - xyz[1], xyz[2] - xyz[1]
                value = torch.atan2(
                    torch.linalg.cross(first, second).norm(), first.dot(second)
                )
            terms[kind] += (
                pair_weights[pi, bi, bj] * 0.5 * constant * (value - target).square()
            )
    return torch.stack(terms)


@pytest.mark.parametrize("block_pairs", [False, True])
def test_generated_attachment_energy_and_gradients(
    conjugate_input, torch_device, block_pairs
):
    fixture, array, database = conjugate_input
    records = generate_conjugate_connection_params(array, database)
    pose = pose_stack_from_biotite(array, torch_device, param_db=database, no_optH=True)
    # Isolate generated terms, so a legacy wildcard cannot mask missing rows.
    isolated = attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            cartbonded=CartBondedDatabase.from_cartres_dict({}, records),
        ),
    )
    term = scoring_term(pose, isolated)
    module = (
        term.render_block_pair_scoring_module
        if block_pairs
        else term.render_whole_pose_scoring_module
    )(pose)
    coords = pose.coords.double().clone()
    coords += 0.03 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    paths = _pose_record_paths(pose, records)
    assert len(paths) == {"biotin": 1, "nglycan": 8, "oglycan": 6}[fixture]
    assert (
        sum(kind == 1 for _, _, _, rows in paths for kind, *_ in rows)
        == {
            "biotin": 4,
            "nglycan": 33,
            "oglycan": 24,
        }[fixture]
    )
    weights = coords.new_ones((pose.n_poses, pose.max_n_blocks, pose.max_n_blocks))
    values = module(coords)
    if block_pairs:
        weights += 0.2 * torch.sin(
            torch.arange(weights.numel(), device=torch_device)
        ).reshape_as(weights)
        actual = (values[:2] * weights).sum(dim=(1, 2, 3))
    else:
        actual = values[:2].sum(dim=1)
    expected = _reference(coords, paths, weights)
    assert torch.all(expected > 0)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    term_weights = coords.new_tensor([0.7, 1.9])
    actual_grad = torch.autograd.grad(
        (actual * term_weights).sum(), coords, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((expected * term_weights).sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)
