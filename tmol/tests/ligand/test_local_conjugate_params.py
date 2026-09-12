"""Coupled attachment types, charges, bonded terms and H construction."""

import attr
import json
import numpy as np
import pytest
import torch

from tmol.database import inject_residue_params
from tmol.io import pose_stack_from_biotite
from tmol.ligand import _connection_params
from tmol.ligand._local_conjugate_params import (
    generate_conjugate_parameters,
    install_conjugate_parameters as install,
)
from tmol.tests.ligand import test_conjugate_model

conjugate_input = test_conjugate_model.conjugate_input


def _charges(database, rt):
    # Read exact/patch/base precedence independently of the generator.
    rows = {
        (p.res, p.atom): p.charge for p in database.scoring.elec.atom_charge_parameters
    }
    base, *patches = rt.name.split(":")
    names = list(dict.fromkeys([rt.name, *(base + ":" + p for p in patches), base]))
    return {
        a.name: next(rows[name, a.name] for name in names if (name, a.name) in rows)
        for a in rt.atoms
    }


def test_install_checks_baseline_and_does_not_apply_twice(conjugate_input):
    _, array, database = conjugate_input
    result = generate_conjugate_parameters(array, database)
    corrected = install(database, result)
    assert install(corrected, result) is corrected
    repeated = install(database, result)
    assert repeated.chemical == corrected.chemical
    assert repeated.scoring.cartbonded == corrected.scoring.cartbonded
    assert repeated.scoring.elec == corrected.scoring.elec
    assert database is not corrected
    with pytest.raises(ValueError, match="already corrected"):
        generate_conjugate_parameters(array, corrected)
    row = result.residues[0]
    name = row.residue_type.name
    atom = next(iter(row.partial_charges))
    changed = inject_residue_params(
        database, [], partial_charges={name: {atom: row.partial_charges[atom] + 0.125}}
    )
    with pytest.raises(ValueError, match="baseline changed"):
        install(changed, result)
    old = next(r for r in database.chemical.residues if r.name == name)
    changed_rt = attr.evolve(
        old,
        icoors=(attr.evolve(old.icoors[0], d=old.icoors[0].d + 0.01), *old.icoors[1:]),
    )
    changed = attr.evolve(
        database,
        chemical=attr.evolve(
            database.chemical,
            residues=tuple(
                changed_rt if r.name == name else r for r in database.chemical.residues
            ),
        ),
    )
    with pytest.raises(ValueError, match="baseline changed"):
        install(changed, result)


def test_local_bundle_roundtrip_preserves_correction_and_baseline(
    conjugate_input, tmp_path
):
    from dataclasses import replace
    from tmol.ligand import load_params_file, write_params_file
    from tmol.ligand._registry import LigandPreparation, inject_ligand_preparations

    _, array, database = conjugate_input
    result = generate_conjugate_parameters(array, database)
    preps = [
        LigandPreparation(
            residue_type=row.residue_type,
            partial_charges=row.partial_charges,
            cartbonded_params=row.cartbonded_params,
            connection_params=result.connections if i == 0 else (),
            baseline_sha256=row.baseline_sha256,
        )
        for i, row in enumerate(result.residues)
    ]
    path = tmp_path / "local.tmol"
    write_params_file(preps, path, format="tmol")
    restored = load_params_file(path)
    hashes = {}
    for connection in restored[0].connection_params:
        provenance = json.loads(connection.provenance)["local_conjugate"]
        assert provenance["charge_model"] == result.charge_model
        hashes.update(provenance["baseline_sha256"])
    reloaded = replace(
        result,
        residues=tuple(
            replace(
                row,
                residue_type=prep.residue_type,
                partial_charges=prep.partial_charges,
                cartbonded_params=prep.cartbonded_params,
                baseline_sha256=hashes[prep.residue_type.name],
            )
            for row, prep in zip(result.residues, restored, strict=True)
        ),
        connections=restored[0].connection_params,
    )
    assert reloaded == result
    corrected = install(database, reloaded)
    ordinary = inject_ligand_preparations(database, restored)
    assert ordinary.chemical == corrected.chemical
    assert ordinary.scoring.elec == corrected.scoring.elec
    assert ordinary.scoring.cartbonded == corrected.scoring.cartbonded
    assert inject_ligand_preparations(ordinary, restored) is ordinary
    assert install(corrected, reloaded) is corrected
    with pytest.raises(ValueError, match="already corrected"):
        generate_conjugate_parameters(array, corrected)


def test_generated_local_parameters_preserve_instance_identity(
    conjugate_input, monkeypatch
):
    fixture, original, database = conjugate_input
    parameterize = _connection_params._parameterized_model
    calls = []

    def counted(model, ph):
        calls.append(ph)
        return parameterize(model, ph)

    monkeypatch.setattr(_connection_params, "_parameterized_model", counted)
    expected = generate_conjugate_parameters(original, database)
    assert len(calls) == (4 if fixture == "nglycan" else 2)
    second = original.copy()
    second.res_id += 10000
    second.chain_id[:] = "ZZ"
    second.coord[:] = np.nan
    from biotite.structure import get_residue_starts

    boundaries = get_residue_starts(second, add_exclusive_stop=True)
    reversed_order = np.concatenate(
        [np.arange(b - 1, a - 1, -1) for a, b in zip(boundaries[:-1], boundaries[1:])]
    )
    assert generate_conjugate_parameters(second[reversed_order], database) == expected
    calls.clear()
    assert (
        generate_conjugate_parameters(original + second + original, database)
        == expected
    )
    assert len(calls) == (4 if fixture == "nglycan" else 2)


def test_biotin_local_changes_and_canonical_ownership(conjugate_input):
    fixture, array, database = conjugate_input
    if fixture != "biotin":
        pytest.skip("Detailed amide reference uses biotin")
    result = generate_conjugate_parameters(array, database)
    row = next(r for r in result.residues if r.residue_type.name == "LYS:conj_NZ")
    old = next(r for r in database.chemical.residues if r.name == row.residue_type.name)
    old_q = _charges(database, old)
    assert row.partial_charges["CE"] - old_q["CE"] == pytest.approx(-0.2029, abs=1e-12)
    assert row.partial_charges["NZ"] - old_q["NZ"] == pytest.approx(-0.7771, abs=1e-12)
    assert row.partial_charges["HZ1"] - old_q["HZ1"] == pytest.approx(-0.08, abs=1e-12)
    assert sum(row.partial_charges.values()) - sum(old_q.values()) == pytest.approx(
        -1.06, abs=1e-12
    )
    atoms = {a.name: a for a in row.residue_type.atoms}
    assert atoms["NZ"].atom_type == "Nad"
    assert (atoms["CE"].atom_type, atoms["CE"].genbonded_type) == ("CH2", "CS2")
    assert (atoms["HZ1"].atom_type, atoms["HZ1"].genbonded_type) == ("Hpol", "HN")
    old_atoms = {a.name: a for a in old.atoms}
    for name in ("N", "CA", "C", "O", "CB", "CG"):
        assert atoms[name] == old_atoms[name]
        assert row.partial_charges[name] == old_q[name]
    for field in ("torsion_parameters", "hxltorsion_parameters"):
        assert all(
            "NZ" not in (p.atm2, p.atm3) for p in getattr(row.cartbonded_params, field)
        )
    angle = next(
        p
        for p in row.cartbonded_params.angle_parameters
        if p.atm2 == "NZ" and {p.atm1, p.atm3} == {"CE", "HZ1"}
    )
    assert 118 < np.degrees(angle.x0) < 123
    icoors = {ic.name: ic for ic in row.residue_type.icoors}
    assert icoors["HZ1"].great_grand_parent == "conj_NZ"
    assert icoors["conj_NZ"].great_grand_parent != "HZ1"
    old_icoors = {ic.name: ic for ic in old.icoors}
    for name in ("N", "CA", "C", "O", "CB", "CG"):
        assert icoors[name] == old_icoors[name]
    from tmol.chemical import ResidueTypeSet

    refined = ResidueTypeSet._refine(row.residue_type)
    assert np.isfinite(refined.ideal_coords).all()


def test_attached_asparagine_is_not_flipped_independently(conjugate_input):
    fixture, array, database = conjugate_input
    if fixture != "nglycan":
        pytest.skip("ASN glycan fixture exercises the terminal amide flip")
    from tmol.chemical import ResidueTypeSet
    from tmol.pack.rotamer import OptHSampler

    sampler = OptHSampler()
    base = next(r for r in database.chemical.residues if r.name == "ASN")
    free = ResidueTypeSet._refine(base)
    sampler._annotate_residue_type(free)
    assert free.opth_sampler_cache.nhq_chi_col == 1
    assert sampler.defines_rotamers_for_rt(free)
    for rt in database.chemical.residues:
        if rt.base_name == "ASN" and "conj_ND2" in rt.name:
            refined = ResidueTypeSet._refine(rt)
            sampler._annotate_residue_type(refined)
            assert refined.opth_sampler_cache.nhq_chi_col == -1
            assert refined.opth_sampler_cache.nhq_chi_atom == -1
            assert sampler.defines_rotamers_for_rt(refined) == any(
                cs.is_proton for cs in refined.chi_samples
            )


@pytest.mark.parametrize("opt_h", [False, True])
def test_corrected_attachment_pose_charge_geometry_and_gradient(
    conjugate_input, torch_device, opt_h
):
    fixture, array, database = conjugate_input
    result = generate_conjugate_parameters(array, database)
    corrected = install(database, result)
    pose = pose_stack_from_biotite(
        array, torch_device, param_db=corrected, no_optH=not opt_h
    )
    assert torch.isfinite(pose.coords).all()
    old_by_name = {r.name: r for r in database.chemical.residues}
    new_by_name = {r.residue_type.name: r for r in result.residues}
    delta = 0.0
    for ti in pose.block_type_ind[0].tolist():
        bt = pose.packed_block_types.active_block_types[ti]
        if bt.name in new_by_name:
            delta += sum(new_by_name[bt.name].partial_charges.values()) - sum(
                _charges(database, old_by_name[bt.name]).values()
            )
    assert delta == pytest.approx(-1.0 if fixture == "biotin" else 0.0, abs=1e-8)
    if fixture == "biotin":
        bi = next(
            bi
            for bi, ti in enumerate(pose.block_type_ind[0].tolist())
            if pose.packed_block_types.active_block_types[ti].base_name == "LYS"
            and "conj_NZ" in pose.packed_block_types.active_block_types[ti].name
        )
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, bi])]
        ci = next(i for i, c in enumerate(bt.connections) if c.atom == "NZ")
        other = int(pose.inter_residue_connections[0, bi, ci, 0])
        ob = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, other])
        ]
        offset = int(pose.block_coord_offset[0, bi])
        n, ce, h = [
            pose.coords[0, offset + bt.atom_to_idx[name]]
            for name in ("NZ", "CE", "HZ1")
        ]
        c = pose.coords[
            0, int(pose.block_coord_offset[0, other]) + ob.atom_to_idx["C11"]
        ]
        normal = torch.linalg.cross(ce - n, c - n)
        distance = ((h - n) * normal).sum().abs() / normal.norm()
        assert distance.item() < 1e-4
    from tmol.score import beta2016_score_function

    scorer = beta2016_score_function(
        torch_device, param_db=corrected
    ).render_whole_pose_scoring_module(pose)
    coords = pose.coords.clone().requires_grad_(True)
    values = scorer(coords)
    gradient = torch.autograd.grad(values.sum(), coords)[0]
    assert torch.isfinite(values).all() and torch.isfinite(gradient).all()
