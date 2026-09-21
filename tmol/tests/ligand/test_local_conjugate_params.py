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

prepared_conjugate_input = test_conjugate_model.conjugate_input


@pytest.fixture(scope="module")
def conjugate_input(prepared_conjugate_input):
    # Compare both parameter sources against the same uncorrected baseline.
    fixture, array, _ = prepared_conjugate_input
    database, _ = test_conjugate_model.prepare_uncorrected_conjugate(
        array, seed=20250828
    )
    return fixture, array, database


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


def test_install_checks_baseline_and_does_not_apply_twice(conjugate_input, tmp_path):
    from tmol.ligand import load_params_file, prepare_ligands
    from tmol.ligand._registry import inject_ligand_preparations

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

    supplied = attr.evolve(
        result.connections[0],
        length_parameters=tuple(
            attr.evolve(row, K=333.0) for row in result.connections[0].length_parameters
        ),
    )
    protected = {supplied.block_type1, supplied.block_type2}
    reference = inject_residue_params(database, [], connection_params=(supplied,))
    additions = generate_conjugate_parameters(array, reference, existing=(supplied,))
    assert not protected.intersection(
        row.residue_type.name for row in additions.residues
    )
    updated = install(reference, additions)
    path = tmp_path / "supplied-reference.tmol"
    prepared, _ = prepare_ligands(array, param_db=reference, params_output=str(path))
    assert prepared.chemical == updated.chemical
    assert prepared.scoring.cartbonded == updated.scoring.cartbonded
    assert prepared.scoring.elec == updated.scoring.elec
    if additions.connections:
        restored = inject_ligand_preparations(reference, load_params_file(path))
        assert restored.chemical == prepared.chemical
        assert restored.scoring.cartbonded == prepared.scoring.cartbonded
        assert restored.scoring.elec == prepared.scoring.elec
    assert supplied in updated.scoring.cartbonded.connection_params
    old = {rt.name: rt for rt in reference.chemical.residues}
    for rt in updated.chemical.residues:
        if rt.name in protected:
            assert rt == old[rt.name]
            assert _charges(updated, rt) == _charges(reference, rt)
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
    write_params_file(preps, path)
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
    second.coord[:] = original.coord[:, [1, 2, 0]] * 2 + [50, -30, 17]
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
    result = generate_conjugate_parameters(
        array, database, parameter_source="mmff94-harmonic"
    )
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
@pytest.mark.parametrize("parameter_source", ["generator-ideals", "mmff94-harmonic"])
def test_corrected_attachment_pose_charge_geometry_and_gradient(
    conjugate_input, torch_device, opt_h, parameter_source
):
    fixture, array, database = conjugate_input
    if parameter_source == "generator-ideals":
        result = generate_conjugate_parameters(array, database)
    else:
        result = generate_conjugate_parameters(
            array, database, parameter_source=parameter_source
        )
    if parameter_source == "generator-ideals":
        assert result.charge_model == "conserved-patched-residue-v1"
        old_types = {r.name: r for r in database.chemical.residues}
        for row in result.residues:
            rt = old_types[row.residue_type.name]
            old_atoms = {a.name: a.atom_type for a in rt.atoms}
            new_atoms = {a.name: a.atom_type for a in row.residue_type.atoms}
            for conn in rt.connections:
                if (
                    conn.name.startswith("conj_")
                    and old_atoms[conn.atom] in database.scoring.genbonded.rosetta_typed
                ):
                    assert new_atoms[conn.atom] == old_atoms[conn.atom]
            assert row.partial_charges == _charges(database, rt)
            records = database.scoring.cartbonded.residue_params
            baseline = records.get(rt.name, records.get(rt.base_name))
            for field, stiffness in (
                ("length_parameters", 300),
                ("angle_parameters", 80),
            ):
                previous = set(getattr(baseline, field))
                assert all(
                    p.K == stiffness
                    for p in getattr(row.cartbonded_params, field)
                    if p not in previous
                )
        assert all(p.K == 300 for r in result.connections for p in r.length_parameters)
        assert all(p.K == 80 for r in result.connections for p in r.angle_parameters)
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
    expected_delta = (
        -1.0 if fixture == "biotin" and parameter_source == "mmff94-harmonic" else 0.0
    )
    assert delta == pytest.approx(expected_delta, abs=1e-8)
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
    if parameter_source == "generator-ideals":
        from tmol.io import build_context_from_biotite
        from tmol.tests.io.test_atomworks_corpus_regressions import _score_and_minimize

        context = build_context_from_biotite(array, torch_device, param_db=corrected)
        _score_and_minimize(pose, context)


def test_terminal_and_internal_n_alkylation_workflow(torch_device, tmp_path):
    """Preserve terminal-amine and internal-amide chemistry through a workflow."""
    import biotite.structure as struc

    from tmol.database import ParameterDatabase
    from tmol.io import atom_array_from_cif, build_context_from_biotite
    from tmol.ligand import load_params_file, prepare_ligands
    from tmol.ligand._conjugate_model import (
        attachment_connection_name,
        capped_conjugate_models,
    )
    from tmol.ligand._connection_params import _protonated_model
    from tmol.ligand._registry import inject_ligand_preparations
    from tmol.score import beta2016_score_function
    from tmol.score.elec._params import ElecParamResolver
    from tmol.tests.data import data_path

    def atom_index(array, residue, atom, residue_id=None):
        selected = (array.res_name == residue) & (array.atom_name == atom)
        if residue_id is not None:
            selected &= array.res_id == residue_id
        (index,) = np.flatnonzero(selected)
        return int(index)

    # A free alpha amine and an unrelated sidechain attachment share one
    # terminal residue. Move only N to keep the synthetic N-C measurement local;
    # preparation derives chemistry from bonds rather than these coordinates.
    terminal = atom_array_from_cif(
        data_path("covalent_fixtures", "oglycan_sia_1g1s.cif")
    )
    standard = struc.filter_amino_acids(terminal)
    target = np.flatnonzero(standard & (terminal.atom_name == "OG1"))
    (target_residue,) = {
        (str(terminal.chain_id[index]), int(terminal.res_id[index])) for index in target
    }
    target_name = str(terminal.res_name[target[0]])
    terminal = terminal[
        (~standard)
        | (
            (terminal.chain_id == target_residue[0])
            & (terminal.res_id == target_residue[1])
        )
    ]
    terminal_n = atom_index(terminal, target_name, "N", target_residue[1])
    terminal_carbon = atom_index(terminal, "NGA", "C2")
    terminal.coord[terminal_n] = terminal.coord[terminal_carbon] + [1.45, 0, 0]
    terminal.bonds.add_bond(terminal_n, terminal_carbon, struc.BondType.SINGLE)
    canonical_target = next(
        residue
        for residue in ParameterDatabase.get_default().chemical.residues
        if residue.name == target_name
    )
    assert (
        attachment_connection_name(
            terminal, terminal_n, terminal_carbon, canonical_target
        )
        == "conj_N"
    )
    phosphorylated = terminal.copy()
    phosphorylated.element[terminal_carbon] = "P"
    assert (
        attachment_connection_name(
            phosphorylated, terminal_n, terminal_carbon, canonical_target
        )
        == "conj_N"
    )

    # Keep both peptide neighbours around an internal backbone N. Rename the
    # target to the chemically equivalent hydroxyl residue after removing its
    # beta methyl, so terminal and internal states have distinct reusable names.
    internal = atom_array_from_cif(
        data_path("covalent_fixtures", "oglycan_sia_1g1s.cif")
    )
    standard = struc.filter_amino_acids(internal)
    target = np.flatnonzero(standard & (internal.atom_name == "OG1"))
    (target_residue,) = {
        (str(internal.chain_id[index]), int(internal.res_id[index])) for index in target
    }
    internal = internal[
        (~standard)
        | (
            (internal.chain_id == target_residue[0])
            & np.isin(
                internal.res_id,
                (
                    target_residue[1] - 1,
                    target_residue[1],
                    target_residue[1] + 1,
                ),
            )
        )
    ]
    target_atoms = (internal.chain_id == target_residue[0]) & (
        internal.res_id == target_residue[1]
    )
    internal = internal[~(target_atoms & (internal.atom_name == "CG2"))]
    target_atoms = (internal.chain_id == target_residue[0]) & (
        internal.res_id == target_residue[1]
    )
    internal.res_name[target_atoms] = "SER"
    internal.atom_name[target_atoms & (internal.atom_name == "OG1")] = "OG"
    internal_n = atom_index(internal, "SER", "N", target_residue[1])
    internal_carbon = atom_index(internal, "NGA", "C2")
    internal.coord[internal_carbon] = internal.coord[internal_n] + [1.45, 0, 0]
    internal.bonds.add_bond(internal_n, internal_carbon, struc.BondType.SINGLE)

    def resolved_charges(database, block):
        resolver = ElecParamResolver.from_database(
            database.scoring.elec, torch.device("cpu")
        )
        values = resolver.get_partial_charges_for_block(block)
        return dict(zip((atom.name for atom in block.atoms), map(float, values)))

    cases = (
        (
            "terminal",
            terminal,
            target_name,
            terminal_n,
            {"nterm", "conj_N", "conj_OG1"},
            {"conj_N", "conj_OG1"},
            {"H1", "H2"},
            "Nlys",
            (1, 2),
        ),
        (
            "internal",
            internal,
            "SER",
            internal_n,
            {"conj_N", "conj_OG"},
            {"down", "up", "conj_N", "conj_OG"},
            set(),
            "Nbb",
            (0, 0),
        ),
    )
    for (
        label,
        array,
        residue_name,
        nitrogen_index,
        expected_variants,
        expected_connections,
        expected_hydrogens,
        expected_type,
        expected_formal,
    ) in reversed(cases):
        params_path = tmp_path / f"{label}.tmol"
        prepared, _ = prepare_ligands(
            array,
            param_db=ParameterDatabase.get_default(),
            params_output=str(params_path),
            seed=20260914,
        )
        context = build_context_from_biotite(array, torch_device, param_db=prepared)
        pose = pose_stack_from_biotite(
            array, torch_device, context=context, no_optH=True
        )
        used = [
            pose.packed_block_types.active_block_types[index]
            for index in pose.block_type_ind[0].tolist()
        ]
        matches = [
            (index, block)
            for index, block in enumerate(used)
            if block.base_name == residue_name
            and expected_variants <= set(block.name.split(":"))
        ]
        assert len(matches) == 1, (label, [block.name for block in used])
        block_index, block = matches[0]
        assert expected_connections <= {
            connection.name for connection in block.connections
        }
        elements = {
            atom_type.name: atom_type.element
            for atom_type in prepared.chemical.atom_types
        }
        hydrogens = {
            other
            for bond in block.bonds
            for center, other in (bond[:2], bond[1::-1])
            if center == "N"
            and elements[next(a.atom_type for a in block.atoms if a.name == other)]
            == "H"
        }
        assert hydrogens == expected_hydrogens
        assert next(a.atom_type for a in block.atoms if a.name == "N") == expected_type

        connection = next(
            i
            for i, candidate in enumerate(block.connections)
            if candidate.name == "conj_N"
        )
        partner, partner_connection = pose.inter_residue_connections[
            0, block_index, connection
        ].tolist()
        assert partner >= 0
        assert pose.inter_residue_connections[
            0, partner, partner_connection
        ].tolist() == [block_index, connection]

        def formal_state(database):
            for model in capped_conjugate_models(array, database.chemical):
                local_by_source = {
                    int(source): local
                    for local, source in enumerate(model.source_atom_indices)
                    if source >= 0
                }
                if nitrogen_index not in local_by_source:
                    continue
                molecule, mapping = _protonated_model(model, 7.4)
                atom = molecule.GetAtomWithIdx(mapping[local_by_source[nitrogen_index]])
                return (
                    atom.GetFormalCharge(),
                    sum(
                        neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
                    ),
                )
            raise AssertionError("N attachment was absent from capped models")

        assert formal_state(prepared) == expected_formal
        charges = resolved_charges(prepared, block)
        assert sum(charges.values()) == pytest.approx(0.0, abs=1e-4)

        reloaded = inject_ligand_preparations(
            ParameterDatabase.get_default(), load_params_file(params_path)
        )
        restored = next(
            residue
            for residue in reloaded.chemical.residues
            if residue.name == block.name
        )
        assert formal_state(reloaded) == expected_formal
        assert {atom.name: atom.atom_type for atom in restored.atoms} == {
            atom.name: atom.atom_type for atom in block.atoms
        }
        assert {
            (connection.name, connection.atom, connection.type)
            for connection in restored.connections
        } == {
            (connection.name, connection.atom, connection.type)
            for connection in block.connections
        }
        assert resolved_charges(reloaded, restored) == charges

        scorer = beta2016_score_function(
            torch_device, param_db=prepared
        ).render_whole_pose_scoring_module(pose)
        coords = pose.coords.clone().requires_grad_(True)
        score = scorer(coords)
        gradient = torch.autograd.grad(score.sum(), coords)[0]
        assert torch.isfinite(score).all()
        assert torch.isfinite(gradient).all()
