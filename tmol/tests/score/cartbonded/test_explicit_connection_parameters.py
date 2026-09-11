"""Independent harmonic reference for complete, connection-owned parameters."""

import attr
import cattr
import pytest
import torch
import yaml

from tmol.database.scoring import (
    AngleGroup,
    CartBondedDatabase,
    ConnectionCartRes,
    LengthGroup,
)
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.score.cartbonded._connection_parameters import compile_connection_parameters
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


def peptide_record(pose):
    bts = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]
    assert len(bts) == 2 and bts[1].base_name != "PRO"
    return ConnectionCartRes(
        block_type1=bts[0].name,
        connection1="up",
        block_type2=bts[1].name,
        connection2="down",
        length_parameters=(LengthGroup("C", "+N", x0=1.43, K=317.0),),
        angle_parameters=(
            AngleGroup("CA", "C", "+N", x0=1.8, K=83.0),
            AngleGroup("O", "C", "+N", x0=1.9, K=91.0),
            AngleGroup("C", "+N", "+CA", x0=2.0, K=107.0),
            AngleGroup("C", "+N", "+H", x0=2.1, K=113.0),
        ),
        provenance="Synthetic harmonic constants for numerical validation; not a fitted force field",
    )


def reverse_record(record):
    def reverse_row(row):
        names = {
            field: (
                getattr(row, field)[1:]
                if getattr(row, field).startswith("+")
                else "+" + getattr(row, field)
            )
            for field in ("atm1", "atm2", "atm3")
            if hasattr(row, field)
        }
        return attr.evolve(row, **names)

    return attr.evolve(
        record,
        block_type1=record.block_type2,
        connection1=record.connection2,
        block_type2=record.block_type1,
        connection2=record.connection1,
        length_parameters=tuple(reverse_row(row) for row in record.length_parameters),
        angle_parameters=tuple(reverse_row(row) for row in record.angle_parameters),
    )


def scoring_term(pose, database):
    term = CartBondedEnergyTerm(database, pose.device)
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    return term


def harmonic_reference(pose, coords, record):
    bts = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]

    def xyz(name):
        block = int(name.startswith("+"))
        name = name[1:] if block else name
        return coords[
            0, int(pose.block_coord_offset[0, block]) + bts[block].atom_to_idx[name]
        ]

    def fp32(value):
        return float(torch.tensor(value, dtype=torch.float32))

    terms = []
    for rows in (record.length_parameters, record.angle_parameters):
        total = coords.new_zeros(())
        for row in rows:
            if hasattr(row, "atm3"):
                a, b, c = [
                    xyz(getattr(row, field)) for field in ("atm1", "atm2", "atm3")
                ]
                first, second = a - b, c - b
                value = torch.acos(first.dot(second) / (first.norm() * second.norm()))
            else:
                value = (xyz(row.atm1) - xyz(row.atm2)).norm()
            total += 0.5 * fp32(row.K) * (value - fp32(row.x0)).square()
        terms.append(total)
    return torch.stack(terms)


@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("retain_legacy", [False, True])
def test_connection_energy_gradient_and_single_ownership(
    ubq_pdb, default_database, torch_device, block_pairs, reverse, retain_legacy
):
    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    # Including equivalent records in both directions must not score twice.
    records = (reverse_record(record), record) if reverse else (record,)
    cart = CartBondedDatabase.from_cartres_dict(
        default_database.scoring.cartbonded.residue_params if retain_legacy else {},
        records,
    )
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, cartbonded=cart)
    )
    term = scoring_term(pose, database)
    render = (
        term.render_block_pair_scoring_module
        if block_pairs
        else term.render_whole_pose_scoring_module
    )
    module = render(pose)
    coords = pose.coords.double().clone()
    coords += 0.2 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    values = module(coords)
    weights = coords.new_tensor([0.7, 1.9])
    if block_pairs:
        # Unequal term weights exercise the native block-pair backward path.
        actual = values[:2, 0, 0, 1]
    else:
        actual = values[:2, 0]
        if retain_legacy:
            baseline = scoring_term(
                pose, default_database
            ).render_block_pair_scoring_module(pose)(coords)
            actual = actual - baseline[:2, 0].diagonal(dim1=-2, dim2=-1).sum(-1)
    expected = harmonic_reference(pose, coords, record)
    assert float(expected.detach().sum()) > 0.1
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    actual_gradient = torch.autograd.grad(
        (actual * weights).sum(), coords, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), coords)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "extraneous",
        "conflict",
        "nan",
        "wrong_type",
        "unknown_connection",
    ],
)
def test_invalid_connection_records_raise_before_native_scoring(
    ubq_pdb, torch_device, problem
):
    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    rows = (record,)
    if problem == "missing":
        rows = (attr.evolve(record, angle_parameters=record.angle_parameters[:-1]),)
    elif problem == "duplicate":
        rows = (attr.evolve(record, length_parameters=record.length_parameters * 2),)
    elif problem == "extraneous":
        rows = (
            attr.evolve(
                record,
                angle_parameters=record.angle_parameters
                + (AngleGroup("CA", "C", "O", 2.0, 10.0),),
            ),
        )
    elif problem == "conflict":
        rows += (
            attr.evolve(
                record,
                length_parameters=(attr.evolve(record.length_parameters[0], K=7.0),),
            ),
        )
    elif problem == "nan":
        rows = (
            attr.evolve(
                record,
                length_parameters=(
                    attr.evolve(record.length_parameters[0], x0=float("nan")),
                ),
            ),
        )
    elif problem == "wrong_type":
        rows = (
            attr.evolve(
                record,
                length_parameters=(attr.evolve(record.length_parameters[0], type=3),),
            ),
        )
    elif problem == "unknown_connection":
        rows = (attr.evolve(record, connection1="absent"),)
    with pytest.raises(ValueError):
        compile_connection_parameters(rows, pose.packed_block_types, 0)


def test_connection_records_survive_database_serialization(
    ubq_pdb, default_database, torch_device, tmp_path
):
    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    original = CartBondedDatabase.from_cartres_dict({}, (record,))
    path = tmp_path / "cartbonded.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "residue_params": {},
                "connection_params": cattr.unstructure(original.connection_params),
            }
        )
    )
    restored = CartBondedDatabase.from_file(path)
    assert restored.connection_params == original.connection_params
    assert restored.hash == original.hash
    path.write_text(yaml.safe_dump(cattr.unstructure(restored)))
    assert CartBondedDatabase.from_file(path) == restored
    extension = tmp_path / "connections.yaml"
    extension.write_text(
        yaml.safe_dump(
            {"connection_params": cattr.unstructure(original.connection_params)}
        )
    )
    path.write_text(yaml.safe_dump({"residue_params": {}}))
    assert CartBondedDatabase.from_file(path, generated=(extension,)) == restored
    assert restored.hash != CartBondedDatabase.from_cartres_dict({}).hash
    database = attr.evolve(
        default_database,
        scoring=attr.evolve(default_database.scoring, cartbonded=restored),
    )
    module = scoring_term(pose, database).render_whole_pose_scoring_module(pose)
    coords = pose.coords.double()
    torch.testing.assert_close(
        module(coords)[:2, 0],
        harmonic_reference(pose, coords, record),
        rtol=1e-7,
        atol=1e-7,
    )


def test_rotamer_connection_pairs_and_gradients(
    ubq_pdb, default_database, torch_device, dun_sampler
):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.pose import PoseStackBuilder

    single = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(single)
    pose = PoseStackBuilder.from_poses([single, single], torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.set_chi_sample_budget(128, 64)
    task.add_conformer_sampler(dun_sampler)
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    cart = CartBondedDatabase.from_cartres_dict({}, (record,))
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, cartbonded=cart)
    )
    scorer = scoring_term(pose, database).render_rotamer_scoring_module(pose, rotamers)
    coords = rotamers.coords.double().clone()
    coords += 0.13 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    scores, indices = scorer(coords)
    rot1, rot2 = indices[1].long(), indices[2].long()
    cross = rotamers.block_ind_for_rot[rot1] != rotamers.block_ind_for_rot[rot2]
    assert int(cross.sum()) == int(
        (rotamers.n_rots_for_block[:, 0] * rotamers.n_rots_for_block[:, 1]).sum()
    )
    assert torch.all(rotamers.block_ind_for_rot[rot1[cross]] == 0)
    assert torch.all(rotamers.block_ind_for_rot[rot2[cross]] == 1)
    bts = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]

    def xyz(name):
        side = int(name.startswith("+"))
        atom = name[1:] if side else name
        rots = rot2[cross] if side else rot1[cross]
        atom_indices = (
            rotamers.coord_offset_for_rot[rots].long() + bts[side].atom_to_idx[atom]
        )
        return coords[atom_indices]

    expected = []
    for rows in (record.length_parameters, record.angle_parameters):
        energy = coords.new_zeros(int(cross.sum()))
        for row in rows:
            if hasattr(row, "atm3"):
                first = xyz(row.atm1) - xyz(row.atm2)
                second = xyz(row.atm3) - xyz(row.atm2)
                value = torch.acos(
                    (first * second).sum(-1)
                    / (first.norm(dim=-1) * second.norm(dim=-1))
                )
            else:
                value = (xyz(row.atm1) - xyz(row.atm2)).norm(dim=-1)
            target, force = [
                float(torch.tensor(v, dtype=torch.float32)) for v in (row.x0, row.K)
            ]
            energy += 0.5 * force * (value - target).square()
        expected.append(energy)
    expected = torch.stack(expected)
    actual = scores[:2, cross]
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    weights = (
        0.2
        + torch.arange(actual.numel(), device=torch_device).reshape_as(actual)
        / actual.numel()
    )
    actual_gradient = torch.autograd.grad(
        (weights * actual).sum(), coords, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad((weights * expected).sum(), coords)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-7, atol=1e-7)


def test_packing_uses_explicit_connection_parameters(
    ubq_pdb, default_database, torch_device, dun_sampler
):
    from tmol.pack import PackerPalette, PackerTask
    from tmol.tests.pack.test_conjugated_group_packing import _pack_and_check_score

    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    cart = CartBondedDatabase.from_cartres_dict(
        default_database.scoring.cartbonded.residue_params, (record,)
    )
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, cartbonded=cart)
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.set_chi_sample_budget(128, 64)
    task.add_conformer_sampler(dun_sampler)
    _pack_and_check_score(pose, database, torch_device, task=task)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_identical_connection_types_require_exchange_symmetry(
    fresh_default_packed_block_types, asymmetric
):
    pbt = fresh_default_packed_block_types
    bt = next(bt for bt in pbt.active_block_types if bt.name == "CYD")
    record = ConnectionCartRes(
        bt.name,
        "dslf",
        bt.name,
        "dslf",
        length_parameters=(LengthGroup("SG", "+SG", 2.03, 201.0),),
        angle_parameters=(
            AngleGroup("CB", "SG", "+SG", 1.9, 81.0),
            AngleGroup("+CB", "+SG", "SG", 1.9, 82.0 if asymmetric else 81.0),
        ),
    )
    if asymmetric:
        with pytest.raises(ValueError, match="exchange-symmetric"):
            compile_connection_parameters((record,), pbt, 0)
    else:
        keys, spans, paths, values = compile_connection_parameters(
            (record, reverse_record(record)), pbt, 0
        )
        assert keys.shape == (2, 5)
        assert spans.tolist() == [[0, 3]]
        assert len(paths) == len(values) == 3


def test_connection_records_are_scoped_in_a_mixed_pose_batch(
    ubq_pdb, default_database, torch_device
):
    from tmol.pose import PoseStackBuilder

    matched = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    unmatched = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(1, 3)])
    record = peptide_record(matched)
    pose = PoseStackBuilder.from_poses([matched, unmatched, matched], torch_device)
    cart = CartBondedDatabase.from_cartres_dict({}, (record,))
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, cartbonded=cart)
    )
    module = scoring_term(pose, database).render_whole_pose_scoring_module(pose)
    coords = pose.coords.double().clone()
    coords += 0.1 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    actual = module(coords)[:2]
    expected = torch.stack(
        [
            harmonic_reference(matched, coords[0:1], record),
            coords.new_zeros(2),
            harmonic_reference(matched, coords[2:3], record),
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    weights = coords.new_tensor([[0.3, 0.7, 1.2], [1.7, 0.9, 0.4]])
    actual_gradient = torch.autograd.grad(
        (actual * weights).sum(), coords, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad((expected * weights).sum(), coords)[0]
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=1e-7, atol=1e-7
    )


def test_cartesian_minimization_restores_explicit_connection_length(
    ubq_pdb, default_database, torch_device
):
    from tmol.optimization import run_cart_min
    from tmol.score import ScoreFunction, ScoreType

    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    cart = CartBondedDatabase.from_cartres_dict(
        default_database.scoring.cartbonded.residue_params, (record,)
    )
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, cartbonded=cart)
    )
    score = ScoreFunction(database, torch_device)
    score.set_weight(ScoreType.cart_lengths, 1.0)
    score.set_weight(ScoreType.cart_angles, 1.0)
    first, second = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]
    start = int(pose.block_coord_offset[0, 1])
    carbon = first.atom_to_idx["C"]
    nitrogen = start + second.atom_to_idx["N"]
    axis = pose.coords[0, nitrogen] - pose.coords[0, carbon]
    axis /= axis.norm()
    pose.coords[0, start : start + len(second.atoms)] += axis
    assert float((pose.coords[0, nitrogen] - pose.coords[0, carbon]).norm()) > 2.0
    mask = torch.zeros_like(pose.real_atoms)
    mask[0, start : start + len(second.atoms)] = True
    module = score.render_whole_pose_scoring_module(pose)
    before = module(pose.coords).detach()
    result = run_cart_min(
        pose, score, coord_mask=mask, optimizer_kwargs={"max_iter": 200}
    )
    after = module(result.coords).detach()
    assert torch.all(after < before)
    distance = float((result.coords[0, nitrogen] - result.coords[0, carbon]).norm())
    assert abs(distance - record.length_parameters[0].x0) < 0.005
    torch.testing.assert_close(
        result.coords[0, :start], pose.coords[0, :start], rtol=0, atol=0
    )


def test_public_injection_retains_connection_records_when_residue_rows_change(
    ubq_pdb, default_database, torch_device
):
    from tmol.database import inject_residue_params

    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    first = inject_residue_params(
        default_database, residue_types=[], connection_params=(record,)
    )
    before = scoring_term(pose, first).render_block_pair_scoring_module(pose)
    old_ala = first.scoring.cartbonded.residue_params["ALA"]
    changed_ala = attr.evolve(
        old_ala,
        length_parameters=tuple(
            attr.evolve(row, K=row.K + 1) for row in old_ala.length_parameters
        ),
    )
    second = inject_residue_params(
        first, residue_types=[], cartbonded_params={"ALA": changed_ala}
    )
    assert second.scoring.cartbonded.connection_params == (record,)
    assert second.scoring.cartbonded.hash != first.scoring.cartbonded.hash
    after = scoring_term(pose, second).render_block_pair_scoring_module(pose)
    coords = pose.coords.double()
    expected = harmonic_reference(pose, coords, record)
    for module in (before, after):
        torch.testing.assert_close(
            module(coords)[:2, 0, 0, 1], expected, rtol=1e-7, atol=1e-7
        )
