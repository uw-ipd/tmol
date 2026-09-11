"""Exact patched residue parameters must not change other forms of the base."""

import attr
import pytest
import torch

from tmol.database import inject_residue_params
from tmol.io import extended_pose_stack_from_sequences
from tmol.score import AtomTypeDependentTerm
from tmol.tests.score.cartbonded.test_explicit_connection_parameters import scoring_term


@pytest.mark.parametrize("kind", ["length", "angle"])
@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize("changed_first", [False, True])
def test_exact_variant_parameters_are_isolated(
    default_database, torch_device, kind, block_pairs, changed_first
):
    pose = extended_pose_stack_from_sequences(["AAA", "AAAA"], device=torch_device)
    pbt = pose.packed_block_types
    names = {bt.name for bt in pbt.active_block_types if bt.base_name == "ALA"}
    target = next(n for n in names if "nterm" in n and "cterm" not in n)
    original = default_database.scoring.cartbonded.residue_params["ALA"]
    field = kind + "_parameters"
    path = ("CA", "CB") if kind == "length" else ("N", "CA", "CB")
    old = next(
        row
        for row in getattr(original, field)
        if tuple(getattr(row, f"atm{i + 1}") for i in range(len(path))) == path
    )
    new = attr.evolve(old, x0=2.3, K=199.0)
    replacement = attr.evolve(
        original,
        **{
            field: tuple(new if row is old else row for row in getattr(original, field))
        },
    )
    changed = inject_residue_params(
        default_database, [], cartbonded_params={target: replacement}
    )
    databases = {"original": default_database, "changed": changed}
    modules = {}
    AtomTypeDependentTerm(default_database, torch_device).setup_packed_block_types(pbt)
    global_ids = pbt.atom_unique_ids.clone()
    for name in (("changed", "original") if changed_first else ("original", "changed")):
        term = scoring_term(pose, databases[name])
        modules[name] = (
            term.render_block_pair_scoring_module(pose)
            if block_pairs
            else term.render_whole_pose_scoring_module(pose)
        )
    torch.testing.assert_close(pbt.atom_unique_ids, global_ids)
    coords = pose.coords.double().clone()
    coords += 0.1 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    weights = (
        0.3
        + torch.arange(
            pose.n_poses * pose.max_n_blocks**2, device=torch_device
        ).double()
    ).reshape(pose.n_poses, pose.max_n_blocks, pose.max_n_blocks)

    def energy(name):
        values = modules[name](coords)
        return (values * weights).sum() if block_pairs else values.sum()

    delta = energy("changed") - energy("original")
    reference = coords.new_zeros(())
    seen = set()
    for pi, types in enumerate(pose.block_type_ind.tolist()):
        for bi, ti in enumerate(types):
            if ti < 0:
                continue
            bt = pbt.active_block_types[ti]
            if bt.base_name == "ALA":
                seen.add(bt.name)
            if bt.name != target:
                continue
            start = int(pose.block_coord_offset[pi, bi])
            atoms = [coords[pi, start + bt.atom_to_idx[n]] for n in path]
            if kind == "length":
                value = (atoms[0] - atoms[1]).norm()
            else:
                a, b = atoms[0] - atoms[1], atoms[2] - atoms[1]
                value = torch.acos(a.dot(b) / (a.norm() * b.norm()))

            def harmonic(row):
                x0, k = (
                    float(torch.tensor(v, dtype=torch.float32)) for v in (row.x0, row.K)
                )
                return 0.5 * k * (value - x0).square()

            reference += (harmonic(new) - harmonic(old)) * (
                weights[pi, bi, bi] if block_pairs else 1
            )
    assert target in seen and len(seen) == 3, seen
    assert reference.detach().abs() > 1
    torch.testing.assert_close(delta, reference, rtol=1e-7, atol=1e-6)
    actual = torch.autograd.grad(delta, coords, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, coords)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-6)


def test_variant_rotamer_energies_and_gradients(
    default_database, torch_device, dun_sampler
):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers

    pose = extended_pose_stack_from_sequences(["KKK", "KKKK"], device=torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.set_chi_sample_budget(128, 64)
    task.add_conformer_sampler(dun_sampler)
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    pbt = pose.packed_block_types
    target = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "LYS:nterm"
    )
    bt = pbt.active_block_types[target]
    original = default_database.scoring.cartbonded.residue_params["LYS"]
    old = next(
        row
        for row in original.length_parameters
        if (row.atm1, row.atm2) == ("CE", "NZ")
    )
    new = attr.evolve(old, x0=2.3, K=199.0)
    replacement = attr.evolve(
        original,
        length_parameters=tuple(
            new if row is old else row for row in original.length_parameters
        ),
    )
    changed = inject_residue_params(
        default_database, [], cartbonded_params={bt.name: replacement}
    )
    modules = [
        scoring_term(pose, db).render_rotamer_scoring_module(pose, rotamers)
        for db in (default_database, changed)
    ]
    coords = rotamers.coords.double().clone()
    coords += 0.1 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    (before, indices), (after, changed_indices) = [module(coords) for module in modules]
    torch.testing.assert_close(indices, changed_indices)
    r1, r2 = indices[1].long(), indices[2].long()
    selected = (r1 == r2) & (rotamers.block_type_ind_for_rot[r1] == target)
    assert (
        int(selected.sum())
        == int((rotamers.block_type_ind_for_rot == target).sum())
        > 1
    )
    starts = rotamers.coord_offset_for_rot[r1[selected]].long()
    distances = (
        coords[starts + bt.atom_to_idx["CE"]] - coords[starts + bt.atom_to_idx["NZ"]]
    ).norm(dim=-1)

    def harmonic(row):
        x0, k = (float(torch.tensor(v, dtype=torch.float32)) for v in (row.x0, row.K))
        return 0.5 * k * (distances - x0).square()

    expected = torch.zeros_like(after)
    expected[0, selected] = harmonic(new) - harmonic(old)
    torch.testing.assert_close(after - before, expected, rtol=1e-7, atol=1e-7)
    weights = torch.linspace(0.3, 1.3, after.numel(), device=torch_device).reshape_as(
        after
    )
    actual_grad = torch.autograd.grad(
        ((after - before) * weights).sum(), coords, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("conflict_with_source", [False, True])
def test_contradictory_bonded_bundle_rejected(
    tmp_path, default_database, conflict_with_source
):
    from dataclasses import replace
    from tmol.ligand import write_params_file
    from tmol.ligand._registry import inject_ligand_preparations
    from tmol.tests.ligand.test_ligand_entry_paths import _single_prep

    prep = _single_prep()
    changed = attr.evolve(prep.cartbonded_params, length_parameters=())
    assert changed != prep.cartbonded_params
    name = prep.residue_type.name if conflict_with_source else "LYS:conj_NZ"
    preps = [replace(prep, additional_cartbonded_params={name: changed})]
    if not conflict_with_source:
        preps.append(
            replace(prep, additional_cartbonded_params={name: prep.cartbonded_params})
        )
    with pytest.raises(ValueError, match="Conflicting bonded parameters"):
        inject_ligand_preparations(default_database, preps)
    with pytest.raises(ValueError, match="Conflicting bonded parameters"):
        write_params_file(preps, tmp_path / "conflict.tmol", format="tmol")


def test_biotin_variant_params_survive_bundle_reload(tmp_path, torch_device):
    from dataclasses import replace
    from tmol.database import ParameterDatabase
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite
    from tmol.ligand import prepare_ligands, load_params_file, write_params_file
    from tmol.ligand._registry import inject_ligand_preparations
    from tmol.tests.data import data_path

    aa = atom_array_from_cif(data_path("covalent_fixtures", "lys_biotin_1bdo.cif"))
    path = tmp_path / "biotin.tmol"
    prepared, _ = prepare_ligands(aa, seed=20250828, params_output=str(path))
    pose = pose_stack_from_biotite(aa, torch_device, param_db=prepared, no_optH=True)
    pbt = pose.packed_block_types
    bi, bt = next(
        (bi, pbt.active_block_types[ti])
        for bi, ti in enumerate(pose.block_type_ind[0].tolist())
        if pbt.active_block_types[ti].name == "LYS:conj_NZ"
    )
    original = prepared.scoring.cartbonded.residue_params["LYS"]
    old = next(
        row
        for row in original.angle_parameters
        if (row.atm1, row.atm2, row.atm3) == ("CE", "NZ", "HZ1")
    )
    # Synthetic parameters test isolation and persistence, not an amide fit.
    new = attr.evolve(old, x0=2.2, K=173.0)
    replacement = attr.evolve(
        original,
        angle_parameters=tuple(
            new if row is old else row for row in original.angle_parameters
        ),
    )
    preps = load_params_file(path)
    preps[0] = replace(preps[0], additional_cartbonded_params={bt.name: replacement})
    write_params_file(preps, path, format="tmol")
    loaded = load_params_file(path)
    assert loaded[0].additional_cartbonded_params == {bt.name: replacement}
    database = inject_ligand_preparations(ParameterDatabase.get_default(), loaded)
    enriched = inject_ligand_preparations(prepared, loaded)
    assert inject_ligand_preparations(enriched, loaded) is enriched
    coords = pose.coords.double().clone().requires_grad_(True)
    scores = [
        scoring_term(pose, db).render_whole_pose_scoring_module(pose)(coords)
        for db in (prepared, database, enriched)
    ]
    torch.testing.assert_close(scores[1], scores[2])
    start = int(pose.block_coord_offset[0, bi])
    a, b, c = [coords[0, start + bt.atom_to_idx[name]] for name in ("CE", "NZ", "HZ1")]
    u, v = a - b, c - b
    angle = torch.acos(u.dot(v) / (u.norm() * v.norm()))

    def harmonic(row):
        x0, k = (float(torch.tensor(v, dtype=torch.float32)) for v in (row.x0, row.K))
        return 0.5 * k * (angle - x0).square()

    expected = torch.zeros_like(scores[0])
    expected[1, 0] = harmonic(new) - harmonic(old)
    delta = scores[1] - scores[0]
    torch.testing.assert_close(delta, expected, rtol=1e-7, atol=1e-7)
    actual_grad = torch.autograd.grad(delta.sum(), coords, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)
