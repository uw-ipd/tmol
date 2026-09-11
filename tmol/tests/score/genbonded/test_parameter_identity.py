"""Generic annotations must follow the scoring database on reused poses."""

import attr
import pytest
import torch

from tmol.io import extended_pose_stack_from_sequences
from tmol.score.genbonded import GenBondedEnergyTerm


def _database(default_database, factor):
    gen = default_database.scoring.genbonded
    changed = attr.evolve(
        gen,
        # Synthetic ownership exercises canonical fixture topology with nonzero
        # generic terms. This does not propose changing production ownership.
        rosetta_typed=frozenset(),
        torsions=tuple(
            attr.evolve(
                p,
                **{
                    field: factor * getattr(p, field)
                    for field in ("k1", "k2", "k3", "k4", "offset")
                },
            )
            for p in gen.torsions
        ),
        impropers=tuple(attr.evolve(p, k=factor * p.k) for p in gen.impropers),
    )
    return attr.evolve(
        default_database,
        scoring=attr.evolve(default_database.scoring, genbonded=changed),
    )


def _setup(term, pose):
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)


def test_block_annotation_tracks_ownership_and_elements(default_database):
    from tmol.tests.score.genbonded.test_generic_type_references import _lysine

    bt = _lysine(default_database, {"CE": "CS2"})
    base = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    generic = GenBondedEnergyTerm(_database(default_database, 1.0), torch.device("cpu"))
    base.setup_block_type(bt)
    assert len(bt.genbonded_intra_subgraphs) == 0
    generic.setup_block_type(bt)
    assert len(bt.genbonded_intra_subgraphs) > 0
    base.setup_block_type(bt)
    assert len(bt.genbonded_intra_subgraphs) == 0
    wrong_elements = attr.evolve(
        default_database,
        chemical=attr.evolve(
            default_database.chemical,
            atom_types=tuple(
                attr.evolve(a, element="N") if a.name == "CS2" else a
                for a in default_database.chemical.atom_types
            ),
        ),
    )
    term = GenBondedEnergyTerm(wrong_elements, torch.device("cpu"))
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid genbonded_type"):
            term.setup_block_type(bt)


def test_database_tables_are_shared_bounded_and_do_not_retain_owner(
    default_database, torch_device, monkeypatch
):
    import gc
    import weakref
    import tmol.score.genbonded._genbonded_energy_term as implementation

    cache = implementation.WeakIdentityLRU(2)
    monkeypatch.setattr(implementation, "_INTER_TABLE_CACHE", cache)
    pose = extended_pose_stack_from_sequences(["KK"], device=torch_device)
    other = extended_pose_stack_from_sequences(["KKK"], device=torch_device)
    databases = [_database(default_database, f) for f in (1.0, 2.0, 3.0)]
    refs = [weakref.ref(d.scoring.genbonded) for d in databases]
    modules = []
    for database in databases:
        term = GenBondedEnergyTerm(database, torch_device)
        _setup(term, pose)
        first = pose.packed_block_types._genbonded_parameters.values[-4:]
        _setup(term, other)
        second = other.packed_block_types._genbonded_parameters.values[-4:]
        assert all(a is b for a, b in zip(first, second))
        modules.append(term.render_whole_pose_scoring_module(pose))
    assert len(cache) == 2
    # Annotation and rendered tensors may live longer than the source database.
    del database, databases, term
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert len(cache) == 0
    energies = [m(pose.coords) for m in modules]
    for factor, energy in enumerate(energies, 1):
        torch.testing.assert_close(energy, energies[0] * factor, atol=1e-4, rtol=1e-5)


def test_rotamer_database_scaling_and_module_snapshot(default_database, torch_device):
    from tmol.pack import PackerTask, PackerPalette, SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

    pose = extended_pose_stack_from_sequences(["KK"], device=torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(default_database, torch_device)
    )
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    assert int(rotamers.n_rots_for_block.min()) > 1
    modules = []
    for factor in (1.0, 2.0):
        term = GenBondedEnergyTerm(_database(default_database, factor), torch_device)
        _setup(term, pose)
        modules.append(term.render_rotamer_scoring_module(pose, rotamers))
    coords = rotamers.coords.double().clone().requires_grad_(True)
    first, first_index = modules[0](coords)
    second, second_index = modules[1](coords)
    torch.testing.assert_close(first_index, second_index)
    assert first.abs().sum().item() > 1
    torch.testing.assert_close(second, first * 2, rtol=1e-6, atol=1e-6)
    weights = 0.1 + (torch.arange(first.numel(), device=torch_device) % 7).reshape_as(
        first
    )
    gradients = [
        torch.autograd.grad((energy * weights).sum(), coords, retain_graph=True)[0]
        for energy in (first, second)
    ]
    torch.testing.assert_close(gradients[1], gradients[0] * 2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_database_scaling_on_reused_pose(
    default_database, torch_device, block_pairs, reverse
):
    pose = extended_pose_stack_from_sequences(["KK", "KKK"], device=torch_device)
    factors = (2.0, 1.0) if reverse else (1.0, 2.0)
    terms = [
        GenBondedEnergyTerm(_database(default_database, f), torch_device)
        for f in factors
    ]
    modules = []
    for term in terms:
        _setup(term, pose)
        render = (
            term.render_block_pair_scoring_module
            if block_pairs
            else term.render_whole_pose_scoring_module
        )
        modules.append(render(pose))
    coords = pose.coords.double().clone()
    coords += 0.17 * torch.sin(
        torch.arange(coords.numel(), device=torch_device).reshape_as(coords)
    )
    coords.requires_grad_(True)
    energies = [m(coords) for m in modules]
    assert energies[0].abs().sum().item() > 1
    scale = factors[1] / factors[0]
    torch.testing.assert_close(energies[1], energies[0] * scale, atol=1e-6, rtol=1e-6)
    weights = 0.2 + torch.arange(energies[0].numel(), device=torch_device).reshape_as(
        energies[0]
    )
    gradients = [
        torch.autograd.grad((e * weights).sum(), coords, retain_graph=True)[0]
        for e in energies
    ]
    torch.testing.assert_close(gradients[1], gradients[0] * scale, atol=1e-6, rtol=1e-6)
    # Rendering again after another database's setup must restore this term's
    # parameters; already-rendered modules must retain their own snapshots.
    for term, module, expected in zip(terms, modules, energies):
        render = (
            term.render_block_pair_scoring_module
            if block_pairs
            else term.render_whole_pose_scoring_module
        )
        torch.testing.assert_close(render(pose)(coords), expected)
        torch.testing.assert_close(module(coords), expected)
