"""Copy plans must follow current sampler instances and chemical ownership."""

import attr
import pytest
import torch

from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import FixedAAChiSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import DunbrackChiSampler
from tmol.score.dunbrack import DunbrackParamResolver
from tmol.tests.pack.rotamer.dunbrack.test_libraryless_polymers import (
    own_chi_pose,
    target_types,
)


def samplers(database, device):
    resolver = DunbrackParamResolver.from_database(database.scoring.dun, device)
    mapping = resolver.all_table_indices.copy()
    mapping.loc["XIL"] = mapping.loc["ILE"]
    return (
        DunbrackChiSampler(attr.evolve(resolver, all_table_indices=mapping)),
        DunbrackChiSampler(resolver),
    )


def build(pose, samplers, chemical, masks=None):
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    for index, sampler in enumerate(samplers):
        if masks is None:
            task.add_conformer_sampler(sampler)
        else:
            task.add_conformer_sampler_by_block_mask(sampler, masks[index])
    task.add_conformer_sampler(FixedAAChiSampler())
    _, rotamers = build_rotamers(pose, SetPackerTask.from_packer_task(task), chemical)
    results = []
    for p, block in torch.nonzero(
        torch.isin(pose.block_type_ind, target_types(pose))
    ).tolist():
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[p, block])
        ]
        first = int(rotamers.rot_offset_for_block[p, block])
        n = int(rotamers.n_rots_for_block[p, block])
        starts = rotamers.coord_offset_for_rot[first : first + n].long()
        atoms = torch.arange(bt.n_atoms, device=pose.device)
        results.append(rotamers.coords[starts[:, None] + atoms].clone())
    return results


@pytest.mark.parametrize("reverse", [False, True])
def test_sampler_switch_preserves_fresh_copy_geometry(
    reverse, ubq_pdb, default_database, torch_device
):
    pair = samplers(default_database, torch_device)[:: -1 if reverse else 1]
    expected = []
    for sampler in pair:
        fresh, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True)
        expected.append(build(fresh, [sampler], default_database.chemical)[0])
    shared, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True)
    for index in (0, 1, 0, 1):
        actual = build(shared, [pair[index]], default_database.chemical)[0]
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=2e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_sampler_class_can_have_different_ownership_in_one_task(
    reverse, ubq_pdb, default_database, torch_device
):
    pair = samplers(default_database, torch_device)[:: -1 if reverse else 1]
    expected = []
    for sampler in pair:
        fresh, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True)
        expected.append(build(fresh, [sampler], default_database.chemical)[0])
    shared, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True, n_poses=2)
    first = torch.zeros_like(shared.block_type_ind, dtype=torch.bool)
    first[0] = True
    actual = build(shared, pair, default_database.chemical, [first, ~first])
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=2e-5)


def test_copy_source_is_union_of_nonnested_regions(torch_device):
    from types import SimpleNamespace
    import numpy
    from tmol.pack.rotamer._mainchain_fingerprint import (
        AtomFingerprint,
        MCFingerprint,
        find_unique_fingerprints,
    )

    tokens = [AtomFingerprint(0, i, 0, 6) for i in range(3)]

    def fingerprint(atoms):
        selected = tuple(tokens[i] for i in atoms)
        return MCFingerprint(
            numpy.array(atoms, dtype=numpy.int32),
            selected,
            selected,
            {tokens[i]: i for i in atoms},
        )

    rt = SimpleNamespace(
        mc_fingerprints={"left": fingerprint([0, 1]), "right": fingerprint([0, 2])},
        _mc_sampler_labels={"left": ("left", 0), "right": ("right", 1)},
    )
    pbt = SimpleNamespace(active_block_types=[rt], n_types=1, device=torch_device)
    packed = find_unique_fingerprints(pbt)
    assert packed.source_atom_mapping.tolist() == [[0, 1, 2]]
    assert packed.atom_mapping[packed.sampler_mapping["left"], 0, 0].tolist() == [
        0,
        1,
        -1,
    ]
    assert packed.atom_mapping[packed.sampler_mapping["right"], 0, 0].tolist() == [
        0,
        -1,
        2,
    ]
    assert find_unique_fingerprints(pbt) is packed


def test_new_equivalent_instances_reuse_tensors_without_retaining_old_samplers(
    ubq_pdb, default_database, torch_device
):
    import gc
    import weakref
    from tmol.pack.rotamer._build_rotamers import annotate_everything

    pose, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True)
    pbt = pose.packed_block_types
    resolver = samplers(default_database, torch_device)[1].dun_param_resolver
    references = []
    pointers = None
    for _ in range(12):
        current = (DunbrackChiSampler(resolver), FixedAAChiSampler())
        references.extend(weakref.ref(s) for s in current)
        annotate_everything(default_database.chemical, current, pbt)
        packed = pbt.mc_fingerprints
        observed = tuple(
            getattr(packed, name).data_ptr()
            for name in ("atom_mapping", "source_atom_mapping", "source_fingerprint")
        )
        if pointers is not None:
            assert observed == pointers
        pointers = observed
        for rt in pbt.active_block_types:
            assert len(rt.mc_fingerprints) <= len(current)
        assert all(id(s) in packed.sampler_mapping for s in current)
        annotate_everything(default_database.chemical, current, pbt)
        assert pbt.mc_fingerprints is packed
    del current
    gc.collect()
    assert all(ref() is None for ref in references)


def test_fingerprint_chemical_identity_and_source_expiry(
    ubq_pdb, default_database, torch_device
):
    import gc
    import weakref
    import numpy
    from tmol.pack.rotamer import construct_single_residue_kinforest
    from tmol.pack.rotamer._mainchain_fingerprint import (
        annotate_residue_type_with_sampler_fingerprints,
        create_mainchain_fingerprint,
    )

    pose, _ = own_chi_pose(ubq_pdb, torch_device, gapped=True)
    rt = next(
        bt for bt in pose.packed_block_types.active_block_types if bt.base_name == "XIL"
    )
    construct_single_residue_kinforest(rt)
    sampler = samplers(default_database, torch_device)[1]
    chemical = default_database.chemical
    alpha_type = rt.atoms[rt.atom_to_idx["CA"]].atom_type
    changed = attr.evolve(
        chemical,
        atom_types=tuple(
            attr.evolve(at, element="N") if at.name == alpha_type else at
            for at in chemical.atom_types
        ),
    )
    observed = []
    for database in (chemical, changed, chemical):
        annotate_residue_type_with_sampler_fingerprints(rt, (sampler,), database)
        cached = rt.mc_fingerprints[sampler.sampler_name()]
        atoms, fingerprints, _ = create_mainchain_fingerprint(
            rt, sampler.first_sc_atoms_for_rt(rt), database
        )
        numpy.testing.assert_array_equal(cached.mc_ats, atoms)
        assert cached.fingerprint == tuple(sorted(fingerprints))
        observed.append(cached.fingerprint)
    assert observed[0] != observed[1]
    assert observed[0] == observed[2]
    private = attr.evolve(chemical)
    reference = weakref.ref(private)
    annotate_residue_type_with_sampler_fingerprints(rt, (sampler,), private)
    del private
    gc.collect()
    assert reference() is None


def test_sampler_without_any_buildable_types_needs_no_copy_plan(
    ubq_pdb, default_database, torch_device
):
    from tmol.io import pose_stack_from_pdb
    from tmol.tests.pack.rotamer.dunbrack.test_sampler_parameter_identity import (
        fresh_pose,
    )

    pose = fresh_pose(
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    )
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampler = DunbrackChiSampler(
        attr.evolve(
            resolver, all_table_indices=resolver.all_table_indices.iloc[:0].copy()
        )
    )
    assert not any(
        sampler.defines_rotamers_for_rt(bt)
        for bt in pose.packed_block_types.active_block_types
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    assert bool(torch.isfinite(rotamers.coords).all())
    assert bool((rotamers.n_rots_for_block == 1).all())
