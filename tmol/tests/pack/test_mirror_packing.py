"""Mirror packing must preserve full conformers and their interaction energies."""

from collections import defaultdict
from itertools import permutations

import numpy
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from tmol.score import beta2016_score_function
from tmol.tests.pack.rotamer.dunbrack.test_mirror_sampling import build_mirror_rotamers


def full_conformer_groups(pose, rotamers):
    pbt = pose.packed_block_types
    elements = {a.name: a.element for a in pbt.chem_db.atom_types}
    metadata = []
    for rt in pbt.active_block_types:
        names = tuple(sorted(rt.atom_to_idx))
        indices = numpy.array([rt.atom_to_idx[n] for n in names])
        atom_types = {a.name: a.atom_type for a in rt.atoms}
        neighbors = defaultdict(set)
        for first, second, *_ in rt.bonds:
            neighbors[first].add(second)
            neighbors[second].add(first)
        equivalent_h = defaultdict(list)
        heavy = []
        for index, name in enumerate(names):
            if elements[atom_types[name]] != "H":
                heavy.append(index)
            else:
                # Only indistinguishable H atoms attached to the same named
                # neighbors and having the same chemical atom type may swap.
                equivalent_h[tuple(sorted(neighbors[name])), atom_types[name]].append(
                    index
                )
        metadata.append((names, indices, heavy, list(equivalent_h.values())))
    coords = rotamers.coords.detach().cpu().numpy()
    groups = defaultdict(list)
    for index, (block, ti, offset) in enumerate(
        zip(
            rotamers.block_ind_for_rot.cpu().tolist(),
            rotamers.block_type_ind_for_rot.cpu().tolist(),
            rotamers.coord_offset_for_rot.cpu().tolist(),
        )
    ):
        names, indices, heavy, hydrogen_groups = metadata[ti]
        groups[block, names].append(
            (index, coords[offset + indices], heavy, hydrogen_groups)
        )
    return groups


def mirror_rotamer_permutation(left, right):
    a_groups, b_groups = full_conformer_groups(*left), full_conformer_groups(*right)
    assert a_groups.keys() == b_groups.keys()
    mapping = numpy.full(len(left[1].block_ind_for_rot), -1, dtype=numpy.int64)
    for key in a_groups:
        a_rows, b_rows = a_groups[key], b_groups[key]
        assert len(a_rows) == len(b_rows), key
        a = numpy.array([row[1] for row in a_rows])
        b = -numpy.array([row[1] for row in b_rows])
        heavy, h_groups = a_rows[0][2:]
        cost = cdist(
            a[:, heavy].reshape(len(a), -1),
            b[:, heavy].reshape(len(b), -1),
            metric="sqeuclidean",
        )
        for group in h_groups:
            # These canonical fixtures have at most three equivalent H atoms.
            assert len(group) <= 3
            alternatives = [
                cdist(
                    a[:, group].reshape(len(a), -1),
                    b[:, perm].reshape(len(b), -1),
                    metric="sqeuclidean",
                )
                for perm in permutations(group)
            ]
            cost += numpy.minimum.reduce(alternatives)
        rows, columns = linear_sum_assignment(cost)
        paired_b = b[columns].copy()
        for group in h_groups:
            permuted = numpy.array(list(permutations(group)))
            errors = numpy.square(
                a[rows][:, None, group, :] - b[columns][:, permuted, :]
            ).sum(axis=(2, 3))
            choice = errors.argmin(axis=1)
            paired_b[:, group] = b[columns][:, permuted, :][
                numpy.arange(len(rows)), choice
            ]
        numpy.testing.assert_allclose(
            a[rows],
            paired_b,
            atol=1e-4,
            rtol=0,
            err_msg=f"Block {key[0]} full-atom mirror mismatch",
        )
        for first, second in zip(rows, columns):
            mapping[a_rows[first][0]] = b_rows[second][0]
    assert numpy.array_equal(numpy.sort(mapping), numpy.arange(len(mapping)))
    return mapping


def test_mirrored_conformers_have_the_same_per_term_packing_energies(
    default_database, torch_device
):
    database = default_database.with_symmetric_gly()
    pairs = [build_mirror_rotamers(database, torch_device, side) for side in ("l", "d")]
    mapping = mirror_rotamer_permutation(*pairs)
    # Map D sparse interaction indices into L conformer order. Comparing the
    # dense values also handles tiny support differences near a cutoff.
    inverse = torch.tensor(numpy.argsort(mapping), device=torch_device)
    sfxn = beta2016_score_function(torch_device, param_db=database)
    modules = [sfxn.render_rotamer_scoring_module(*pair) for pair in pairs]
    n_rotamers = len(mapping)
    with torch.no_grad():
        for term, left, right in zip(
            sfxn.all_terms(), modules[0].term_modules, modules[1].term_modules
        ):
            l_values, l_indices = left(pairs[0][1].coords)
            r_values, r_indices = right(pairs[1][1].coords)
            r_indices = r_indices.clone().long()
            r_indices[1:] = inverse[r_indices[1:]]
            for score_type, l_score, r_score in zip(
                term.score_types(), l_values, r_values
            ):
                shape = (1, n_rotamers, n_rotamers)
                l_dense = (
                    torch.sparse_coo_tensor(l_indices, l_score, shape)
                    .coalesce()
                    .to_dense()
                )
                r_dense = (
                    torch.sparse_coo_tensor(r_indices, r_score, shape)
                    .coalesce()
                    .to_dense()
                )
                assert torch.isfinite(l_dense).all() and torch.isfinite(r_dense).all()
                torch.testing.assert_close(
                    l_dense, r_dense, atol=5e-4, rtol=5e-5, msg=str(score_type)
                )


def test_single_position_mirror_packing_reaches_the_enumerated_minimum(
    default_database, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("Simulated annealing is a CUDA implementation")
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask, pack_rotamers
    from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
    from tmol.tests.score.test_mirror_image_scoring import _pose, MIRROR_PAIR

    database = default_database.with_symmetric_gly()
    sfxn = beta2016_score_function(torch_device, param_db=database)
    packed_pairs, minima = [], []
    for side in ("l", "d"):
        pose = _pose(f"{MIRROR_PAIR}_{side}", database, torch_device)
        task = PackerTask(pose, PackerPalette())
        # Fix chemical identities and every other coordinate, leaving one
        # semirotameric PHE sidechain whose complete choice set is affordable.
        task.per_block_is_block_type_allowed &= (
            task.per_block_considered_block_types_is_orig
        )
        task.add_conformer_sampler(IncludeCurrentSampler())
        types = pose.packed_block_types.active_block_types
        block = next(
            i
            for i, ti in enumerate(pose.block_type_ind64[0].tolist())
            if ti >= 0 and types[ti].dunbrack_reference == "PHE"
        )
        mask = torch.zeros_like(task.is_real_block)
        mask[0, block] = True
        task.add_conformer_sampler_by_block_mask(
            create_dunbrack_sampler_from_database(database, torch_device), mask
        )
        built, rotamers = build_rotamers(
            pose, SetPackerTask.from_packer_task(task), database.chemical
        )
        choices = torch.nonzero(rotamers.block_ind_for_rot == block).flatten()
        assert len(choices) > 2
        whole = sfxn.render_whole_pose_scoring_module(built)
        offset = int(built.block_coord_offset[0, block])
        ti = int(built.block_type_ind64[0, block])
        n_atoms = built.packed_block_types.active_block_types[ti].n_atoms
        energies = []
        with torch.no_grad():
            for rotamer in choices.tolist():
                coords = built.coords.clone()
                rot_offset = int(rotamers.coord_offset_for_rot[rotamer])
                coords[0, offset : offset + n_atoms] = rotamers.coords[
                    rot_offset : rot_offset + n_atoms
                ]
                energies.append(whole(coords).sum())
            minimum = torch.stack(energies).min()
            packed = pack_rotamers(pose, sfxn, task)
            energy = sfxn.render_whole_pose_scoring_module(packed)(packed.coords).sum()
            torch.testing.assert_close(energy, minimum, atol=2e-3, rtol=1e-5)
        minima.append(minimum)
        # Represent each packed block as one conformer, reusing the full-atom
        # matching oracle (which permits only equivalent hydrogen exchanges).
        current_task = PackerTask(packed, PackerPalette())
        current_task.per_block_is_block_type_allowed &= (
            current_task.per_block_considered_block_types_is_orig
        )
        current_task.add_conformer_sampler(IncludeCurrentSampler())
        packed_pairs.append(
            build_rotamers(
                packed,
                SetPackerTask.from_packer_task(current_task),
                database.chemical,
            )
        )
    torch.testing.assert_close(minima[0], minima[1], atol=2e-3, rtol=1e-5)
    mirror_rotamer_permutation(*packed_pairs)
