import math

import pytest
import torch

from tmol import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
from tmol.score import (
    ScoreType,
    beta2016_score_function,
    beta_nov16_dens_score_function,
)
from tmol.score.density import (
    DensityCorrelation,
    DensityEnergyTerm,
    ElectronDensityMap,
    FastDensityScore,
    atomic_numbers_for_pose_stack,
)

RESOLUTION = 3.0


def _random_map(dtype=torch.float64, n=14):
    generator = torch.Generator().manual_seed(0)
    return ElectronDensityMap(
        torch.rand((n, n + 2, n + 4), generator=generator, dtype=dtype),
        torch.tensor((-5.0, -6.0, -7.0), dtype=dtype),
        torch.tensor((1.0, 1.1, 1.2), dtype=dtype),
    )


def _map_around(pose_stack, dtype=torch.float64):
    """A synthetic map of the pose's heavy atoms, on a grid that contains the pose."""
    coords = pose_stack.coords[0].to(dtype)
    z = atomic_numbers_for_pose_stack(pose_stack)[0]
    real = z > 1
    low, high = coords[real].min(0).values - 8.0, coords[real].max(0).values + 8.0
    voxel = 1.0
    n = [int(math.ceil(float(h - l) / voxel)) for h, l in zip(high, low, strict=True)]
    template = ElectronDensityMap(
        torch.zeros((n[2], n[1], n[0]), dtype=dtype, device=coords.device),
        low,
        torch.full((3,), voxel, dtype=dtype, device=coords.device),
    )
    synth = DensityCorrelation(template, RESOLUTION).synthesize_density(
        coords[real], z[real]
    )
    return ElectronDensityMap(synth.detach(), template.origin, template.voxel_size)


def _term(pose_stack, density_map, default_database, device, **options):
    term = DensityEnergyTerm(param_db=default_database, device=device)
    term.set_options(
        {
            "density_map": density_map,
            "density_resolution": RESOLUTION,
            "density_scale_sidechains": False,
            **options,
        }
    )
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    return term


def test_score_grid_interpolates_nodes_and_is_differentiable():
    density_map = _random_map()
    scorer = FastDensityScore(density_map, RESOLUTION)
    nz, ny, nx = density_map.density.shape
    # the cubic B-spline interpolates the grid: at a voxel centre it returns the
    # normalized score exactly
    nodes = torch.tensor(
        [[3, 4, 5], [0, 0, 0], [nx - 1, ny - 2, nz - 3]], dtype=torch.float64
    )
    positions = density_map.origin + nodes * density_map.voxel_size
    expected = scorer.score[nodes[:, 0].long(), nodes[:, 1].long(), nodes[:, 2].long()]
    torch.testing.assert_close(scorer(positions), expected, atol=1e-10, rtol=1e-8)
    # periodic: one box length away reads the same value
    box = torch.tensor([nx, ny, nz], dtype=torch.float64) * density_map.voxel_size
    point = torch.tensor([[0.37, 0.52, -1.1]], dtype=torch.float64)
    torch.testing.assert_close(
        scorer(point), scorer(point + box), atol=1e-10, rtol=1e-8
    )
    # coordinate gradient
    point = point.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(scorer, (point,))


def test_whole_pose_energy_matches_direct_sum_and_gradient(
    ubq_pdb, default_database, torch_device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    density_map = _map_around(pose_stack).to(torch_device)
    term = _term(pose_stack, density_map, default_database, torch_device)
    module = term.render_whole_pose_scoring_module(pose_stack)

    coords = pose_stack.coords.detach().double().requires_grad_(True)
    energy = module(coords)
    assert energy.shape == (1, 1)

    scorer = FastDensityScore(density_map, RESOLUTION)
    z = atomic_numbers_for_pose_stack(pose_stack)[0]
    heavy = z > 1
    expected = -(scorer.amplitude(z)[heavy] * scorer(coords[0][heavy])).sum()
    torch.testing.assert_close(energy[0, 0], expected, atol=1e-8, rtol=1e-8)
    assert (
        float(energy.detach()) < 0
    )  # the map was built from this pose, so its heavy atoms sit in density

    # analytic coordinate gradient is correct and zero on hydrogens
    (gradient,) = torch.autograd.grad(energy.sum(), coords)
    assert torch.all(gradient[0][~heavy] == 0)
    assert torch.autograd.gradcheck(
        lambda x: module(x), (coords.detach().clone().requires_grad_(True),), atol=1e-5
    )


# scale_sc_dens_byres of Rosetta's cryoem_glycan_refinement.xml, by one-letter code
_ROSETTA_SC_SCALE = {
    **dict.fromkeys("RKEDM", 0.66),
    **dict.fromkeys("CQHNTS", 0.71),
    **dict.fromkeys("YWAFPILV", 0.78),
}
_THREE_TO_ONE = {
    "ARG": "R",
    "LYS": "K",
    "GLU": "E",
    "ASP": "D",
    "MET": "M",
    "CYS": "C",
    "GLN": "Q",
    "HIS": "H",
    "ASN": "N",
    "THR": "T",
    "SER": "S",
    "TYR": "Y",
    "TRP": "W",
    "ALA": "A",
    "PHE": "F",
    "PRO": "P",
    "ILE": "I",
    "LEU": "L",
    "VAL": "V",
    "GLY": "G",
}


def test_sidechain_scale_follows_rosetta_cryoem_script(
    ubq_pdb, default_database, torch_device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=76)
    density_map = _map_around(pose_stack).to(torch_device)
    pbt = pose_stack.packed_block_types
    scorer = FastDensityScore(density_map, RESOLUTION)
    plain = _term(
        pose_stack, density_map, default_database, torch_device
    )._block_type_atom_weights(pbt, scorer)
    scaled = _term(
        pose_stack,
        density_map,
        default_database,
        torch_device,
        density_scale_sidechains=True,
    )
    scaled = scaled._block_type_atom_weights(pbt, scorer)
    checked = set()
    for i, block_type in enumerate(pbt.active_block_types):
        letter = _THREE_TO_ONE.get(block_type.base_name)
        for j, atom in enumerate(block_type.atoms):
            if plain[i, j] == 0:
                assert scaled[i, j] == 0  # hydrogens and padding never score
            elif (
                atom.name in ("N", "CA", "C", "O", "OXT")
                or letter not in _ROSETTA_SC_SCALE
            ):
                assert scaled[i, j] == plain[i, j], (block_type.name, atom.name)
            else:
                torch.testing.assert_close(
                    scaled[i, j], plain[i, j] * _ROSETTA_SC_SCALE[letter]
                )
                checked.add(letter)
    assert len(checked) >= 15  # ubiquitin exercises most residue types


def test_block_pair_and_rotamer_scoring_match_whole_pose(
    ubq_pdb, default_database, torch_device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    density_map = _map_around(pose_stack).to(torch_device)
    term = _term(pose_stack, density_map, default_database, torch_device)
    coords = pose_stack.coords.detach()
    whole = term.render_whole_pose_scoring_module(pose_stack)(coords)

    block_pair = term.render_block_pair_scoring_module(pose_stack)(coords)
    torch.testing.assert_close(block_pair.sum(dim=(-1, -2)), whole)
    off_diagonal = block_pair - torch.diag_embed(
        torch.diagonal(block_pair, dim1=-2, dim2=-1)
    )
    assert torch.all(off_diagonal == 0)  # a one-body term has no block-pair entries

    task = PackerTask(pose_stack, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task = SetPackerTask.from_packer_task(task)
    rotamer_pose, rotamers = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    rotamer_term = _term(rotamer_pose, density_map, default_database, torch_device)
    scores, indices = rotamer_term.render_rotamer_scoring_module(
        rotamer_pose, rotamers
    )(rotamers.coords)
    assert indices.shape[0] == 3 and torch.all(
        indices[1] == indices[2]
    )  # one-body: rotamer on the diagonal
    torch.testing.assert_close(scores.sum(), whole.sum(), atol=1e-4, rtol=1e-4)


def test_missing_map_is_an_error(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=3)
    term = DensityEnergyTerm(param_db=default_database, device=torch_device)
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    with pytest.raises(ValueError, match="density_map"):
        term.render_whole_pose_scoring_module(pose_stack)


def test_beta_nov16_dens_differs_from_beta2016_only_by_the_density_term(
    ubq_pdb, torch_device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    density_map = _map_around(pose_stack).to(torch_device)
    sfxn = beta_nov16_dens_score_function(torch_device, density_map, RESOLUTION)
    plain = beta2016_score_function(torch_device)

    for score_type in ScoreType:
        if score_type is ScoreType.n_score_types:
            continue
        expected = (
            35.0
            if score_type is ScoreType.elec_dens_fast
            else float(plain.get_weight(score_type))
        )
        assert float(sfxn.get_weight(score_type)) == pytest.approx(expected), score_type

    # the density term enters the total with its weight, and nothing else changes
    coords = pose_stack.coords.detach()
    total = sfxn.render_whole_pose_scoring_module(pose_stack)(coords)
    sfxn.set_weight(ScoreType.elec_dens_fast, 0.0)
    without_density = sfxn.render_whole_pose_scoring_module(pose_stack)(coords)
    torch.testing.assert_close(
        without_density, plain.render_whole_pose_scoring_module(pose_stack)(coords)
    )
    for score_type in ScoreType:
        if score_type not in (ScoreType.n_score_types, ScoreType.elec_dens_fast):
            sfxn.set_weight(score_type, 0.0)
    sfxn.set_weight(ScoreType.elec_dens_fast, 1.0)
    unit_density = sfxn.render_whole_pose_scoring_module(pose_stack)(coords)
    torch.testing.assert_close(
        total - without_density, 35.0 * unit_density, rtol=1e-4, atol=1e-3
    )
    assert float(unit_density) < 0
