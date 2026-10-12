import attr
import pytest
import torch

from tmol.chemical import l_base_name
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
from tmol.pose import PoseStackBuilder
from tmol.score import (
    ScoreType,
    beta2016_score_function,
    beta_nov16_dens_score_function,
)
from tmol.score.density import (
    DensityEnergyTerm,
    FastDensityScore,
    block_type_atomic_numbers,
)
from tmol.score.density._density_energy_term import sidechain_atom_mask
from tmol.score.density.density import frac_to_cart

from .conftest import (
    EM_MAP,
    EM_MODEL,
    XRAY_CARBON_SIGMA,
    XTAL_MAP,
    XTAL_MODEL,
    load_map,
    model_pose,
    reference_score_grid,
    smoothed_node_value,
)

MAPS = {
    "em": (EM_MAP, EM_MODEL),
    "xtal": (XTAL_MAP, XTAL_MODEL),
}


def _scorer(key, periodic=True):
    """The X-ray-table score grid; periodic by default, as Rosetta scores."""
    return FastDensityScore(
        load_map(MAPS[key][0]), XRAY_CARBON_SIGMA, periodic=periodic
    )


def _node_positions(density_map, nodes, pad=(0, 0, 0)):
    index = torch.tensor(nodes, dtype=torch.float64) - torch.tensor(pad)
    return density_map.origin + index @ density_map.voxel_basis.T


@pytest.mark.parametrize("key", ["em", "xtal"])
def test_score_grid_matches_direct_sum_at_nodes(key):
    density_map = load_map(MAPS[key][0])
    grid = reference_score_grid(density_map, XRAY_CARBON_SIGMA)
    n = grid.shape
    nodes = [
        (0, 0, 0),
        (n[0] - 1, n[1] - 1, n[2] - 1),
        (n[0] // 2, n[1] // 3, n[2] // 4),
        tuple(int(i) for i in (grid == grid.max()).nonzero()[0]),
    ]
    expected = torch.tensor([smoothed_node_value(grid, node) for node in nodes])
    scores = _scorer(key)(_node_positions(density_map, nodes))
    torch.testing.assert_close(scores, expected.double(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("key", ["em", "xtal"])
def test_score_grid_matches_rosetta(key, rosetta_reference):
    points = torch.tensor(rosetta_reference[key]["points"], dtype=torch.float64)
    torch.testing.assert_close(
        _scorer(key)(points[:, :3]), points[:, 3], atol=2e-6, rtol=1e-5
    )


@pytest.mark.parametrize("key", ["em", "xtal"])
@pytest.mark.parametrize("scatterers", ["xray", "electron"])
def test_total_matches_rosetta(
    key, scatterers, rosetta_reference, default_database, torch_device
):
    name, model = MAPS[key]
    pose_stack = model_pose(
        model, torch_device, ligands=True, density_map=load_map(name)
    )
    term = _term(
        pose_stack,
        default_database,
        torch_device,
        density_scatterers=scatterers,
        density_periodic=True,
    )
    total = term.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)
    expected = rosetta_reference[key]["total"][scatterers]
    assert float(total) == pytest.approx(expected, abs=1e-3)


def test_score_grid_is_periodic_and_differentiable(xtal_map):
    scorer = _scorer("xtal")
    lattice = xtal_map.voxel_basis @ torch.tensor([60.0, -76.0, 104.0]).double()
    points = torch.tensor([[3.7, 5.2, -1.1], [12.0, 20.5, 31.3]], dtype=torch.float64)
    torch.testing.assert_close(scorer(points), scorer(points + lattice))
    points = points.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(scorer, (points,))


def test_crystal_symmetry_mates_score_equally():
    """Atoms related by the P21 screw axis of 7RSA score the same."""
    scorer = _scorer("xtal")
    f2c = torch.from_numpy(frac_to_cart((30.18, 38.40, 53.32, 90.0, 105.85, 90.0)))
    xyz = model_pose(XTAL_MODEL, torch.device("cpu")).coords[0].double()
    xyz = xyz[torch.isfinite(xyz).all(-1)][::7]
    frac = xyz @ torch.linalg.inv(f2c).T
    mate = frac * torch.tensor([-1.0, 1.0, -1.0]) + torch.tensor([0.0, 0.5, 0.0])
    torch.testing.assert_close(scorer(xyz), scorer(mate @ f2c.T), atol=1e-6, rtol=1e-5)


def test_nonperiodic_map_does_not_wrap(em_map):
    """Outside the EM box the open map matches a zero-padded direct sum."""
    open_box = _scorer("em", periodic=False)
    grid = reference_score_grid(em_map, XRAY_CARBON_SIGMA, pad=open_box.pad)
    n, p = grid.shape, open_box.pad
    # just past each face of the original box, and inside it
    nodes = [
        (p[0] - 2, n[1] // 2, n[2] // 2),
        (n[0] - p[0] + 1, n[1] // 2, n[2] // 2),
        (n[0] // 2, n[1] // 2, p[2] - 2),
        (n[0] // 2, n[1] // 2, n[2] - p[2] + 1),
        (n[0] // 2, n[1] // 2, n[2] // 2),
    ]
    expected = torch.tensor([smoothed_node_value(grid, node) for node in nodes])
    scores = open_box(_node_positions(em_map, nodes, pad=p))
    torch.testing.assert_close(scores, expected.double(), atol=1e-6, rtol=1e-5)

    far = em_map.origin - 100.0
    assert float(open_box(far[None])) == 0
    point = _node_positions(em_map, [nodes[0]], pad=p).clone().requires_grad_(True)
    assert torch.autograd.gradcheck(open_box, (point,))


def _term(pose_stack, default_database, device, **options):
    term = DensityEnergyTerm(param_db=default_database, device=device)
    term.set_options(options)
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    return term


def _pose_atomic_numbers(pose_stack):
    """Atomic numbers aligned to pose 0's coordinates."""
    pbt = pose_stack.packed_block_types
    z_bt = block_type_atomic_numbers(pbt)
    z = torch.zeros(pose_stack.max_n_pose_atoms, dtype=torch.int64)
    for block, bt in enumerate(pose_stack.block_type_ind64[0].tolist()):
        if bt >= 0:
            offset = int(pose_stack.block_coord_offset64[0, block])
            n_atoms = int(pbt.n_atoms[bt])
            z[offset : offset + n_atoms] = z_bt[bt, :n_atoms]
    return z


def test_whole_pose_energy_matches_direct_sum_and_gradient(
    xtal_map, default_database, torch_device
):
    pose_stack = model_pose(
        XTAL_MODEL, torch_device, first=1, last=4, density_map=xtal_map
    )
    term = _term(
        pose_stack,
        default_database,
        torch_device,
        density_scatterers="xray",
        density_periodic=True,
    )
    module = term.render_whole_pose_scoring_module(pose_stack)

    coords = pose_stack.coords.detach().double().requires_grad_(True)
    energy = module(coords)
    assert energy.shape == (1, 1)

    scorer = _scorer("xtal")
    z = _pose_atomic_numbers(pose_stack).to(torch_device)
    xray_weight = {6: 1.0, 7: 7 / 6, 8: 8 / 6, 16: 16 / 6}
    weight = torch.tensor(
        [xray_weight.get(int(zi), 0.0) for zi in z], dtype=torch.float64
    ).to(torch_device)
    real = weight != 0
    expected = -(weight[real] * scorer(coords[0][real].cpu()).to(torch_device)).sum()
    torch.testing.assert_close(energy[0, 0], expected, atol=1e-8, rtol=1e-8)
    assert float(energy.detach()) < 0  # the deposited model sits in its density

    (gradient,) = torch.autograd.grad(energy.sum(), coords)
    assert torch.all(gradient[0][z <= 1] == 0)
    assert torch.autograd.gradcheck(
        lambda x: module(x), (coords.detach().clone().requires_grad_(True),), atol=1e-5
    )


def test_defaults_are_cryoem(em_map, default_database, torch_device):
    """Electron scattering and an open (non-periodic) map unless set otherwise."""
    pose_stack = model_pose(
        EM_MODEL, torch_device, first=240, last=250, density_map=em_map
    )
    default = _term(pose_stack, default_database, torch_device)
    assert (default.scatterers, default.periodic) == ("electron", False)
    explicit = _term(
        pose_stack,
        default_database,
        torch_device,
        density_scatterers="electron",
        density_periodic=False,
    )
    coords = pose_stack.coords
    torch.testing.assert_close(
        default.render_whole_pose_scoring_module(pose_stack)(coords),
        explicit.render_whole_pose_scoring_module(pose_stack)(coords),
    )
    sfxn = beta_nov16_dens_score_function(torch_device)
    assert "density_scatterers" not in sfxn.term_options
    assert "density_periodic" not in sfxn.term_options


def test_scatterer_tables_set_atom_weights(xtal_map, default_database, torch_device):
    pose_stack = model_pose(
        XTAL_MODEL, torch_device, first=1, last=30, density_map=xtal_map
    )
    pbt = pose_stack.packed_block_types
    z = block_type_atomic_numbers(pbt, torch_device)
    for table, expected in [
        ("xray", {1: 0.0, 6: 1.0, 7: 7 / 6, 8: 8 / 6, 16: 16 / 6}),
        ("electron", {1: 0.0, 6: 1.0, 7: 5 / 6, 8: 4 / 6, 16: 12 / 6}),
    ]:
        term = _term(
            pose_stack,
            default_database,
            torch_device,
            density_scatterers=table,
        )
        weight = term._block_type_atom_weights(pbt)
        for atomic_number, w in expected.items():
            selected = weight[z == atomic_number]
            assert selected.numel() > 0
            torch.testing.assert_close(selected, torch.full_like(selected, w))


def test_sidechain_scale_option(xtal_map, default_database, torch_device):
    pose_stack = model_pose(XTAL_MODEL, torch_device, density_map=xtal_map)
    pbt = pose_stack.packed_block_types
    plain = _term(pose_stack, default_database, torch_device)
    plain = plain._block_type_atom_weights(pbt)
    scale = {"LYS": 0.5, "PRO": 0.25, "ILE": 0.75}
    scaled = _term(
        pose_stack,
        default_database,
        torch_device,
        density_sc_scale=scale,
    )._block_type_atom_weights(pbt)

    backbone = {"N", "CA", "C", "O", "OXT"}
    checked = set()
    for i, block_type in enumerate(pbt.active_block_types):
        for j, atom in enumerate(block_type.atoms):
            # d-amino acids take the scale of the l form they mirror
            factor = scale.get(l_base_name(block_type), 1.0)
            if atom.name in backbone:
                factor = 1.0
            elif factor != 1.0 and plain[i, j] != 0:
                checked.add((block_type.base_name, atom.name))
            torch.testing.assert_close(scaled[i, j], plain[i, j] * factor)
    assert {("PRO", "CD"), ("LYS", "NZ"), ("ILE", "CB"), ("DLYS", "NZ")} <= checked


def test_sidechain_atoms_hang_off_interior_mainchain(torch_device):
    pose_stack = model_pose(XTAL_MODEL, torch_device)
    pbt = pose_stack.packed_block_types
    by_name = {bt.base_name: bt for bt in pbt.active_block_types}
    for name, sidechain in [
        ("GLY", {"HA2", "HA3"}),
        ("ALA", {"CB", "HA", "HB1", "HB2", "HB3"}),
        ("PRO", {"CB", "CG", "CD", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"}),
    ]:
        bt = by_name[name]
        mask = sidechain_atom_mask(bt)
        assert {a.name for a, m in zip(bt.atoms, mask, strict=True) if m} == sidechain


def test_block_pair_and_rotamer_scoring_match_whole_pose(
    em_map, default_database, torch_device
):
    pose_stack = model_pose(
        EM_MODEL, torch_device, first=240, last=260, density_map=em_map
    )
    term = _term(pose_stack, default_database, torch_device)
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
    assert rotamer_pose.density_map is em_map
    rotamer_term = _term(rotamer_pose, default_database, torch_device)
    scores, indices = rotamer_term.render_rotamer_scoring_module(
        rotamer_pose, rotamers
    )(rotamers.coords)
    assert indices.shape[0] == 3 and torch.all(indices[1] == indices[2])
    torch.testing.assert_close(scores.sum(), whole.sum(), atol=1e-4, rtol=1e-4)


def test_pose_stack_carries_one_density_map(xtal_map, em_map, torch_device):
    pose_stack = model_pose(
        XTAL_MODEL, torch_device, first=1, last=3, density_map=xtal_map
    )
    assert pose_stack.clone().density_map is xtal_map
    assert pose_stack.clone_sharing_topology().density_map is xtal_map
    assert pose_stack.split(0).density_map is xtal_map
    joined = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    assert joined.density_map is xtal_map
    other = attr.evolve(pose_stack, density_map=em_map)
    with pytest.raises(ValueError, match="density maps"):
        PoseStackBuilder.from_poses([pose_stack, other], torch_device)


def test_missing_map_is_an_error(default_database, torch_device):
    pose_stack = model_pose(XTAL_MODEL, torch_device, first=1, last=3)
    term = DensityEnergyTerm(param_db=default_database, device=torch_device)
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    with pytest.raises(ValueError, match="density_map"):
        term.render_whole_pose_scoring_module(pose_stack)


def test_beta_nov16_dens_differs_from_beta2016_only_by_the_density_term(
    xtal_map, torch_device
):
    pose_stack = model_pose(
        XTAL_MODEL, torch_device, first=1, last=20, density_map=xtal_map
    )
    sfxn = beta_nov16_dens_score_function(torch_device)
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
