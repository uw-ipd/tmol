import os

import pytest
import torch
from yaml import safe_load

from tmol.io import atom_array_from_cif, pose_stack_from_biotite, pose_stack_from_pdb
from tmol.score.metal._metal_coordination_term import (
    MetalCoordinationEnergyTerm,
    fan_energies,
    site_energies,
)
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("metal_fixtures")

with open(os.path.join(FIXTURE_DIR, "expected.yaml")) as infile:
    EXPECTED = safe_load(infile)["fixtures"]

LOADABLE = [stem for stem, spec in EXPECTED.items() if "xfail" not in spec]


@pytest.fixture(scope="module")
def built():
    cache = {}

    def build(stem):
        if stem not in cache:
            structure = atom_array_from_cif(os.path.join(FIXTURE_DIR, stem + ".cif.gz"))
            cache[stem] = pose_stack_from_biotite(
                structure, torch.device("cpu"), prepare_ligands=True
            )
        return cache[stem]

    return build


def render(term, pose_stack, block_pair=False):
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    if block_pair:
        return term.render_block_pair_scoring_module(pose_stack)
    return term.render_whole_pose_scoring_module(pose_stack)


def global_index(pose_stack, pose, block, atom):
    return int(pose_stack.block_coord_offset64[pose, block]) + atom


def idealized(term, pose_stack):
    """Coordinates with every paired donor moved onto its site at ideal distance."""
    coords = pose_stack.coords.detach().clone().double()
    site_rows, site_d0, _, _ = term.restraints(pose_stack)
    for (pose, mblock, matom, vatom, dblock, datom), d0 in zip(site_rows, site_d0):
        metal = coords[pose, global_index(pose_stack, pose, mblock, matom)]
        donor = global_index(pose_stack, pose, dblock, datom)
        if vatom >= 0:
            ray = coords[pose, global_index(pose_stack, pose, mblock, vatom)] - metal
        else:
            ray = coords[pose, donor] - metal
        coords[pose, donor] = metal + d0 * ray / ray.norm()
    return coords, site_rows, site_d0


@pytest.mark.parametrize("stem", LOADABLE)
def test_site_pairing_matches_detection(built, stem, default_database):
    pose_stack = built(stem)
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    site_rows, _, _, _ = term.restraints(pose_stack)
    info = pose_stack.pdb_info
    pbt = pose_stack.packed_block_types

    def key(pose, block):
        return (
            str(info.chain_labels[pose, block]),
            int(info.residue_labels[pose, block]),
        )

    found = {}
    for pose, mblock, _, _, dblock, datom in site_rows:
        bt = pbt.active_block_types[int(pose_stack.block_type_ind[pose, dblock])]
        found.setdefault(key(pose, mblock), set()).add(
            (*key(pose, dblock), bt.atoms[datom].name)
        )
    for metal in EXPECTED[stem]["metals"]:
        expected = {
            (d["chain"], d["res"], d["atom"])
            for d in EXPECTED[stem]["donors"][
                f"{metal['chain']}/{metal['comp']}/{metal['res']}"
            ]
        }
        assert found.get((metal["chain"], metal["res"]), set()) == expected


@pytest.mark.parametrize("stem", LOADABLE)
def test_ideal_sites_score_zero(built, stem, default_database):
    pose_stack = built(stem)
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    scorer = render(term, pose_stack)
    coords, _, _ = idealized(term, pose_stack)
    assert float(scorer(coords).sum()) < 1e-6


def test_known_distortions(built, default_database):
    pose_stack = built("zn_tetrahedral_3ks3")
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    scorer = render(term, pose_stack)
    coords, site_rows, _ = idealized(term, pose_stack)
    sd_radial, sd_lateral, _ = term.widths

    pose, mblock, matom, _, dblock, datom = site_rows[0]
    metal = coords[pose, global_index(pose_stack, pose, mblock, matom)]
    donor = global_index(pose_stack, pose, dblock, datom)
    u = (coords[pose, donor] - metal) / (coords[pose, donor] - metal).norm()
    perp = torch.linalg.cross(u, torch.tensor([1.0, 0.0, 0.0], dtype=u.dtype))
    perp = perp / perp.norm()

    radial = coords.clone()
    radial[pose, donor] += 0.05 * u
    assert float(scorer(radial).sum()) == pytest.approx(
        (0.05 / sd_radial) ** 2, rel=1e-5
    )

    # sideways by s leaves the distance off by sqrt(d0^2 + s^2) - d0
    lateral = coords.clone()
    lateral[pose, donor] += 0.1 * perp
    d0 = float((coords[pose, donor] - metal).norm())
    stretch = (d0**2 + 0.01) ** 0.5 - d0
    expected = (0.1 / sd_lateral) ** 2 + (stretch / sd_radial) ** 2
    assert float(scorer(lateral).sum()) == pytest.approx(expected, rel=1e-5)


def test_built_fan_is_at_rest(built, default_database):
    pose_stack = built("cu_zn_sod_3f7l")
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    _, _, fan_rows, fan_l0 = term.restraints(pose_stack)
    assert len(fan_rows) > 0
    coords = pose_stack.coords.double()
    for (pose, block, a, b), l0 in zip(fan_rows, fan_l0):
        xyz_a = coords[pose, global_index(pose_stack, pose, block, a)]
        xyz_b = coords[pose, global_index(pose_stack, pose, block, b)]
        sep = xyz_a - xyz_b
        assert float(sep.norm()) == pytest.approx(l0, abs=1e-4)


def test_energy_gradcheck():
    torch.manual_seed(0)
    n = 5
    widths = torch.tensor([0.1, 0.25, 0.05], dtype=torch.float64)
    metal = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    donor = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    virt = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    has_virt = torch.tensor([True, True, False, True, False])
    d0 = torch.full((n,), 2.1, dtype=torch.float64)
    torch.autograd.gradcheck(
        lambda m, d, v: site_energies(m, d, v, has_virt, d0, widths),
        (metal, donor, virt),
    )
    torch.autograd.gradcheck(
        lambda a, b: fan_energies(a, b, d0, widths), (metal, donor)
    )


def test_block_pair_sums_to_whole_pose(built, default_database):
    pose_stack = built("mg_one_donor_4e3y")
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    coords, site_rows, _ = idealized(term, pose_stack)
    pose, mblock, matom = site_rows[0][:3]
    coords[pose, global_index(pose_stack, pose, mblock, matom)] += 0.3
    whole = render(term, pose_stack)(coords)
    pairs = render(term, pose_stack, block_pair=True)(coords)
    torch.testing.assert_close(pairs.sum(dim=(2, 3)), whole)


def test_pose_without_metals_is_zero(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    assert term.pose_score_term_is_invariant_zero(pose_stack)
