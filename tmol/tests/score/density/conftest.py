import functools
import math

import attr
import pytest
import torch
import yaml

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.score.density import read_mrc
from tmol.tests.data import data_path

EM_MAP, EM_MODEL, EM_RESOLUTION = "emd_52727.map.gz", "9i8j.cif.zst", 2.9
XTAL_MAP, XTAL_MODEL, XTAL_RESOLUTION = (
    "7rsa_2mFo-DFc.ccp4.gz",
    "7rsa_final.cif.zst",
    1.26,
)
XRAY_CARBON_SIGMA = 4.88398
RESOLUTION = {EM_MAP: EM_RESOLUTION, XTAL_MAP: XTAL_RESOLUTION}


@functools.lru_cache(maxsize=None)
def load_map(name):
    path = data_path("density", name)
    return read_mrc(path, RESOLUTION[name], dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def load_atoms(name):
    return atom_array_from_cif(data_path("density", name))


def model_pose(name, device, first=None, last=None, ligands=False, density_map=None):
    """Pose of a fixture model: protein residues first..last, or everything."""
    atoms = load_atoms(name)
    keep = ~atoms.hetero | (ligands & (atoms.res_name != "HOH"))
    if first is not None:
        keep &= ~atoms.hetero & (atoms.res_id >= first) & (atoms.res_id <= last)
    pose_stack = pose_stack_from_biotite(atoms[keep], device, prepare_ligands=ligands)
    return attr.evolve(pose_stack, density_map=density_map)


@pytest.fixture(scope="session")
def em_map():
    return load_map(EM_MAP)


@pytest.fixture(scope="session")
def xtal_map():
    return load_map(XTAL_MAP)


@pytest.fixture(scope="session")
def rosetta_reference():
    with open(data_path("density", "rosetta_elec_dens_fast.yaml")) as infile:
        return yaml.safe_load(infile)["maps"]


def reference_score_grid(density_map, carbon_sigma, pad=(0, 0, 0)):
    """Normalized elec_dens_fast grid [x, y, z] by a direct sum over kernel offsets.

    The map is zero-padded by pad voxels per axis, then treated as periodic.
    """
    rho = density_map.density.permute(2, 1, 0).double()
    rho = rho - rho.mean()
    rho = torch.nn.functional.pad(rho, (pad[2],) * 2 + (pad[1],) * 2 + (pad[0],) * 2)
    basis = density_map.voxel_basis.double()
    spacing = basis.norm(dim=0).max()
    b_eff = 4.0 * max(2.0 * float(spacing), density_map.resolution) ** 2
    k = min(math.pi**2 / carbon_sigma, 4.0 * math.pi**2 / b_eff)
    r2_max = (1.5 * math.log(k) - math.log(1e-4)) / k
    reach = [
        math.ceil(math.sqrt(r2_max) * float(r))
        for r in torch.linalg.inv(basis).norm(dim=1)
    ]
    score = torch.zeros_like(rho)
    for i in range(-reach[0], reach[0] + 1):
        for j in range(-reach[1], reach[1] + 1):
            for m in range(-reach[2], reach[2] + 1):
                d2 = float(
                    (basis @ torch.tensor([i, j, m], dtype=torch.float64))
                    .square()
                    .sum()
                )
                if d2 <= r2_max:
                    w = k**1.5 * math.exp(-k * d2)
                    score += w * torch.roll(rho, (i, j, m), dims=(0, 1, 2))
    flat = score.reshape(-1).sort().values
    return score / (4.0 * (flat[-5] - flat[4]))


def smoothed_node_value(grid, node):
    """The cubic B-spline with coefficients grid, read at an integer node."""
    w = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    n = grid.shape
    total = 0.0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                index = tuple((node[i] + d - 1) % n[i] for i, d in enumerate((a, b, c)))
                total += w[a] * w[b] * w[c] * float(grid[index])
    return total
