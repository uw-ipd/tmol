"""Real deposited-site regression tests for explicit metal coordination."""

from io import StringIO

import attr
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile, get_structure
import numpy as np
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.io._metal_bonds import (
    _metal_cross_residue_bonds,
    augment_database_for_metals,
)
from tmol.score import ScoreFunction, ScoreType, beta2016_score_function
from tmol import run_cart_min

# Heavy atoms and deposited CONECT records from PDB 1CA2 (Zn carbonic anhydrase)
# and 1CLL (a seven-coordinate Ca site in calmodulin).
_ZN_SITE = """\
ATOM      1  N   HIS A  94     -11.224  -0.437  10.993  1.00  0.00           N
ATOM      2  CA  HIS A  94     -10.286  -1.397  10.390  1.00  0.00           C
ATOM      3  C   HIS A  94     -10.332  -2.694  11.250  1.00  0.00           C
ATOM      4  O   HIS A  94     -11.009  -2.721  12.306  1.00  0.00           O
ATOM      5  CB  HIS A  94      -8.804  -0.880  10.319  1.00  0.00           C
ATOM      6  CG  HIS A  94      -8.360  -0.482  11.734  1.00  0.00           C
ATOM      7  ND1 HIS A  94      -8.477   0.648  12.357  1.00  0.00           N
ATOM      8  CD2 HIS A  94      -7.755  -1.329  12.614  1.00  0.00           C
ATOM      9  CE1 HIS A  94      -7.986   0.567  13.611  1.00  0.00           C
ATOM     10  NE2 HIS A  94      -7.528  -0.707  13.770  1.00  0.00           N
ATOM     11  N   HIS A  96      -8.072  -6.530  12.430  1.00  0.00           N
ATOM     12  CA  HIS A  96      -6.846  -7.256  12.722  1.00  0.00           C
ATOM     13  C   HIS A  96      -7.309  -8.744  12.547  1.00  0.00           C
ATOM     14  O   HIS A  96      -8.430  -9.033  13.024  1.00  0.00           O
ATOM     15  CB  HIS A  96      -6.188  -7.093  14.129  1.00  0.00           C
ATOM     16  CG  HIS A  96      -5.970  -5.603  14.353  1.00  0.00           C
ATOM     17  ND1 HIS A  96      -4.786  -5.048  14.101  1.00  0.00           N
ATOM     18  CD2 HIS A  96      -6.823  -4.621  14.782  1.00  0.00           C
ATOM     19  CE1 HIS A  96      -4.844  -3.734  14.356  1.00  0.00           C
ATOM     20  NE2 HIS A  96      -6.001  -3.468  14.754  1.00  0.00           N
ATOM     21  N   HIS A 119     -12.039  -2.559  14.939  1.00  0.00           N
ATOM     22  CA  HIS A 119     -11.835  -1.368  15.750  1.00  0.00           C
ATOM     23  C   HIS A 119     -12.500  -0.136  15.066  1.00  0.00           C
ATOM     24  O   HIS A 119     -12.086   0.160  13.911  1.00  0.00           O
ATOM     25  CB  HIS A 119     -10.343  -1.014  15.977  1.00  0.00           C
ATOM     26  CG  HIS A 119      -9.511  -1.987  16.739  1.00  0.00           C
ATOM     27  ND1 HIS A 119      -8.127  -2.175  16.625  1.00  0.00           N
ATOM     28  CD2 HIS A 119      -9.934  -2.873  17.699  1.00  0.00           C
ATOM     29  CE1 HIS A 119      -7.768  -3.117  17.429  1.00  0.00           C
ATOM     30  NE2 HIS A 119      -8.867  -3.564  18.160  1.00  0.00           N
HETATM   31 ZN    ZN A 262      -6.788  -1.621  15.381  1.00  0.00          ZN
HETATM   32  O   HOH A 263      -5.366  -0.357  16.238  1.00  0.00           O
CONECT   10   31
CONECT   20   31
CONECT   27   31
CONECT   31   10   20   27   32
CONECT   32   31
END
"""

_CA_SITE = """\
ATOM      1  N   ASP A  20       4.266  43.121  25.376  1.00  0.00           N
ATOM      2  CA  ASP A  20       3.304  44.245  25.573  1.00  0.00           C
ATOM      3  C   ASP A  20       3.670  44.929  26.900  1.00  0.00           C
ATOM      4  O   ASP A  20       2.889  44.743  27.823  1.00  0.00           O
ATOM      5  CB  ASP A  20       1.892  43.688  25.483  1.00  0.00           C
ATOM      6  CG  ASP A  20       0.838  44.766  25.732  1.00  0.00           C
ATOM      7  OD1 ASP A  20       1.362  45.900  25.680  1.00  0.00           O
ATOM      8  OD2 ASP A  20      -0.327  44.468  26.038  1.00  0.00           O
ATOM      9  N   ASP A  22       3.444  47.960  28.072  1.00  0.00           N
ATOM     10  CA  ASP A  22       2.467  48.811  28.776  1.00  0.00           C
ATOM     11  C   ASP A  22       1.114  48.178  28.881  1.00  0.00           C
ATOM     12  O   ASP A  22       0.189  48.826  29.396  1.00  0.00           O
ATOM     13  CB  ASP A  22       2.549  50.211  28.151  1.00  0.00           C
ATOM     14  CG  ASP A  22       1.791  50.191  26.830  1.00  0.00           C
ATOM     15  OD1 ASP A  22       1.480  49.126  26.287  1.00  0.00           O
ATOM     16  OD2 ASP A  22       1.487  51.294  26.337  1.00  0.00           O
ATOM     17  N   ASP A  24      -1.454  47.546  26.710  1.00  0.00           N
ATOM     18  CA  ASP A  24      -2.616  48.079  25.966  1.00  0.00           C
ATOM     19  C   ASP A  24      -3.219  47.054  25.040  1.00  0.00           C
ATOM     20  O   ASP A  24      -4.213  47.333  24.351  1.00  0.00           O
ATOM     21  CB  ASP A  24      -2.213  49.407  25.299  1.00  0.00           C
ATOM     22  CG  ASP A  24      -1.522  49.054  23.983  1.00  0.00           C
ATOM     23  OD1 ASP A  24      -0.647  48.198  24.144  1.00  0.00           O
ATOM     24  OD2 ASP A  24      -1.796  49.554  22.920  1.00  0.00           O
ATOM     25  N   THR A  26      -1.468  45.522  22.517  1.00  0.00           N
ATOM     26  CA  THR A  26      -0.742  45.570  21.225  1.00  0.00           C
ATOM     27  C   THR A  26       0.732  45.703  21.494  1.00  0.00           C
ATOM     28  O   THR A  26       1.230  46.259  22.472  1.00  0.00           O
ATOM     29  CB  THR A  26      -1.250  46.702  20.252  1.00  0.00           C
ATOM     30  OG1 THR A  26      -0.824  47.960  20.889  1.00  0.00           O
ATOM     31  CG2 THR A  26      -2.752  46.752  19.973  1.00  0.00           C
ATOM     32  N   GLU A  31       8.220  48.516  21.601  1.00  0.00           N
ATOM     33  CA  GLU A  31       8.160  47.434  22.629  1.00  0.00           C
ATOM     34  C   GLU A  31       8.991  46.269  22.136  1.00  0.00           C
ATOM     35  O   GLU A  31       9.814  45.682  22.880  1.00  0.00           O
ATOM     36  CB  GLU A  31       6.723  47.039  22.947  1.00  0.00           C
ATOM     37  CG  GLU A  31       5.971  48.075  23.781  1.00  0.00           C
ATOM     38  CD  GLU A  31       4.517  47.919  23.975  1.00  0.00           C
ATOM     39  OE1 GLU A  31       3.830  47.450  23.067  1.00  0.00           O
ATOM     40  OE2 GLU A  31       3.900  48.247  24.996  1.00  0.00           O
HETATM   41 CA    CA A 149       1.708  47.776  24.197  1.00  0.00          CA
HETATM   42  O   HOH A 181       1.728  50.102  22.982  1.00  0.00           O
CONECT    7   41
CONECT   15   41
CONECT   23   41
CONECT   28   41
CONECT   39   41
CONECT   40   41
CONECT   41    7   15   23   28
CONECT   41   39   40   42
CONECT   42   41
END
"""


def _structure(pdb_text):
    return get_structure(PDBFile.read(StringIO(pdb_text)), model=1, include_bonds=True)


def _block_types(pose):
    return [
        pose.packed_block_types.active_block_types[int(index)]
        for index in pose.block_type_ind64[0]
        if index >= 0
    ]


@pytest.mark.parametrize(
    "site,metal,n_contacts,donor_atoms",
    [
        pytest.param(_ZN_SITE, "ZN", 4, {"NE2", "ND1", "O"}, id="1CA2-zinc"),
        pytest.param(_CA_SITE, "CA", 7, {"OD1", "O", "OE1", "OE2"}, id="1CLL-calcium"),
    ],
)
def test_real_metal_sites_import_multicoordinate_topology(
    torch_device, site, metal, n_contacts, donor_atoms
):
    structure = _structure(site)
    contacts = _metal_cross_residue_bonds(structure)
    assert len(contacts) == n_contacts
    assert {donor[1] for _metal_endpoint, donor in contacts} == donor_atoms

    pose, context = pose_stack_from_biotite(
        structure, torch_device, no_optH=True, return_context=True
    )
    block_types = _block_types(pose)
    metal_block = next(i for i, block in enumerate(block_types) if block.name3 == metal)
    metal_type = block_types[metal_block]
    assert len(metal_type.connections) == n_contacts
    assert (
        torch.count_nonzero(pose.inter_residue_connections64[0, metal_block, :, 0] >= 0)
        == n_contacts
    )
    metal_offset = int(pose.block_coord_offset64[0, metal_block])
    for connection_index in range(n_contacts):
        partner_block, partner_connection = pose.inter_residue_connections64[
            0, metal_block, connection_index
        ].tolist()
        partner_type = block_types[partner_block]
        donor_atom = partner_type.connections[partner_connection].atom
        donor_coord = pose.coords[
            0,
            int(pose.block_coord_offset64[0, partner_block])
            + partner_type.atom_to_idx[donor_atom],
        ]
        proxy_coord = pose.coords[
            0, metal_offset + metal_type.atom_to_idx[f"V{connection_index + 1}"]
        ]
        torch.testing.assert_close(proxy_coord, donor_coord)
    water_type = next(block for block in block_types if block.name3 == "HOH")
    assert {"H1", "H2"} <= set(water_type.properties.virtual)
    assert torch.isfinite(pose.coords).all()
    expected_distance_constraints = 2 * n_contacts + n_contacts * (n_contacts - 1) // 2
    assert pose.constraint_set is not None
    assert pose.constraint_set.constraint_atoms.shape[0] == (
        expected_distance_constraints + n_contacts - 1
    )

    constraint_score_function = ScoreFunction(context.parameter_database, torch_device)
    constraint_score_function.set_weight(ScoreType.constraint, 1.0)
    constraint_scorer = constraint_score_function.render_whole_pose_scoring_module(pose)
    initial_constraint_score = constraint_scorer(pose.coords).sum()
    torch.testing.assert_close(
        initial_constraint_score,
        torch.zeros_like(initial_constraint_score),
        atol=1e-5,
        rtol=0,
    )
    partner_block, partner_connection = pose.inter_residue_connections64[
        0, metal_block, 0
    ].tolist()
    partner_type = block_types[partner_block]
    partner_atom = partner_type.connections[partner_connection].atom
    partner_coord = (
        int(pose.block_coord_offset64[0, partner_block])
        + partner_type.atom_to_idx[partner_atom]
    )
    perturbed = pose.coords.detach().clone()
    perturbed[0, partner_coord, 0] += 0.2
    perturbed.requires_grad_(True)
    perturbed_constraint_score = constraint_scorer(perturbed).sum()
    assert perturbed_constraint_score > 0
    perturbed_constraint_score.backward()
    assert torch.isfinite(perturbed.grad).all()
    minimized = run_cart_min(
        attr.evolve(pose, coords=perturbed.detach()),
        constraint_score_function,
        optimizer_kwargs={"max_iter": 20},
    )
    assert (
        constraint_scorer(minimized.coords).sum() < perturbed_constraint_score.detach()
    )

    score_function = beta2016_score_function(
        torch_device, param_db=context.parameter_database
    )
    score_function.set_weight(ScoreType.constraint, 1.0)
    score = score_function.render_whole_pose_scoring_module(pose)(pose.coords).sum()
    assert torch.isfinite(score)


def test_generated_metal_parameters_match_rosetta_defaults():
    structure = struc.AtomArray(3)
    structure.atom_name = np.asarray(("MG", "CA", "ZN"))
    structure.element = np.asarray(("Mg", "Ca", "Zn"))
    structure.res_name = np.asarray(("MG", "CAL", "ZINC"))
    structure.chain_id[:] = "A"
    structure.res_id = np.arange(1, 4)
    structure.coord = np.zeros((3, 3))
    database = augment_database_for_metals(structure, ParameterDatabase.get_default())
    ljlk = {entry.name: entry for entry in database.scoring.ljlk.atom_type_parameters}
    expected = {
        "Zn2p": (1.09, 0.25, -5.0, 3.5, 5.4),
        "Mg2p": (1.185, 0.015, -5.0, 3.5, 7.0),
        "Ca2p": (1.367, 0.12, 0.0, 2.0, 10.7),
    }
    for name, values in expected.items():
        actual = ljlk[name]
        assert (
            actual.lj_radius,
            actual.lj_wdepth,
            actual.lk_dgfree,
            actual.lk_lambda,
            actual.lk_volume,
        ) == pytest.approx(values)
    assert {residue.name for residue in database.chemical.residues} >= {
        "MG",
        "CAL",
        "ZINC",
    }
    charges = {
        (parameter.res, parameter.atom): parameter.charge
        for parameter in database.scoring.elec.atom_charge_parameters
    }
    assert charges[("MG", "MG")] == 2.0
    assert charges[("CAL", "CA")] == 2.0
    assert charges[("ZINC", "ZN")] == 2.0
