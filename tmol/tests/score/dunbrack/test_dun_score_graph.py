import numpy
import torch

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph


@score_graph
class CartDunbrackGraph(
    CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


@score_graph
class KinematicDunbrackGraph(
    KinematicAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


def test_dunbrack_score_graph_smoke(ubq_system, default_database, torch_device):
    CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def expected_ndihe_from_test_dunbrack_score_setup():
    ndihe_gold = numpy.array(
        [
            [
                5,
                5,
                4,
                4,
                3,
                6,
                3,
                4,
                3,
                6,
                3,
                4,
                3,
                4,
                5,
                3,
                5,
                5,
                3,
                4,
                3,
                4,
                5,
                4,
                3,
                6,
                6,
                4,
                5,
                4,
                6,
                5,
                4,
                5,
                5,
                4,
                5,
                5,
                6,
                4,
                4,
                4,
                6,
                5,
                4,
                5,
                4,
                6,
                3,
                4,
                3,
                4,
                4,
                4,
                4,
                5,
                6,
                5,
                3,
                3,
                4,
                4,
                4,
                3,
                4,
                6,
                4,
                6,
            ]
        ],
        dtype=int,
    )
    return ndihe_gold


def test_dunbrack_score_setup(ubq_system, default_database, torch_device):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    dun_params = dunbrack_graph.dun_resolve_indices
    ndihe_gold = expected_ndihe_from_test_dunbrack_score_setup()
    numpy.testing.assert_array_equal(ndihe_gold, dun_params.ndihe_for_res.cpu().numpy())


def test_dunbrack_score(ubq_system, torch_device, default_database):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )
    intra_graph = dunbrack_graph.intra_score()
    e_dun_tot = intra_graph.dun_score
    e_dun_gold = torch.Tensor([[70.6497, 240.3100, 99.6609]])
    torch.testing.assert_allclose(e_dun_gold, e_dun_tot.cpu())


def test_dunbrack_w_twoubq_stacks(ubq_system, torch_device, default_database):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    dunbrack_graph = CartDunbrackGraph.build_for(
        twoubq, device=torch_device, parameter_database=default_database
    )
    intra_graph = dunbrack_graph.intra_score()
    e_dun_tot = intra_graph.dun_score
    e_dun_gold = torch.Tensor(
        [[70.6497, 240.3100, 99.6609], [70.6497, 240.3100, 99.6609]]
    )
    torch.testing.assert_allclose(e_dun_gold, e_dun_tot.cpu())


def test_setup_params_for_jagged_system(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = CartDunbrackGraph.build_for(ubq40, device=torch_device)
    score60 = CartDunbrackGraph.build_for(ubq60, device=torch_device)
    score_both = CartDunbrackGraph.build_for(twoubq, device=torch_device)

    params40 = score40.dun_resolve_indices
    params60 = score60.dun_resolve_indices
    params_both = score_both.dun_resolve_indices

    for i, params in enumerate([params40, params60]):

        torch.testing.assert_allclose(
            params_both.ndihe_for_res[i, : params.ndihe_for_res.shape[1]],
            params.ndihe_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.dihedral_offset_for_res[
                i, : params.dihedral_offset_for_res.shape[1]
            ],
            params.dihedral_offset_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.dihedral_atom_inds[i, : params.dihedral_atom_inds.shape[1]],
            params.dihedral_atom_inds[0],
        )
        torch.testing.assert_allclose(
            params_both.rottable_set_for_res[i, : params.rottable_set_for_res.shape[1]],
            params.rottable_set_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.nchi_for_res[i, : params.nchi_for_res.shape[1]],
            params.nchi_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.nrotameric_chi_for_res[
                i, : params.nrotameric_chi_for_res.shape[1]
            ],
            params.nrotameric_chi_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.rotres2resid[i, : params.rotres2resid.shape[1]],
            params.rotres2resid[0],
        )
        torch.testing.assert_allclose(
            params_both.prob_table_offset_for_rotresidue[
                i, : params.prob_table_offset_for_rotresidue.shape[1]
            ],
            params.prob_table_offset_for_rotresidue[0],
        )
        torch.testing.assert_allclose(
            params_both.rotmean_table_offset_for_residue[
                i, : params.rotmean_table_offset_for_residue.shape[1]
            ],
            params.rotmean_table_offset_for_residue[0],
        )
        torch.testing.assert_allclose(
            params_both.rotind2tableind_offset_for_res[
                i, : params.rotind2tableind_offset_for_res.shape[1]
            ],
            params.rotind2tableind_offset_for_res[0],
        )
        torch.testing.assert_allclose(
            params_both.rotameric_chi_desc[i, : params.rotameric_chi_desc.shape[1]],
            params.rotameric_chi_desc[0],
        )
        torch.testing.assert_allclose(
            params_both.semirotameric_chi_desc[
                i, : params.semirotameric_chi_desc.shape[1]
            ],
            params.semirotameric_chi_desc[0],
        )


def test_jagged_scoring(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = CartDunbrackGraph.build_for(ubq40, device=torch_device)
    score60 = CartDunbrackGraph.build_for(ubq60, device=torch_device)
    score_both = CartDunbrackGraph.build_for(twoubq, device=torch_device)

    total40 = score40.intra_score().dun_score
    total60 = score60.intra_score().dun_score
    total_both = score_both.intra_score().dun_score

    torch.testing.assert_allclose(total_both[0], total40[0])
    torch.testing.assert_allclose(total_both[1], total60[0])


def test_jagged_scoring2(ubq_res, default_database, torch_device):
    ubq1050 = PackedResidueSystem.from_residues(ubq_res[10:50])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    threeubq = PackedResidueSystemStack((ubq1050, ubq60, ubq40))

    score1050 = CartDunbrackGraph.build_for(ubq1050, device=torch_device)
    score40 = CartDunbrackGraph.build_for(ubq40, device=torch_device)
    score60 = CartDunbrackGraph.build_for(ubq60, device=torch_device)
    score_all = CartDunbrackGraph.build_for(threeubq, device=torch_device)

    total1050 = score1050.intra_score().dun_score
    total60 = score60.intra_score().dun_score
    total40 = score40.intra_score().dun_score
    total_all = score_all.intra_score().dun_score

    torch.testing.assert_allclose(total_all[0], total1050[0])
    torch.testing.assert_allclose(total_all[1], total60[0])
    torch.testing.assert_allclose(total_all[2], total40[0])


def test_cartesian_space_dun_gradcheck(ubq_res, torch_device):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = CartDunbrackGraph.build_for(test_system, device=torch_device)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords
        real_space.coords = state_coords
        return real_space.intra_score().total

    torch.autograd.gradcheck(
        total_score, (start_coords,), eps=2e-3, atol=5e-2, raise_exception=False
    )


# Only run the CPU version of this test, since on the GPU
#     f1s = torch.cross(Xs, Xs - dsc_dx)
# creates non-zero f1s even when dsc_dx is zero everywhere
def test_kinematic_space_dun_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    torsion_space = KinematicDunbrackGraph.build_for(test_system)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.intra_score().total

    # x = total_score(start_dofs)

    assert torch.autograd.gradcheck(total_score, (start_dofs,), eps=2e-3, atol=5e-2)
