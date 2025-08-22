import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    assert hbond_energy.device == torch_device
    assert hbond_energy.hb_param_db.global_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_poly_table.device == torch_device


def test_hbond_in_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    assert len(sfxn.all_terms()) == 1
    assert isinstance(sfxn.all_terms()[0], HBondEnergyTerm)


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    for rt in pbt.active_block_types:
        hbond_energy.setup_block_type(rt)
        assert hasattr(rt, "hbbt_params")
    hbond_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "hbpbt_params")

    assert pbt.hbpbt_params.tile_n_donH.device == torch_device
    assert pbt.hbpbt_params.tile_n_acc.device == torch_device
    assert pbt.hbpbt_params.tile_donH_inds.device == torch_device
    assert pbt.hbpbt_params.tile_acc_inds.device == torch_device
    assert pbt.hbpbt_params.tile_donorH_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_hybridization.device == torch_device
    assert pbt.hbpbt_params.is_hydrogen.device == torch_device


def test_whole_pose_scoring_module_smoke(ubq_pdb, default_database, torch_device):
    gold_vals = numpy.array([[-55.6756]], dtype=numpy.float32)
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(p1.packed_block_types)
    hbond_energy.setup_poses(p1)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = hbond_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_rotamer_scoring(
    ubq_repacking_rotamers,
    default_database,
    torch_device,
):
    n_poses = 1

    print("pass")
    return

    p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

    write_pose_stack_pdb(
        pn,
        "test_whole_pose_scoring_10_new.pdb",
        chain_ind_for_block=torch.zeros(
            (pn.n_poses, pn.max_n_blocks), dtype=torch.int64
        ),
    )

    if edit_pose_stack_fn is not None:
        edit_pose_stack_fn(pn)

    pose_scorer = cls.get_pose_scorer(pn, default_database, torch_device)

    nres = pn.block_coord_offset.size(1)

    block_pair_dispatch_indices = cls.get_block_pair_dispatch_indices(
        nres, device=torch_device
    )

    coords = torch.nn.Parameter(pn.coords.clone())
    scores, indices = pose_scorer(
        coords, block_pair_dispatch_indices, output_block_pair_energies=True
    )
    # .cpu()
    # .detach()
    # .numpy()

    """print(scores.shape)
    print(scores_t[torch.nonzero(scores_t)])
    print(torch.sum(scores_t))"""

    sparse = torch.sparse_coo_tensor(indices, scores)
    print("sparse", sparse.to_dense()[0, 0:6, 0:6])
    print("SHAPE: ", scores.shape, indices.shape)
    sparse_csr = sparse.to_dense().to_sparse_csr()
    print(sparse_csr)
    print("TOTAL: ", torch.sum(sparse))
    torchshow.show(sparse.to_dense())

    if update_baseline:
        cls.save_test_baseline_data(
            cls.test_whole_pose_scoring_10.__name__, cls.whole_pose_to_dict(scores)
        )
    gold_vals = cls.get_test_baseline_data(cls.test_whole_pose_scoring_10.__name__)

    assert_allclose(gold_vals, scores, atol, rtol)


class TestHBondEnergyTerm(EnergyTermTestBase):
    energy_term_class = HBondEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        resnums = [(6, 8), (10, 12)]
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        ubq_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(6, 8), (10, 12)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(6, 8), (10, 12)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(6, 8), (10, 12)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )
