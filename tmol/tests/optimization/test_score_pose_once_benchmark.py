import pytest

from tmol.tests.torch import zero_padded_counts
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.optimization.sfxn_modules import CartesianSfxnNetwork

from tmol.chemical.restypes import three2one


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.benchmark(group="pose_stack_construct_from_seq_and_score")
def test_pose_construction_from_sequence(
    benchmark,
    n_poses,
    rts_ubq_res,
    default_database,
    fresh_default_packed_block_types,
    torch_device,
):
    n_poses = int(n_poses)

    seq = [three2one(res.residue_type.name3) for res in rts_ubq_res]
    seq[0] = rts_ubq_res[0].residue_type.name3 + ":nterm"
    seq[-1] = rts_ubq_res[-1].residue_type.name3 + ":cterm"
    n_pose_seq = [seq] * n_poses

    ps1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    @benchmark
    def construct_and_get_derivs():
        pose_stack_n = PoseStackBuilder.pose_stack_from_monomer_polymer_sequences(
            fresh_default_packed_block_types, n_pose_seq
        )
        # copy over resident coordinates from some other source
        pose_stack_n.coords[:, :, :] = ps1.coords[None, :, :]

        cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack_n)

        # for param in cart_sfxn_network.parameters():
        #     param.grad.data.zero_()

        E = cart_sfxn_network()
        E.backward()

        return E
