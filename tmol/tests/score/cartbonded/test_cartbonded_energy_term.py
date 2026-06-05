import torch
import attrs

from tmol.database.scoring.cartbonded import CartBondedDatabase
from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert cartbonded_energy.device == torch_device


def test_annotate_twice(fresh_default_restype_set, default_database, torch_device):
    cpu_device = torch.device("cpu")
    cartbonded_energy_cpu = CartBondedEnergyTerm(
        param_db=default_database, device=cpu_device
    )
    cartbonded_energy_device = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    pbt_cpu = PackedBlockTypes.from_restype_list(
        fresh_default_restype_set.chem_db,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        cpu_device,
    )
    pbt_device = PackedBlockTypes.from_restype_list(
        fresh_default_restype_set.chem_db,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    bt_list_cpu = pbt_cpu.active_block_types
    bt_list_device = pbt_device.active_block_types

    for bt in bt_list_cpu:
        cartbonded_energy_cpu.setup_block_type(bt)
    cartbonded_energy_cpu.setup_packed_block_types(pbt_cpu)

    for bt in bt_list_device:
        cartbonded_energy_device.setup_block_type(bt)
    cartbonded_energy_device.setup_packed_block_types(pbt_device)

    assert hasattr(pbt_cpu, "cartbonded_annotations")
    assert hasattr(pbt_device, "cartbonded_annotations")


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

    for bt in bt_list:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pbt)

    assert hasattr(pbt, "cartbonded_annotations")
    assert cartbonded_energy.hash in pbt.cartbonded_annotations
    cb_pbt_ann = pbt.cartbonded_annotations[cartbonded_energy.hash]

    assert cb_pbt_ann.cartbonded_subgraphs.device == torch_device
    assert cb_pbt_ann.cartbonded_subgraph_offsets.device == torch_device
    assert cb_pbt_ann.cartbonded_params_hash_keys.device == torch_device
    assert cb_pbt_ann.cartbonded_params_hash_values.device == torch_device

    cartbonded_subgraphs = cb_pbt_ann.cartbonded_subgraphs
    cartbonded_energy.setup_packed_block_types(pbt)
    assert (
        cartbonded_subgraphs
        is pbt.cartbonded_annotations[cartbonded_energy.hash].cartbonded_subgraphs
    )


def test_hack_cartbonded_params(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    # let's define a new set of cartbonded parameters where we have changed the
    # ideal CA-CB bond length for alanine
    cartb_db = default_database.scoring.cartbonded
    ala_res_params = cartb_db.residue_params["ALA"]
    ind_of_cacb_length_param = next(
        i
        for i, param in enumerate(ala_res_params.length_parameters)
        if param.atm1 == "CA" and param.atm2 == "CB"
    )
    new_ca_cb_length_params = (
        ala_res_params.length_parameters[:ind_of_cacb_length_param]
        + (
            attrs.evolve(
                ala_res_params.length_parameters[ind_of_cacb_length_param], x0=2.0
            ),
        )
        + ala_res_params.length_parameters[ind_of_cacb_length_param + 1 :]
    )
    new_ala_res_params = attrs.evolve(
        ala_res_params, length_parameters=new_ca_cb_length_params
    )
    new_residue_params = dict(cartb_db.residue_params)
    new_residue_params["ALA"] = new_ala_res_params

    new_cartb_db = CartBondedDatabase.from_cartres_dict(new_residue_params)
    new_scoring_db = attrs.evolve(default_database.scoring, cartbonded=new_cartb_db)
    new_database = attrs.evolve(default_database, scoring=new_scoring_db)

    pbt = fresh_default_packed_block_types

    dflt_cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    new_cartbonded_energy = CartBondedEnergyTerm(
        param_db=new_database, device=torch_device
    )
    assert dflt_cartbonded_energy.hash != new_cartbonded_energy.hash

    for bt in pbt.active_block_types:
        dflt_cartbonded_energy.setup_block_type(bt)
        new_cartbonded_energy.setup_block_type(bt)
    dflt_cartbonded_energy.setup_packed_block_types(pbt)
    new_cartbonded_energy.setup_packed_block_types(pbt)

    assert dflt_cartbonded_energy.hash in pbt.cartbonded_annotations
    assert new_cartbonded_energy.hash in pbt.cartbonded_annotations

    ala_bt = next(bt for bt in pbt.active_block_types if bt.name == "ALA")
    dflt_ala_cartb_params = ala_bt.cartbonded_annotations[dflt_cartbonded_energy.hash]
    new_ala_cartb_params = ala_bt.cartbonded_annotations[new_cartbonded_energy.hash]

    dflt_ala_params = dflt_ala_cartb_params.cartbonded_params
    new_ala_params = new_ala_cartb_params.cartbonded_params

    for key in dflt_ala_params:
        dflt_param = dflt_ala_params[key]
        new_param = new_ala_params[key]
        if key == ("UNIQUE_ID:ALA:CA", "UNIQUE_ID:ALA:CB"):
            assert len(dflt_param) == len(new_param)
            for i, (dflt_subparam, new_subparam) in enumerate(
                zip(dflt_param, new_param)
            ):
                if i == 1:
                    assert new_subparam == 2.0
                    assert dflt_subparam != new_subparam
                else:
                    assert dflt_subparam == new_subparam

        else:
            assert dflt_param == new_param


class TestCartBondedEnergyTerm(EnergyTermTestBase):
    energy_term_class = CartBondedEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
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
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums
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
        resnums = [(0, 4)]
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
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums, nondet_tol=1e-5
        )
