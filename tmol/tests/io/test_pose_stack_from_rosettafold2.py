import os
import torch

from tmol.tests.score.common.test_energy_term import assert_allclose

from tmol.io.pose_stack_from_rosettafold2 import (
    pose_stack_from_rosettafold2,
    pose_stack_to_rosettafold2,
    canonical_form_from_rosettafold2,
    _paramdb_for_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
)

# from tmol.io.pose_stack_from_rosettafold2 import pose_stack_from_rosettafold2


def test_load_rosettafold2_dictionary(rosettafold2_ubq_pred, torch_device):
    # print("rosettafold_ubq_pred")
    # print(rosettafold_ubq_pred)
    ps = pose_stack_from_rosettafold2(**rosettafold2_ubq_pred)
    assert len(ps) == 1
    assert ps.max_n_blocks == 76
    pbt = packed_block_types_for_rosettafold2(torch_device)
    assert ps.packed_block_types is pbt


def test_load_rosettafold2_dictionary2(rosettafold2_sumo_pred, torch_device):
    # print("rosettafold_ubq_pred")
    # print(rosettafold_ubq_pred)
    ps = pose_stack_from_rosettafold2(**rosettafold2_sumo_pred)
    assert len(ps) == 1
    assert ps.max_n_blocks == 81
    pbt = packed_block_types_for_rosettafold2(torch_device)
    assert ps.packed_block_types is pbt


def test_multi_chain_rosettafold2_pose_stack_construction(
    rosettafold2_ubq_pred, torch_device
):
    """Just fake a multi-chain prediction by saying ubq is two chains"""
    # print("rosettafold_ubq_pred")
    # print(rosettafold2_ubq_pred)
    rosettafold2_ubq_pred["chainlens"] = [30, 46]

    ps = pose_stack_from_rosettafold2(**rosettafold2_ubq_pred)
    assert len(ps) == 1
    assert ps.max_n_blocks == 76
    pbt = packed_block_types_for_rosettafold2(torch_device)
    assert ps.packed_block_types is pbt


def test_from_to_rosettafold2(rosettafold2_ubq_pred, torch_device):
    rosettafold2_ubq_pred["chainlens"] = [76]

    # RF2->tmol
    ps = pose_stack_from_rosettafold2(**rosettafold2_ubq_pred)

    # tmol->RF2
    rf2ubq, rf2_ats = pose_stack_to_rosettafold2(ps, rosettafold2_ubq_pred["chainlens"])

    assert_allclose(
        rosettafold2_ubq_pred["xyz"].unsqueeze(0)[rf2_ats], rf2ubq[rf2_ats], 1e-5, 1e-3
    )


def test_create_canonical_form_from_rosettafold2_ubq_stability(
    rosettafold2_ubq_pred, torch_device
):
    cf = canonical_form_from_rosettafold2(**rosettafold2_ubq_pred)
    gold_cf_path = os.path.join(
        __file__.rpartition("/")[0], "gold_ubq_rosettafold2_canform.pt"
    )
    # if torch_device == torch.device("cpu"):
    #    torch.save(cf, gold_cf_path)
    gold_cf = torch.load(gold_cf_path)
    for n, t in gold_cf.items():
        assert n in cf
        torch.testing.assert_close(t, cf[n].cpu(), equal_nan=True, atol=1e-5, rtol=1e-5)


def test_create_canonical_form_from_rosettafold2_sumo_stability(
    rosettafold2_sumo_pred, torch_device
):
    cf = canonical_form_from_rosettafold2(**rosettafold2_sumo_pred)
    gold_cf_path = os.path.join(
        __file__.rpartition("/")[0], "gold_sumo_rosettafold2_canform.pt"
    )
    # if torch_device == torch.device("cpu"):
    #    torch.save(cf, gold_cf_path)
    gold_cf = torch.load(gold_cf_path)
    for n, t in gold_cf.items():
        assert n in cf
        torch.testing.assert_close(t, cf[n].cpu(), equal_nan=True, atol=1e-5, rtol=1e-5)


def test_memoization_of_rosettafold2_paramdb():
    paramdb1 = _paramdb_for_rosettafold2()
    paramdb2 = _paramdb_for_rosettafold2()
    assert paramdb1 is paramdb2


def test_memoization_of_canonical_ordering():
    co1 = canonical_ordering_for_rosettafold2()
    co2 = canonical_ordering_for_rosettafold2()
    assert co1 is co2


def test_memoization_of_packed_block_types_for_rosettafold2(torch_device):
    pbt1 = packed_block_types_for_rosettafold2(torch_device)
    pbt2 = packed_block_types_for_rosettafold2(torch_device)
    assert pbt1 is pbt2


def test_device_of_packed_block_types(torch_device):
    pbt = packed_block_types_for_rosettafold2(torch_device)
    assert pbt.device == torch_device
