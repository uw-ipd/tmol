import os
import torch


from tmol.io.pose_stack_from_rosettafold2 import (
    pose_stack_from_rosettafold2,
    canonical_form_from_rosettafold2,
    _paramdb_for_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
)

# from tmol.io.pose_stack_from_rosettafold2 import pose_stack_from_rosettafold2


def test_load_rosettafold2_dictionary(rosettafold_ubq_pred, torch_device):
    # print("rosettafold_ubq_pred")
    # print(rosettafold_ubq_pred)
    ps = pose_stack_from_rosettafold2(**rosettafold_ubq_pred)
    assert len(ps) == 1
    assert ps.max_n_blocks == 76
    pbt = packed_block_types_for_rosettafold2(torch_device)
    assert ps.packed_block_types is pbt


def test_create_canonical_form_from_rosettafold2_stability(
    rosettafold_ubq_pred, torch_device
):
    cf = canonical_form_from_rosettafold2(**rosettafold_ubq_pred)
    gold_cf_path = os.path.join(
        __file__.rpartition("/")[0], "gold_ubq_rosettafold2_canform.pt"
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
