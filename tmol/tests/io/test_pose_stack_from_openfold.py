from tmol.io.pose_stack_from_openfold import (
    pose_stack_from_openfold,
    _paramdb_for_openfold,
    canonical_ordering_for_openfold,
    packed_block_types_for_openfold,
)


def test_create_pose_stack_from_openfold_result(
    openfold_ubq_and_sumo_pred, torch_device
):
    ps = pose_stack_from_openfold(openfold_ubq_and_sumo_pred)
    assert len(ps) == 2
    assert ps.max_n_blocks == openfold_ubq_and_sumo_pred["positions"].shape[2]
    assert ps.coords.device == torch_device
    pbt = packed_block_types_for_openfold(torch_device)
    assert ps.packed_block_types is pbt


def test_memoization_of_openfold_paramdb():
    paramdb1 = _paramdb_for_openfold()
    paramdb2 = _paramdb_for_openfold()
    assert paramdb1 is paramdb2


def test_memoization_of_canonical_ordering():
    co1 = canonical_ordering_for_openfold()
    co2 = canonical_ordering_for_openfold()
    assert co1 is co2


def test_memoization_of_packed_block_types_for_openfold(torch_device):
    pbt1 = packed_block_types_for_openfold(torch_device)
    pbt2 = packed_block_types_for_openfold(torch_device)
    assert pbt1 is pbt2


def test_device_of_packed_block_types(torch_device):
    pbt = packed_block_types_for_openfold(torch_device)
    assert pbt.device == torch_device
