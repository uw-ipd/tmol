from tmol.io.canonical_ordering import (
    default_packed_block_types,
)


def test_default_packed_block_types(torch_device):
    pbt = default_packed_block_types(torch_device)
    assert pbt.device == torch_device
    assert pbt.n_types >= 86


def test_default_packed_block_types_memoization(torch_device):
    # this function should return the same PackedBlockTypes object each time it's called
    pbt1 = default_packed_block_types(torch_device)

    pbt2 = default_packed_block_types(torch_device)
    assert pbt1 is pbt2
