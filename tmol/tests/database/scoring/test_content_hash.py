import torch

from tmol.database.scoring._content_hash import content_hash


def test_grid_metadata_boundaries_are_unambiguous():
    assert content_hash((1.0, 23.0)) != content_hash((1.02, 3.0))
    assert content_hash((1, (2, 3))) != content_hash(((1, 2), 3))


def test_tensor_values_shape_and_dtype_contribute():
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert content_hash(original) == content_hash(original.clone())
    assert content_hash(original) == content_hash(original.t().contiguous().t())
    assert content_hash(original) != content_hash(original + 1)
    assert content_hash(original) != content_hash(original.reshape(4, 3))
    assert content_hash(original) != content_hash(original.double())
    assert content_hash(torch.empty((0, 3))) != content_hash(torch.empty((0, 4)))
