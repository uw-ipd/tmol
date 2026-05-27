import pytest
import torch


@pytest.fixture(scope="session")
def tensor_collection():
    from tmol.tests.utility.tensor import _tensor_collection

    return _tensor_collection


def test_tensor_collection(tensor_collection):
    tcoll = [
        torch.arange(4, dtype=torch.float).reshape(2, 2),
        torch.arange(1, 5, dtype=torch.float).reshape(2, 2),
    ]
    tsum = tensor_collection.sum_tensor_collection(tcoll)
    expected = tcoll[0] + tcoll[1]
    torch.testing.assert_close(expected, tsum)
