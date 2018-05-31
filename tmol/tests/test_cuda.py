from .torch import requires_cuda


@requires_cuda
def test_torch_cuda_is_available():
    import torch
    assert torch.cuda.is_available()


@requires_cuda
def test_torch_cuda_smoke():
    import torch

    rs = (100, 100)
    a = torch.rand(rs)
    b = torch.rand(rs)

    c = a.cuda() @ b.cuda()

    torch.testing.assert_allclose(a @ b, c.cpu())
