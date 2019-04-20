import torch
from tmol.utility.cpp_extension import load, relpaths, modulename


def test_load():
    load(
        modulename(f"{__name__}"),
        relpaths(__file__, "custom_op.cpp"),
        is_python_module=False,
    )

    cpow = torch.ops.tmol.CPow
    assert cpow is not None

    @torch.jit.script
    def pow3(x):
        return cpow(x, 3.0)

    i = torch.arange(10).to(torch.float)
    res = cpow(i, 3.0)
    jres = pow3(i)
    eres = i.pow(3.0)

    assert not res.requires_grad
    assert not jres.requires_grad
    assert not eres.requires_grad

    torch.testing.assert_allclose(eres, res)
    torch.testing.assert_allclose(eres, jres)

    gi = torch.arange(10).to(torch.float).requires_grad_(True)
    gres = cpow(gi, 3.0)
    gjres = pow3(gi)
    geres = gi.pow(3.0)

    torch.testing.assert_allclose(geres, gres)
    torch.testing.assert_allclose(geres, gjres)

    assert gres.requires_grad
    assert gjres.requires_grad
    assert geres.requires_grad
