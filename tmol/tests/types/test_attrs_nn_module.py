import attr
import torch


@attr.s(auto_attribs=True, kw_only=True, eq=False)
class BufferModule(torch.nn.Module):

    _module_init: bool = attr.ib(init=False, repr=False)

    @_module_init.default
    def _init_module(self):
        """Call Module.__init__ before other attrs for parameter registration."""
        super().__init__()
        self.register_buffer("a_buffer", None)

    a_buffer: torch.Tensor
    b_tensor: torch.Tensor
    c_parameter: torch.Tensor = attr.ib(converter=torch.nn.Parameter)


def test_buffers():
    a = torch.arange(10, dtype=torch.float)
    b = torch.arange(20, dtype=torch.float)
    c = torch.arange(30, dtype=torch.float)
    bm = BufferModule(a_buffer=a, b_tensor=b, c_parameter=c)

    assert list(dict(bm.named_buffers()).keys()) == ["a_buffer"]
    assert list(dict(bm.named_parameters()).keys()) == ["c_parameter"]

    assert bm.a_buffer.requires_grad is False
    assert bm.b_tensor.requires_grad is False
    assert bm.c_parameter.requires_grad is True

    assert bm.a_buffer.dtype == torch.float
    assert bm.b_tensor.dtype == torch.float
    assert bm.c_parameter.dtype == torch.float

    bm.to(torch.double)

    assert bm.a_buffer.dtype == torch.double
    assert bm.b_tensor.dtype == torch.float
    assert bm.c_parameter.dtype == torch.double
