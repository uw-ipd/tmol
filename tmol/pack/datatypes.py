import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ConvertAttrs


@attr.s(auto_attribs=True, frozen=True)
class PackerEnergyTables(TensorGroup, ConvertAttrs):
    nrotamers_for_res: Tensor(torch.int32)[:] # [nres]
    oneb_offsets: Tensor(torch.int32)[:] # [nres]
    res_for_rot: Tensor(torch.int32)[:] # [nrotamers_total]
    nenergies: Tensor(torch.int32)[:,:] # [nres x nres]
    twob_offsets: Tensor(torch.int64)[:,:] # [nres x nres]
    energy1b: Tensor(torch.float32)[:] # [nrotamers_total]
    energy2b: Tensor(torch.float32)[:] # [ntwob_energies]
    
