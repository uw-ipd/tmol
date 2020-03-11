import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ConvertAttrs


@attr.s(auto_attribs=True, slots=True)
class SimAParams(TensorGroup):
    raw: Tensor(torch.float)[:, 5]

    @staticmethod
    def new_instance():
        return SimAParams(torch.zeros((1,5), dtype=torch.float))
    
    @property
    def hitemp(self):
        return self.raw[0][0]

    @hitemp.setter
    def hitemp(self, val):
        self.raw[0][0] = val    

    @property
    def lotemp(self):
        return self.raw[0][1]

    @lotemp.setter
    def lotemp(self, val):
        self.raw[0][1] = val    

    @property
    def n_outer(self):
        return self.raw[0][2]

    @n_outer.setter
    def n_outer(self, val):
        self.raw[0][2] = val    

    @property
    def n_inner_scale(self):
        return self.raw[0][3]

    @n_inner_scale.setter
    def n_inner_scale(self, val):
        self.raw[0][3] = val    

    @property
    def quench(self):
        return self.raw[0][4]

    @quench.setter
    def quench(self, val):
        self.raw[0][4] = val    



@attr.s(auto_attribs=True, frozen=True)
class PackerEnergyTables(TensorGroup, ConvertAttrs):
    nrotamers_for_res: Tensor(torch.int32)[:] # [nres]
    oneb_offsets: Tensor(torch.int32)[:] # [nres]
    res_for_rot: Tensor(torch.int32)[:] # [nrotamers_total]
    nenergies: Tensor(torch.int32)[:,:] # [nres x nres]
    twob_offsets: Tensor(torch.int64)[:,:] # [nres x nres]
    energy1b: Tensor(torch.float32)[:, :] # [ncontexts x nrotamers_total]
    energy2b: Tensor(torch.float32)[:] # [ntwob_energies]
    
