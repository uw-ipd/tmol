import attr
import cattr

import torch

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.functional import validate_args

from tmol.database.scoring.disulfide import DisulfideDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DisulfideGlobalParams(TensorGroup):
    d_location: Tensor[torch.float32][...]
    d_scale: Tensor[torch.float32][...]
    d_shape: Tensor[torch.float32][...]

    a_logA: Tensor[torch.float32][...]
    a_kappa: Tensor[torch.float32][...]
    a_mu: Tensor[torch.float32][...]

    dss_logA1: Tensor[torch.float32][...]
    dss_kappa1: Tensor[torch.float32][...]
    dss_mu1: Tensor[torch.float32][...]
    dss_logA2: Tensor[torch.float32][...]
    dss_kappa2: Tensor[torch.float32][...]
    dss_mu2: Tensor[torch.float32][...]

    dcs_logA1: Tensor[torch.float32][...]
    dcs_mu1: Tensor[torch.float32][...]
    dcs_kappa1: Tensor[torch.float32][...]
    dcs_logA2: Tensor[torch.float32][...]
    dcs_mu2: Tensor[torch.float32][...]
    dcs_kappa2: Tensor[torch.float32][...]
    dcs_logA3: Tensor[torch.float32][...]
    dcs_mu3: Tensor[torch.float32][...]
    dcs_kappa3: Tensor[torch.float32][...]

    wt_dih_ss: Tensor[torch.float32][...]
    wt_dih_cs: Tensor[torch.float32][...]
    wt_ang: Tensor[torch.float32][...]
    wt_len: Tensor[torch.float32][...]
    shift: Tensor[torch.float32][...]

    @classmethod
    @validate_args
    def from_database(cls, disulfide_database: DisulfideDatabase, device: torch.device):

        global_params = DisulfideGlobalParams(
            **{
                n: torch.tensor(v, device=device).expand((1,))
                for n, v in cattr.unstructure(
                    disulfide_database.global_parameters
                ).items()
            }
        )

        return global_params
