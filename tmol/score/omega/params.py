import attr
import cattr

import torch

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.omega import OmegaDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OmegaGlobalParams(TensorGroup):
    K: Tensor[torch.float32][...]

    @classmethod
    @validate_args
    def from_database(cls, omega_database: OmegaDatabase, device: torch.device):
        # Convert float entries into 1-d tensors
        def at_least_1d(t):
            if t.dim() == 0:
                return t.expand((1,))
            else:
                return t

        global_params = OmegaGlobalParams(
            **{
                n: at_least_1d(torch.tensor(v, device=device))
                for n, v in cattr.unstructure(omega_database.global_parameters).items()
            }
        )

        return global_params
