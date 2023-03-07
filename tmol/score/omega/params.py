import attr
import cattr

import pandas

import torch
import numpy


from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.omega import OmegaDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OmegaGlobalParams(TensorGroup):
    K: Tensor[torch.float32][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class OmegaParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    # shape [1] global parameters
    global_params: OmegaGlobalParams

    device: torch.device

    @classmethod
    @validate_args
    def from_database(cls, omega_database: OmegaDatabase, device: torch.device):
        """Initialize param resolver for all atom types in database."""
        return cls.from_param_resolver(omega_database, device)

    @classmethod
    @validate_args
    def from_param_resolver(cls, omega_database: OmegaDatabase, device: torch.device):

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

        return cls(global_params=global_params, device=device)
