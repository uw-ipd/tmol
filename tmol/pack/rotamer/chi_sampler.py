import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.scoring.dunbrack.params import (
    SamplingDunbrackDatabaseView,
    DunbrackParamResolver,
)


@attr.s(auto_attribs=True)
class ChiSampler:
    sampling_params: SamplingDunbrackDatabaseView

    @classmethod
    @validate_args
    def from_database(cls, param_resolver: DunbrackParamResolver):
        return cls(sampling_params=param_resolver.sampling_db)

    # @validate_args
    # def chi_samples_for_residues(
    #
    # ) -> Tensor(torch.float32)[:,:] :
