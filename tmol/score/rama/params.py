import attr
import cattr

import numpy
import pandas
import torch

from typing import List

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.rama import RamaDatabase

# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    probs: List[Tensor(torch.float)[...]]
    bbsteps: List[Tensor(torch.float)[...]]
    bbstarts: List[Tensor(torch.float)[...]]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RamaParamResolver(ValidateAttrs):
    rama_indices: pandas.Index
    rama_params: PackedRamaDatabase

    device: torch.device

    def resolve_ramatables(
        self,
        resnames1: NDArray(object),
        atm1s: NDArray(object),
        atm2s: NDArray(object),
        atm3s: NDArray(object),
        atm4s: NDArray(object),
    ) -> NDArray("i8")[...]:
        return i

    @classmethod
    @validate_args
    def from_database(cls, rama_database: RamaDatabase, device: torch.device):
        rama_records = pandas.DataFrame.from_records(
            cattr.unstructure(rama_database.rama_lookup)
        )
        rama_index = pandas.Index(length_records[["res", "atm1", "atm2"]])
        bondlength_params = PackedRamaDatabase()

        return cls(rama_indices=rama_indices, rama_params=rama_params, device=device)
