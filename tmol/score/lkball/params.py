from typing import Tuple, Sequence
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

from tmol.database.scoring.ljlk import LJLKDatabase

from . import potentials


@attr.s(auto_attribs=True, slots=True, frozen=True)
class WaterBuildingParams:
    max_acc_wat: Tensor("f")[...]
    dists_sp2: Tensor("f")[...]
    angles_sp2: Tensor("f")[...]
    tors_sp2: Tensor("f")[...]
    dists_sp3: Tensor("f")[...]
    angles_sp3: Tensor("f")[...]
    tors_sp3: Tensor("f")[...]
    dists_ring: Tensor("f")[...]
    angles_ring: Tensor("f")[...]
    tors_ring: Tensor("f")[...]
    dist_donor: Tensor("f")[...]
