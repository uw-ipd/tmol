import attr
import torch
from typing import Tuple

from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    rotamers: Tensor[int][:, :]  # nrotamers x nchi
    rotamer_probabilities: Tensor[float]  # (1 + n-bb) dimensional table
    rotamer_means: Tensor[
        float
    ]  # (1 + n-bb + 1) dimensional table: nrots x [nbb] x nchi
    rotamer_stdvs: Tensor[
        float
    ]  # ( 1 + n-bb + 1 ) dimensional table: nrotx x [nbb] x nchi
    prob_sorted_rot_inds: Tensor[int]  # (n-bb+1)-dimensional table
    backbone_dihedral_start: Tensor[float][:]
    backbone_dihedral_step: Tensor[float][:]
    rotamer_alias: Tensor[int][:, :]

    def __repr__(self):
        return "testing __repr__ for RotamericDataForAA"

    def __str__(self):
        return "testing __str__ for RotamericDataForAA"

    def nrotamers(self):
        return self.rotamers.shape[0]

    def nchi(self):
        return self.rotamers.shape[1]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA

    def __repr__(self):
        return "testing __repr__ for RotamericAADunbrackLibrary " + self.table_name


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_period: float  # 180 or 360, e.g.
    rotameric_chi_rotamers: Tensor[int][:, :]  # nrots x n-rotameric-chi
    nonrotameric_chi_probabilities: Tensor[float]  # (1+nbb+1)-dimensional table
    rotamer_boundaries: Tensor[float][:, 2]  # 2nd dimension: 0=left, 1=right

    def __repr__(self):
        return "testing __repr__ for SemiRotamericAADunbrackLibrary " + self.table_name


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunMappingParams:
    dun_table_name: str
    residue_name: str


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    dun_lookup: Tuple[DunMappingParams, ...]
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...]
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...]

    @classmethod
    def from_file(cls, fname: str):
        with torch.serialization.safe_globals(
            [
                DunbrackRotamerLibrary,
                DunMappingParams,
                SemiRotamericAADunbrackLibrary,
                RotamericAADunbrackLibrary,
                RotamericDataForAA,
            ]
        ):
            return torch.load(fname, weights_only=True)
