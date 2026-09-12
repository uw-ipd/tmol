import attr
import torch
from typing import Tuple

from tmol.types import Tensor


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
    # Probability-order cells are selected in the source grid before reflection.
    backbone_is_mirrored: bool = False

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
        _OLD = "tmol.database.scoring.dunbrack_libraries"
        with torch.serialization.safe_globals(
            [
                DunbrackRotamerLibrary,
                (DunbrackRotamerLibrary, f"{_OLD}.DunbrackRotamerLibrary"),
                DunMappingParams,
                (DunMappingParams, f"{_OLD}.DunMappingParams"),
                SemiRotamericAADunbrackLibrary,
                (
                    SemiRotamericAADunbrackLibrary,
                    f"{_OLD}.SemiRotamericAADunbrackLibrary",
                ),
                RotamericAADunbrackLibrary,
                (RotamericAADunbrackLibrary, f"{_OLD}.RotamericAADunbrackLibrary"),
                RotamericDataForAA,
                (RotamericDataForAA, f"{_OLD}.RotamericDataForAA"),
            ]
        ):
            library = torch.load(fname, mmap=True)
        # Older binary records predate this field. These are newly loaded,
        # owned objects; no source file or previously shared record is changed.
        for entry in (*library.rotameric_libraries, *library.semi_rotameric_libraries):
            data = entry.rotameric_data
            if not hasattr(data, "backbone_is_mirrored"):
                object.__setattr__(data, "backbone_is_mirrored", False)
        return library
