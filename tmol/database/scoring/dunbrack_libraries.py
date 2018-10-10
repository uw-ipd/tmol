import attr
import cattr
import json
from typing import Tuple


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerTable:
    rotamer_index: int
    probabilities: Tuple  # n-bb-dimensional table
    means: Tuple  # (n-bb+1)-dimensional table
    stdevs: Tuple  # (n-bb+1)-dimensional table


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    nchi: int
    rotamers: Tuple[Tuple[int, ...], ...]  # nrotamers x nchi
    rotamer_tables: Tuple[DunbrackRotamerTable, ...]
    prob_sorted_rot_ind: Tuple  # (n-bb+1)-dimensional table
    backbone_dihedral_start: Tuple[float]
    backbone_dihedral_step: Tuple[float]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericChiTable:
    rotameric_rot_index: int
    probabilities: Tuple  # (n-bb+1)-dimensional table


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericRotamerDefinition:
    rotamer: Tuple[int, ...]
    left: float
    right: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_periodicity: float  # 180 or 360, e.g.
    rotameric_chi_rotamers: Tuple[Tuple[int, ...]]
    nonrotameric_chi_probabilities: Tuple[SemiRotamericChiTable, ...]
    rotamer_definitions: Tuple[SemiRotamericRotamerDefinition, ...]


# ==========================================
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class DunbrackRotamerEntry:
#    rotamer: Tuple[int, ...]
#    prob: float
#    means: Tuple[float, ...]
#    stdev: Tuple[float, ...]
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class MainchainBinRotamerEntries:
#    bb_dihedrals: Tuple[float, ...]
#    sorted_rotamers: Tuple[DunbrackRotamerEntry, ...]
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class AARotamericDunbrackLibrary:
#    aa_name: str
#    mc_dihedral_entries: Tuple[MainchainBinRotamerEntries, ...]
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class SemiRotamericProbabilitiesForChi:
#    chi: float
#    prob: float
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class SemiRotamericRotamerEntry:
#    bb_dihedrals: Tuple[float, ...]
#    chi_probabilities: Tuple[SemiRotamericProbabilitiesForChi, ...]
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class SemiRotamericTableForRotamer:
#    rotamer: Tuple[int, ...]
#    entries: Tuple[SemiRotamericRotamerEntry, ...]
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class SemiRotamericRotamerDefinition:
#    rotamer: Tuple[int, ...]
#    left: float
#    right: float
#
#
# @attr.s(auto_attribs=True, slots=True, frozen=True)
# class AASemiRotamericRotamerLibrary:
#    aa_name: str
#    mc_dihedral_entries: Tuple[MainchainBinRotamerEntries, ...]
#    semi_rotameric_tables: Tuple[SemiRotamericTableForRotamer, ...]
#    rotamer_definitions: Tuple[SemiRotamericRotamerDefinition]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...]
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = json.load(infile)
        return cattr.structure(raw, cls)
