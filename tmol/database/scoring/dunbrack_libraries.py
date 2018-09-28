import attr
import cattr
import json
from typing import Tuple


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerEntry:
    rotamer: Tuple[int, ...]
    prob: float
    means: Tuple[float, ...]
    stdev: Tuple[float, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MainchainBinRotamerEntries:
    bb_dihedrals: Tuple[float, ...]
    sorted_rotamers: Tuple[DunbrackRotamerEntry, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AARotamericDunbrackLibrary:
    aa_name: str
    mc_dihedral_entries: Tuple[MainchainBinRotamerEntries, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericProbabilitiesForChi:
    chi: float
    prob: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericRotamerEntry:
    bb_dihedrals: Tuple[float, ...]
    chi_probabilities: Tuple[SemiRotamericProbabilitiesForChi, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericTableForRotamer:
    rotamer: Tuple[int, ...]
    entries: Tuple[SemiRotamericRotamerEntry, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericRotamerDefinition:
    rotamer: Tuple[int, ...]
    left: float
    right: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AASemiRotamericRotamerLibrary:
    aa_name: str
    mc_dihedral_entries: Tuple[MainchainBinRotamerEntries, ...]
    semi_rotameric_tables: Tuple[SemiRotamericTableForRotamer, ...]
    rotamer_definitions: Tuple[SemiRotamericRotamerDefinition]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    rotameric_libraries: Tuple[AARotamericDunbrackLibrary, ...]
    semi_rotameric_libraries: Tuple[AASemiRotamericRotamerLibrary, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = json.load(infile)
        return cattr.structure(raw, cls)
