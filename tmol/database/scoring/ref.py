import yaml

import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefWeights:
    ALA: float
    ARG: float
    ASN: float
    ASP: float
    CYS: float
    GLN: float
    GLU: float
    GLY: float
    HIS: float
    ILE: float
    LEU: float
    LYS: float
    MET: float
    PHE: float
    PRO: float
    SER: float
    THR: float
    TRP: float
    TYR: float
    VAL: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: RefWeights

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
