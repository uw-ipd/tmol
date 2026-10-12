from tmol.database._yaml import load_yaml

import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScatteringParams:
    weight: float
    sigma: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecDensDatabase:
    # table name ("xray", "electron") -> element name -> parameters
    scatterers: dict[str, dict[str, ScatteringParams]]

    @classmethod
    def from_file(cls, path):
        return cattr.structure(load_yaml(path), cls)
