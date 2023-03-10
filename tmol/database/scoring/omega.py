import yaml

import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OmegaGlobalParameters:
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OmegaDatabase:
    global_parameters: OmegaGlobalParameters

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
