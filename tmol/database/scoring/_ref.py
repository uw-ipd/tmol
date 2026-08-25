import yaml

import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: dict[str, float]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
