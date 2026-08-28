from tmol.database._yaml import safe_load

import os

import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: dict[str, float]

    @classmethod
    def from_file(cls, path, generated=()):
        with open(path, "r") as infile:
            raw = safe_load(infile)
        for extra in generated:
            if not os.path.exists(extra):
                continue
            with open(extra, "r") as infile:
                raw["weights"].update(safe_load(infile)["weights"])
        return cattr.structure(raw, cls)
