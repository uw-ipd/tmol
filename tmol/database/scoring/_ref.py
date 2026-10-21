from tmol.database._yaml import _load_yaml, _existing_paths


import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: dict[str, float]

    @classmethod
    def from_file(cls, path, generated=()):
        raw = _load_yaml(path)
        for extra in _existing_paths(generated):
            raw["weights"].update(_load_yaml(extra)["weights"])
        return cattr.structure(raw, cls)
