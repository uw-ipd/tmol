from tmol.database._yaml import load_yaml, existing_paths


import attr
import cattr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: dict[str, float]

    @classmethod
    def from_file(cls, path, generated=()):
        raw = load_yaml(path)
        for extra in existing_paths(generated):
            raw["weights"].update(load_yaml(extra)["weights"])
        return cattr.structure(raw, cls)
