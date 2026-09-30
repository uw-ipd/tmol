import os

import yaml

_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def safe_load(stream):
    return yaml.load(stream, Loader=_SAFE_LOADER)


def load_yaml(path):
    """The document of the yaml file at path."""
    with open(path, "r") as infile:
        return safe_load(infile)


def existing_paths(paths):
    """The paths that exist, in order."""
    return [path for path in paths if os.path.exists(path)]
