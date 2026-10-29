import os

import yaml

_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _safe_load(stream):
    return yaml.load(stream, Loader=_SAFE_LOADER)


def _load_yaml(path):
    """The document of the yaml file at path."""
    with open(path, "r") as infile:
        return _safe_load(infile)


def _existing_paths(paths):
    """The paths that exist, in order."""
    return [path for path in paths if os.path.exists(path)]
