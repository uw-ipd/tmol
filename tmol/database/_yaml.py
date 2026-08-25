import yaml

_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def safe_load(stream):
    return yaml.load(stream, Loader=_SAFE_LOADER)
