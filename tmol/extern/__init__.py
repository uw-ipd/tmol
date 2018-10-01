def include_paths():
    """Get the -I compatible include dirs for external modules."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__))]
