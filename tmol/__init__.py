def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
