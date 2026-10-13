"""Keep the optional JAX suite out of Torch-only test environments."""

from importlib.util import find_spec

collect_ignore_glob = ["test_*.py"] if find_spec("jax") is None else []
