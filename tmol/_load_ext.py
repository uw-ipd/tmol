"""Helpers to choose between precompiled extensions and JIT dev mode."""

from __future__ import annotations

import logging
import os

from tmol._cpp_lib import (
    TmolExtensionIncompatibleError,
    TmolExtensionNotBuiltError,
    _ensure_loaded,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def ensure_compiled_or_jit() -> bool:
    """Return True if JIT should be used; False if precompiled is loaded.

    Behavior:
    - If TMOL_USE_JIT=1 -> returns True (no precompiled load attempt)
    - Else tries to load precompiled _C and returns False if successful
    - If load fails and TMOL_JIT_FALLBACK=1 -> returns True
    - Otherwise re-raises the load error
    """
    if _env_flag("TMOL_USE_JIT"):
        logger.info("tmol: TMOL_USE_JIT=1; using JIT extensions.")
        return True

    try:
        _ensure_loaded()
        return False
    except (TmolExtensionNotBuiltError, TmolExtensionIncompatibleError) as exc:
        if _env_flag("TMOL_JIT_FALLBACK"):
            logger.warning(
                "tmol: precompiled extensions unavailable (%s); falling back to JIT.",
                exc,
            )
            return True
        raise
