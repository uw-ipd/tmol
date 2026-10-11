"""Reference-energy parameter database.

Thin wrapper around a ``{residue_name: weight}`` mapping. No numeric
arrays are needed until score time — the loader only needs ``yaml``.
"""

from __future__ import annotations

from typing import Dict

import attr
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RefDatabase:
    weights: Dict[str, float]

    @classmethod
    def from_file(cls, path: str) -> "RefDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return structure(raw, cls)


__all__ = ["RefDatabase"]
