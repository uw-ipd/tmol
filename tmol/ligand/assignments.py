"""Shared atom-type assignment record for ligand preparation."""

from typing import NamedTuple


class AtomTypeAssignment(NamedTuple):
    atom_name: str
    atom_type: str
    element: str
    index: int
