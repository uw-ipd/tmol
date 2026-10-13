"""JAX-friendly port of ``tmol.database.scoring``.

Exposes one :class:`ScoringDatabase` aggregating every scoring-term
sub-database, plus a :meth:`ScoringDatabase.from_file` classmethod that
reads the full set of YAMLs (and the optional tables binaries) from a
directory.

Individual term databases live in submodules and can also be loaded
and used in isolation:

* :mod:`.ljlk` — Lennard-Jones / Lazaridis-Karplus parameters.
* :mod:`.elec` — electrostatics (sigmoidal dielectric) parameters.
* :mod:`.hbond` — hydrogen-bond donor/acceptor + polynomial parameters.
* :mod:`.cartbonded` — Cartesian-bonded per-residue bond/angle/torsion
  parameters.
* :mod:`.ref` — per-residue reference energies.
* :mod:`.rama` — Ramachandran energy-grid mappings (tables read from
  ``rama.npz``; falls back to ``rama.zip`` via torch if missing).
* :mod:`.omega_bbdep` — backbone-dependent omega mean / sigma tables
  (read from ``omega_bbdep.npz``; torch fallback for the legacy zip).
* :mod:`.dunbrack_libraries` — rotamer libraries (read from
  ``dunbrack.npz``; falls back to ``dunbrack.bin`` via torch).
* :mod:`.disulfide` — disulfide polynomial / gaussian mixture globals.
"""

from __future__ import annotations

import os
from typing import Optional

import attr

from .cartbonded import CartBondedDatabase
from .disulfide import DisulfideDatabase
from .genbonded import GenBondedDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary
from .elec import ElecDatabase
from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .omega_bbdep import OmegaBBDepDatabase
from .rama import RamaDatabase
from .ref import RefDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    """Aggregate scoring-parameter database (one object per term)."""

    ljlk: LJLKDatabase
    elec: ElecDatabase
    hbond: HBondDatabase
    cartbonded: CartBondedDatabase
    genbonded: GenBondedDatabase
    ref: RefDatabase
    rama: RamaDatabase
    dunbrack: DunbrackRotamerLibrary
    disulfide: DisulfideDatabase
    omega_bbdep: OmegaBBDepDatabase

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "ScoringDatabase":
        """Load every scoring sub-database from ``<path>/*.yaml``.

        ``path`` defaults to the package's bundled
        ``default/scoring/`` directory. Binary tables are read from
        ``rama.npz`` / ``omega_bbdep.npz`` / ``dunbrack.npz`` if
        present (no torch needed); the loaders fall back to the
        legacy ``rama.zip`` / ``omega_bbdep.zip`` / ``dunbrack.bin``
        via ``torch.load`` only if the ``.npz`` is missing.
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "default", "scoring")
        path = os.path.abspath(path)

        return cls(
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            elec=ElecDatabase.from_file(os.path.join(path, "elec.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            cartbonded=CartBondedDatabase.from_file(
                os.path.join(path, "cartbonded.yaml")
            ),
            genbonded=GenBondedDatabase.from_file(os.path.join(path, "genbonded.yaml")),
            ref=RefDatabase.from_file(os.path.join(path, "ref.yaml")),
            rama=RamaDatabase.from_file(os.path.join(path, "rama.yaml")),
            dunbrack=DunbrackRotamerLibrary.from_file(
                os.path.join(path, "dunbrack.yaml")
            ),
            disulfide=DisulfideDatabase.from_file(os.path.join(path, "disulfide.yaml")),
            omega_bbdep=OmegaBBDepDatabase.from_file(
                os.path.join(path, "omega_bbdep.yaml")
            ),
        )


__all__ = [
    "ScoringDatabase",
    "CartBondedDatabase",
    "GenBondedDatabase",
    "DisulfideDatabase",
    "DunbrackRotamerLibrary",
    "ElecDatabase",
    "HBondDatabase",
    "LJLKDatabase",
    "OmegaBBDepDatabase",
    "RamaDatabase",
    "RefDatabase",
]
