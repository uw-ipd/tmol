import logging
import typing
from typing import Mapping, Collection, Optional

import attr
import cattr

from toolz.curried import groupby

from tmol.utility import unique_val
import tmol.io.pdb_parsing as pdb_parsing
import tmol.database.chemical

from tmol.database import ParameterDatabase
from tmol.database.chemical import ChemicalDatabase

from tmol.chemical.restypes import RefinedResidueType, Residue
from .packed import PackedResidueSystem

from tmol.utility.log import ClassLogger
import tmol.io.generic

ResName3 = typing.NewType("ResName3", str)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ResidueReader:
    __default = None

    @classmethod
    def get_default(cls) -> "ResidueReader":
        """Load and return reader over default parameter database."""
        if cls.__default is None:
            cls.__default = cls.from_database(ParameterDatabase.get_default().chemical)
        return cls.__default

    @classmethod
    def from_database(cls, chemical_db: ChemicalDatabase):
        residue_types = groupby(
            lambda restype: restype.name3,
            (
                cattr.structure(cattr.unstructure(r), RefinedResidueType)
                for r in chemical_db.residues
            ),
        )

        return cls(chemical_db=chemical_db, residue_types=residue_types)

    chemical_db: ChemicalDatabase
    residue_types: Mapping[ResName3, RefinedResidueType]

    logger: logging.Logger = ClassLogger

    def resolve_type(
        self, resn: ResName3, atomns: Collection[str]
    ) -> RefinedResidueType:
        """Return the best-match residue type for a collection of atom records."""

        atomns = set(atomns)

        candidate_types = self.residue_types.get(resn, [])
        self.logger.debug(
            f"resolved candidate_types resn: {resn} "
            f"types:{[t.name for t in candidate_types]}"
        )

        if not candidate_types:
            raise ValueError(f"Unknown residue name: {resn}")

        missing_atoms = [
            set(a.name for a in t.atoms).symmetric_difference(atomns)
            for t in candidate_types
        ]

        if len(candidate_types) == 1:
            self.logger.debug("unambiguous residue type")
            best_idx = 0
        else:
            self.logger.debug(
                f"ambiguous residue types: {[t.name for t in candidate_types]}"
            )

            best_idx = min(
                range(len(candidate_types)), key=lambda i: len(missing_atoms[i])
            )

        if missing_atoms[best_idx]:
            self.logger.info(f"missing atoms in input: {missing_atoms[best_idx]}")

        return candidate_types[best_idx]

    def parse_atom_block(self, atoms):
        residue_name = unique_val(atoms.resn)
        atom_coords = atoms.set_index("atomn", verify_integrity=True)[["x", "y", "z"]]
        residue_type = self.resolve_type(residue_name, atom_coords.index)

        res = Residue(residue_type=residue_type)

        for atomn, coord in atom_coords.iterrows():
            if atomn in res.residue_type.atom_to_idx:
                res.atom_coords[atomn] = coord.values
            else:
                self.logger.info(f"unknown atom name: {atomn}")
        return res

    def parse_pdb(self, source_pdb):
        """Resolve list of residue objects from input pdb."""

        atom_records = pdb_parsing.parse_pdb(source_pdb)

        assert atom_records["modeli"].nunique() == 1
        assert atom_records["chaini"].nunique() == 1

        return [
            self.parse_atom_block(atoms)
            for (m, c, resi), atoms in atom_records.groupby(
                ["modeli", "chaini", "resi"]
            )
        ]


def read_pdb(
    pdb_string: str, residue_reader: Optional[ResidueReader] = None
) -> PackedResidueSystem:
    if not residue_reader:
        residue_reader = ResidueReader.get_default()

    res = residue_reader.parse_pdb(pdb_string)

    return PackedResidueSystem.from_residues(res)


@tmol.io.generic.to_cdjson.register(Residue)
def residue_to_cdjson(res: Residue):
    coords = res.coords
    elems = [a.atom_type[0] for a in res.residue_type.atoms]
    bonds = [
        (res.residue_type.atom_to_idx[b], res.residue_type.atom_to_idx[e])
        for b, e in res.residue_type.bonds
    ]

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)


@tmol.io.generic.to_cdjson.register(PackedResidueSystem)
def packed_system_to_cdjson(system: PackedResidueSystem):
    coords = system.coords
    elems = [t[0] if t else "X" for t in system.atom_metadata["atom_type"]]
    bonds = [tuple(b) for b in system.bonds]

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)
