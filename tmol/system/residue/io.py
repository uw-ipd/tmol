import cattr
import properties
from tmol.properties.reactive import cached

from toolz.curried import groupby

from tmol.utility import unique_val
import tmol.io.pdb_parsing as pdb_parsing
import tmol.database.chemical
from tmol.database.chemical import ChemicalDatabase

from .restypes import ResidueType, Residue
from .packed import PackedResidueSystem

from tmol.utility.log import LoggerMixin
import tmol.io.generic


class ResidueReader(properties.HasProperties, LoggerMixin):
    chemical_db: ChemicalDatabase = properties.Instance(
        "source chemical db",
        ChemicalDatabase,
        default=tmol.database.default.chemical,
    )

    @cached(properties.Dictionary("residue types from db by 3-letter code"))
    def residue_types(self):
        return groupby(
            lambda restype: restype.name3,
            (
                cattr.structure(cattr.unstructure(r), ResidueType)
                for r in self.chemical_db.residues
            )
        )

    def resolve_type(self, resn, atomns):
        atomns = set(atomns)

        candidate_types = self.residue_types.get(resn, [])
        self.logger.debug(
            f"resolved candidate_types resn: {resn} types:{[t.name for t in candidate_types]}"
        )

        if not candidate_types:
            raise ValueError(f"Unknown residue name: {resn}")

        missing_atoms = [
            set(a.name
                for a in t.atoms).difference(atomns)
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
                range(len(candidate_types)),
                key=lambda i: len(missing_atoms[i])
            )

        if missing_atoms[best_idx]:
            self.logger.info(
                f"missing atoms in input: {missing_atoms[best_idx]}"
            )

        return candidate_types[best_idx]

    def parse_atom_block(self, atoms):
        residue_name = unique_val(atoms.resn)
        atom_coords = atoms.set_index(
            "atomn", verify_integrity=True
        )[["x", "y", "z"]]
        residue_type = self.resolve_type(residue_name, atom_coords.index)

        res = Residue(residue_type=residue_type)

        for atomn, coord in atom_coords.iterrows():
            if atomn in res.residue_type.atom_to_idx:
                res.atom_coords[atomn] = coord.values
            else:
                self.logger.info(f"unknown atom name: {atomn!r}")
        return res

    def parse_pdb(self, source_pdb):
        """Resolve list of residue objects from input pdb."""

        atom_records = pdb_parsing.parse_pdb(source_pdb)

        assert atom_records["modeli"].nunique() == 1
        assert atom_records["chaini"].nunique() == 1

        return [
            self.parse_atom_block(atoms)
            for (m, c, resi), atoms
            in atom_records.groupby(["modeli", "chaini", "resi"])
        ]  # yapf: disable


def read_pdb(pdb_string: str) -> PackedResidueSystem:
    res = ResidueReader().parse_pdb(pdb_string)

    return PackedResidueSystem.from_residues(res)


@tmol.io.generic.to_cdjson.register(Residue)
def residue_to_cdjson(res):
    coords = res.coords
    elems = [a.atom_type[0] for a in res.residue_type.atoms]
    bonds = [
        (res.residue_type.atom_to_idx[b], res.residue_type.atom_to_idx[e])
        for b, e in res.residue_type.bonds
    ]

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)


@tmol.io.generic.to_cdjson.register(PackedResidueSystem)
def packed_system_to_cdjson(system):
    coords = system.coords
    elems = [t[0] if t else "x" for t in system.atom_types]
    bonds = list(map(tuple, system.bonds))

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)
