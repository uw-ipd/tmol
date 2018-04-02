import toolz
import properties
from tmol.properties.reactive import cached

from toolz.curried import groupby

from tmol.utility import unique_val, just_one
import tmol.io.pdb_parsing as pdb_parsing
import tmol.database.chemical
from tmol.database.chemical import ChemicalDatabase

from .restypes import AtomType, ResidueType
from .packed import Residue

from tmol.utility.log import LoggerMixin

def residue_type_from_database(tbls):
    """Create residue type from database tables."""
    name = just_one(tbls["NAME"]["name"].values)
    aa = just_one(tbls["AA"]["name"].values)

    atoms = tuple(
        AtomType(fq_name = f"{name}.{atom['name']}", **atom)
        for atom in tbls["ATOM"].to_dict("records")
    )

    bonds = tuple(map(tuple, tbls["BOND"][["atom_a", "atom_b"]].values))

    lower_connect = just_one(tbls["LOWER_CONNECT"]["name"].values)
    upper_connect = just_one(tbls["UPPER_CONNECT"]["name"].values)

    return ResidueType(
        name=name,
        aa=aa,
        atoms=atoms,
        bonds=bonds,
        lower_connect=lower_connect,
        upper_connect=upper_connect
    )


class ResidueReader(properties.HasProperties, LoggerMixin):
    chemical_db : ChemicalDatabase = properties.Instance(
        "source chemical db",
        ChemicalDatabase,
        default = tmol.database.default.chemical,
    )

    @cached(properties.Dictionary("residue types from db by 3-letter code"))
    def residue_types(self):
        return groupby(
            lambda restype: restype.name3,
            [ResidueType(**r) for r in self.chemical_db.parameters["residues"]]
        )


    def resolve_type(self, resn, atomns):
        atomns = set(atomns)
        
        candidate_types = self.residue_types.get(resn, [])
        self.logger.debug(
            f"resolved candidate_types resn: {resn} types:{[t.name for t in candidate_types]}")
        
        if not candidate_types:
            raise ValueError(f"Unknown residue name: {resn}")
        missing_atoms = [
            set(a.name for a in t.atoms).difference(atomns)
            for t in candidate_types
        ]
        
        if len(candidate_types) == 1:
            self.logger.debug("unambiguous residue type")
            best_idx = 0
        else:
            self.logger.debug(f"ambiguous residue types: {[t.name for t in candidate_types]}")
            best_idx = min(range(len(candidate_types)), key = lambda i: len(missing_atoms[i]))

        if missing_atoms[best_idx]:
            self.logger.info(f"missing atoms in input: {missing_atoms[best_idx]}")
        return candidate_types[best_idx]
    
    def parse_atom_block(self, atoms):
        residue_name = unique_val(atoms.resn)
        atom_coords = atoms.set_index("atomn", verify_integrity=True)[["x", "y", "z"]]
        residue_type = self.resolve_type(residue_name, atom_coords.index)
        
        res = Residue(residue_type = residue_type)

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
            for (m, c, resi), atoms 
            in atom_records.groupby(["modeli", "chaini", "resi"])
        ]
