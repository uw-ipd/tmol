from .io import ResidueReader
from .packed import PackedResidueSystem
import tmol.database

def read_pdb(pdb_string, **kwargs):
    res = (
        ResidueReader()
        .parse_pdb(pdb_string)
    )


    return PackedResidueSystem(**kwargs).from_residues(res)
