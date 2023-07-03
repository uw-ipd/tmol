import cattr
import torch

from tmol.database import ParameterDatabase
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.chemical.restypes import RefinedResidueType
import toolz.functoolz


@toolz.functoolz.memoize
def default_canonical_packed_block_types(device: torch.device):
    chem_database = ParameterDatabase.get_default().chemical
    atom_type_resolver = AtomTypeParamResolver.from_database(
        chem_database, torch.device("cpu")
    )

    # for now, nab the "mid" residue types
    # TODO: add N- and C-term variants
    wanted = [
        "ALA",
        "CYS",
        "CYD",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "HIS_D",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]

    restype_list = [
        cattr.structure(
            cattr.unstructure(
                next(r for r in chem_database.residues if r.name == rname)
            ),
            RefinedResidueType,
        )
        for rname in wanted
    ]

    return PackedBlockTypes.from_restype_list(restype_list, device), atom_type_resolver
