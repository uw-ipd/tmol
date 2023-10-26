import cattr
import torch

from tmol.database import ParameterDatabase
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.chemical.restypes import RefinedResidueType
import toolz.functoolz


# @toolz.functoolz.memoize
# def default_canonical_packed_block_types(device: torch.device):
#     chem_database = ParameterDatabase.get_default().chemical
#
#     restype_list = [
#         cattr.structure(
#             cattr.unstructure(r),
#             RefinedResidueType,
#         )
#         for r in chem_database.residues
#     ]
#
#     return PackedBlockTypes.from_restype_list(chem_database, restype_list, device)


# import cattr
# import torch
#
# from tmol.database import ParameterDatabase
# from tmol.pose.packed_block_types import PackedBlockTypes
# from tmol.score.chemical_database import AtomTypeParamResolver
# from tmol.chemical.restypes import RefinedResidueType
# import toolz.functoolz


@toolz.functoolz.memoize
def default_canonical_packed_block_types(device: torch.device):
    chem_database = ParameterDatabase.get_default().chemical
    # atom_type_resolver = AtomTypeParamResolver.from_database(
    #     chem_database, torch.device("cpu")
    # )

    wanted_base_types = [
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
        "HOH",  # <-- currently cannot be accommodated!?
    ]
    patches = ["", "nterm", "cterm", "cterm:nterm"]
    all_restypes = [
        x if y == "" else ":".join((x, y)) for x in wanted_base_types for y in patches
    ]

    restype_list = [
        cattr.structure(
            cattr.unstructure(
                next(r for r in chem_database.residues if r.name == rname)
            ),
            RefinedResidueType,
        )
        for rname in all_restypes
        if len([r for r in chem_database.residues if r.name == rname])
    ]

    return PackedBlockTypes.from_restype_list(chem_database, restype_list, device)
