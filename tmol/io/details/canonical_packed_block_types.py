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

    restype_list = [
        cattr.structure(
            cattr.unstructure(r),
            RefinedResidueType,
        )
        for r in chem_database.residues
    ]

    return PackedBlockTypes.from_restype_list(chem_database, restype_list, device)
