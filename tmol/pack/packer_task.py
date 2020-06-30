import torch
import pandas
import numpy

from tmol.database.chemical import ChemicalDatabase
from tmol.system.restypes import ResidueType, Residue
from tmol.system.packed import PackedResidueSystem


# Architecture is stolen from Rosetta3:
# PackerTask: a class holding data describing how the
#   packer should behave. Each position in the
#   PackerTask corresponds to a residue in the input
#   PackedResidueSystem. For each residue, it holds
#   a list of the allowed residue types that can be
#   built at those positions. The PackerTask can be
#   modified only by removing residue types from those
#   list, not by adding new ones.
#
# PackerPallete: a class that decides how to construct
#   a PackerTask, deciding which residue types to allow
#   based on the residue type of the input structure.
#   The PackerPallete can be given


class PackerPalette:
    def __init__(self, chemdb: ChemicalDatabase):
        self.chemdb = chemdb

    def restypes_from_original(self, original: ResidueType):
        # ok, this is where we figure out what the allowed restypes
        # are for a residue; this might be complex logic.
        # for now, I'm going to punt.
        keepers = []
        is_l_aa = sum(1 for h in original.hierarchies if h.startswith("aa.alpha.l")) > 0
        if is_l_aa:
            for res in self.chemdb.residues:
                for hstring in res.hierarchies:
                    if hstring.startswith("aa.alpha.l"):
                        keepers.append(res)
                        break
        else:
            # just for now, keep the original residue type
            keepers.append(
                [rt for rt in self.chemdb.residues if rt.name == original.name]
            )
        return keepers


class ResidueLevelTask:
    def __init__(self, seqpos: int, restype: ResidueType, palette: PackerPalette):
        self.seqpos = seqpos
        self.original_restype = restype
        self.allowed_restypes = palette.restypes_from_original(restype)

    def restrict_to_repacking(self):
        # Wow, the hierarchies scheme seems so janky
        self.allowed_restypes = [
            rt
            for rt in self.allowed_restypes
            if (
                len(rt.hierarchies[0].split(".")) >= 4
                and len(self.original_restype.hierarchies[0].split(".")) >= 4
                and rt.hierarchies[0].split(".")[:4]
                == self.original_restype.hierarchies[0].split(".")[:4]
            )
        ]


class PackerTask:
    def __init__(self, system: PackedResidueSystem, palette: PackerPalette):
        self.rlts = [
            ResidueLevelTask(i, res.residue_type, palette)
            for i, res in enumerate(system.residues)
        ]

    def restrict_to_repacking(self):
        for rlt in self.rlts:
            rls.restrict_to_repacking()
