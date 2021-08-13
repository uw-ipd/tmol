from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.pose_stack import Poses
from tmol.pack.rotamer.chi_sampler import ChiSampler


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


def set_compare(x, y):
    """Treat the collections x and y as if they are sets. Return true if they
    contain the same elements and false otherwise
    """
    if len(x) != len(y):
        return False
    for val in x:
        if val not in y:
            return False
    return True


class PackerPalette:
    def __init__(self, rts: ResidueTypeSet):
        self.rts = rts

    def restypes_from_original(self, orig: RefinedResidueType):
        # ok, this is where we figure out what the allowed restypes
        # are for a residue; this might be complex logic.

        keepers = []
        for rt in self.rts.residue_types:
            if (
                rt.properties.polymer.is_polymer == orig.properties.polymer.is_polymer
                and rt.properties.polymer.polymer_type
                == orig.properties.polymer.polymer_type
                and rt.properties.polymer.backbone_type
                == orig.properties.polymer.backbone_type
                and rt.properties.polymer.termini_variants
                == orig.properties.polymer.termini_variants
                and set_compare(
                    rt.properties.chemical_modifications,
                    orig.properties.chemical_modifications,
                )
                and set_compare(
                    rt.properties.connectivity, orig.properties.connectivity
                )
                and rt.properties.protonation.protonation_state
                == orig.properties.protonation.protonation_state
            ):
                if (
                    rt.properties.polymer.sidechain_chirality
                    == orig.properties.polymer.sidechain_chirality
                ):
                    keepers.append(rt)
                elif orig.properties.polymer.polymer_type == "amino_acid" and (
                    (
                        orig.properties.polymer.sidechain_chirality == "l"
                        and rt.properties.polymer.sidechain_chirality == "achiral"
                    )
                    or (
                        orig.properties.polymer.sidechain_chirality == "achiral"
                        and rt.properties.polymer.sidechain_chirality == "l"
                    )
                ):
                    # allow glycine <--> l-caa mutations
                    keepers.append(rt)
                elif (
                    orig.properties.polymer.polymer_type == "amino_acid"
                    and orig.properties.polymer.sidechain_chirality == "d"
                    and rt.properties.polymer.sidechain_chirality == "achiral"
                ):
                    # allow d-caa --> glycine mutations;
                    # dangerous because this packer pallete will allow
                    # your d-caa to become glycine, and then later
                    # to an l-caa, but not the other way around
                    keepers.append(rt)

        return keepers


class ResidueLevelTask:
    def __init__(
        self, seqpos: int, restype: RefinedResidueType, palette: PackerPalette
    ):
        self.seqpos = seqpos
        self.original_restype = restype
        self.allowed_restypes = palette.restypes_from_original(restype)
        self.chi_samplers = []

    def restrict_to_repacking(self):
        orig = self.original_restype
        self.allowed_restypes = [
            rt
            for rt in self.allowed_restypes
            if rt.name3 == orig.name3  # this isn't what we want long term
        ]

    def disable_packing(self):
        self.allowed_restypes = []

    def add_chi_sampler(self, sampler: ChiSampler):
        self.chi_samplers.append(sampler)

    def restrict_absent_name3s(self, name3s):
        self.allowed_restypes = [
            rt for rt in self.allowed_restypes if rt.name3 in name3s
        ]


class PackerTask:
    # def __init__(self, system: PackedResidueSystem, palette: PackerPalette):
    #     self.rlts = [
    #         ResidueLevelTask(i, res.residue_type, palette)
    #         for i, res in enumerate(system.residues)
    #     ]

    def __init__(self, systems: Poses, palette: PackerPalette):
        # print("task ctor")
        # print(
        #     "system sizes",
        #     len(systems.residues),
        #     [len(ires) for ires in systems.residues],
        # )
        self.rlts = [
            [
                ResidueLevelTask(j, res.residue_type, palette)
                for j, res in enumerate(ires)
                if systems.block_type_ind[i, j] >= 0
            ]
            for i, ires in enumerate(systems.residues)
        ]

    def restrict_to_repacking(self):
        for one_pose_rlts in self.rlts:
            for rlt in one_pose_rlts:
                rlt.restrict_to_repacking()

    def add_chi_sampler(self, sampler: ChiSampler):
        for one_pose_rlts in self.rlts:
            for rlt in one_pose_rlts:
                rlt.add_chi_sampler(sampler)
