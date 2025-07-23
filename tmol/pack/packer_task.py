from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.pose_stack import PoseStack
from tmol.pack.rotamer.conformer_sampler import ConformerSampler
from tmol.pack.rotamer.chi_sampler import ChiSampler


# Architecture is borrowed from Rosetta3:
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
#   Different PackerPalletes will construct different
#   starting points which can be refined towards the
#   set of design choices that make sense for your
#   application


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

    def block_types_from_original(self, orig: RefinedResidueType):
        # ok, this is where we figure out what the allowed restypes
        # are for a residue; this might be complex logic.

        keepers = []
        for bt in self.rts.residue_types:
            if (
                bt.properties.polymer.is_polymer == orig.properties.polymer.is_polymer
                and bt.properties.polymer.polymer_type
                == orig.properties.polymer.polymer_type
                and bt.properties.polymer.backbone_type
                == orig.properties.polymer.backbone_type
                and bt.connections
                == orig.connections  # fd  use this instead of terminal variant check
                and set_compare(
                    bt.properties.chemical_modifications,
                    orig.properties.chemical_modifications,
                )
                and set_compare(
                    bt.properties.connectivity, orig.properties.connectivity
                )
                and bt.properties.protonation.protonation_state
                == orig.properties.protonation.protonation_state
            ):
                if (
                    bt.properties.polymer.sidechain_chirality
                    == orig.properties.polymer.sidechain_chirality
                ):
                    keepers.append(bt)
                elif orig.properties.polymer.polymer_type == "amino_acid" and (
                    (
                        orig.properties.polymer.sidechain_chirality == "l"
                        and bt.properties.polymer.sidechain_chirality == "achiral"
                    )
                    or (
                        orig.properties.polymer.sidechain_chirality == "achiral"
                        and bt.properties.polymer.sidechain_chirality == "l"
                    )
                ):
                    # allow glycine <--> l-caa mutations
                    keepers.append(bt)
                elif (
                    orig.properties.polymer.polymer_type == "amino_acid"
                    and orig.properties.polymer.sidechain_chirality == "d"
                    and bt.properties.polymer.sidechain_chirality == "achiral"
                ):
                    # allow d-caa --> glycine mutations;
                    # dangerous because this packer pallete will allow
                    # your d-caa to become glycine, and then later
                    # to an l-caa, but not the other way around
                    keepers.append(bt)

        return keepers


class BlockLevelTask:
    def __init__(
        self, seqpos: int, blocktype: RefinedResidueType, palette: PackerPalette
    ):
        self.seqpos = seqpos
        self.original_blocktype = blocktype
        self.allowed_blocktypes = palette.block_types_from_original(blocktype)
        self.conformer_samplers = []
        self.is_chi_sampler = []

    def restrict_to_repacking(self):
        orig = self.original_blocktype
        self.allowed_blocktypes = [
            bt
            for bt in self.allowed_blocktypes
            if bt.name3 == orig.name3  # this isn't what we want long term
        ]

    def disable_packing(self):
        # Note: we will always include at least one rotamer from every block
        # in the RotamerSet, falling back on the coordinates of the starting
        # block if this block-level-task is marked as kept fixed.
        self.allowed_blocktypes = []

    def add_conformer_sampler(self, sampler: ConformerSampler):
        self.conformer_samplers.append(sampler)
        self.is_chi_sampler.append(isinstance(sampler, ChiSampler))

    def restrict_absent_name3s(self, name3s):
        self.allowed_blocktypes = [
            bt for bt in self.allowed_blocktypes if bt.name3 in name3s
        ]


class PackerTask:

    def __init__(self, systems: PoseStack, palette: PackerPalette):
        self.blts = [
            [
                BlockLevelTask(j, systems.block_type(i, j), palette)
                for j in range(systems.max_n_blocks)
                if systems.is_real_block(i, j)
            ]
            for i in range(systems.n_poses)
        ]

    def restrict_to_repacking(self):
        for one_pose_blts in self.blts:
            for blt in one_pose_blts:
                blt.restrict_to_repacking()

    def add_conformer_sampler(self, sampler: ConformerSampler):
        for one_pose_blts in self.blts:
            for blt in one_pose_blts:
                blt.add_conformer_sampler(sampler)
