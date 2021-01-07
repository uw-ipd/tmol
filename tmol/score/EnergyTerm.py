from tmol.system.pose import PackedBlockTypes, Poses
from tmol.system.restypes import RefinedResidueType


class EnergyTerm:
    def setup_block_type(self, block_type: RefinedResidueType):
        pass

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    def setup_poses(self, poses: Poses):
        pass
