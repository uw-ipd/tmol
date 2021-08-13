from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import Poses


class EnergyTerm:
    def __init__(self, **kwargs):
        pass

    def setup_block_type(self, block_type: RefinedResidueType):
        pass

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    def setup_poses(self, poses: Poses):
        pass
