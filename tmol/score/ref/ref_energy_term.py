import torch
import numpy

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.ref.ref_whole_pose_module import RefWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

from collections import OrderedDict


class RefEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(RefEnergyTerm, self).__init__(param_db=param_db, device=device)

        """self.global_params = RefGlobalParams.from_database(
            param_db.scoring.ref, device
        )"""
        self.device = device

        self.ref_weights = OrderedDict()

        # 1.32468 3.25479 -2.14574 -2.72453 1.21829 0.79816 -0.30065 2.30374 -0.71458 1.66147 1.65735 -1.34026 -1.64321 -1.45095 -0.09474 -0.28969 1.15175 2.64269 2.26099 0.58223

        self.ref_weights["ALA"] = 1.32468
        self.ref_weights["ARG"] = 3.25479
        self.ref_weights["ASN"] = -2.14574
        self.ref_weights["ASP"] = -2.72453
        self.ref_weights["CYS"] = 1.21829
        self.ref_weights["GLN"] = 0.79816
        self.ref_weights["GLU"] = -0.30065
        self.ref_weights["GLY"] = 2.30374
        self.ref_weights["HIS"] = -0.71458
        self.ref_weights["ILE"] = 1.66147
        self.ref_weights["LEU"] = 1.65735
        self.ref_weights["LYS"] = -1.34026
        self.ref_weights["MET"] = -1.64321
        self.ref_weights["PHE"] = -1.45095
        self.ref_weights["PRO"] = -0.09474
        self.ref_weights["SER"] = -0.28969
        self.ref_weights["THR"] = 1.15175
        self.ref_weights["TRP"] = 2.64269
        self.ref_weights["TYR"] = 2.26099
        self.ref_weights["VAL"] = 0.58223

    @classmethod
    def score_types(cls):
        import tmol.score.terms.ref_creator

        return tmol.score.terms.ref_creator.RefTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(RefEnergyTerm, self).setup_block_type(block_type)

        print(block_type.name3)
        ref_weight = self.ref_weights[block_type.name3]

        # if hasattr(block_type, "ref_value"):
        # return

        setattr(block_type, "ref_weight", ref_weight)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(RefEnergyTerm, self).setup_packed_block_types(packed_block_types)

        """ref_weights = torch.full(
            (packed_block_types.n_types),
            -1,
            dtype=torch.float32,
            device=self.device,
        )"""
        ref_weights = []
        for i, bt in enumerate(packed_block_types.active_block_types):
            ref_weights += [bt.ref_weight]

        print(self.ref_weights.values())
        ref_weights = torch.as_tensor(
            ref_weights, dtype=torch.float32, device=self.device
        )

        setattr(packed_block_types, "ref_weights", ref_weights)

    def setup_poses(self, poses: PoseStack):
        super(RefEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return RefWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            ref_weights=pbt.ref_weights,
            # global_params=self.global_params,
        )
