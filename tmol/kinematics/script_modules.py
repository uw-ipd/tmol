import torch


from .datatypes import KinForest
from .fold_forest import FoldForest
from .check_fold_forest import validate_fold_forest

from tmol import PoseStack
from tmol.kinematics.compiled import forward_kin_op

from tmol.kinematics.scan_ordering import (
    KinForestScanOrdering,
    construct_kin_module_data_for_pose,
    # _annotate_block_type_with_gen_scan_path_segs,
    _annotate_packed_block_type_with_gen_scan_path_segs,
)

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class KinematicModule(torch.jit.ScriptModule):
    """torch.autograd compatible forward kinematic operator.

    Perform forward (dof to coordinate) kinematics within torch.autograd
    compute graph. Provides support for forward kinematics over of a subset of
    source dofs, as specified by the provided DOFMetadata entries.

    The kinematic system maps between the natm x 9 internal coordinate frame
    and the natm x 3 coordinate frame.  Some of this natm x 9 array is unused
    or is redundant but this is not known by the kinematic module.

    See KinDOF for a description of the internal coordinate representation.
    """

    def __init__(self, kinforest: KinForest, device: torch.device):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        self.kinforest = _p(
            torch.stack(
                _tint(
                    [
                        kinforest.id,
                        kinforest.doftype,
                        kinforest.parent,
                        kinforest.frame_x,
                        kinforest.frame_y,
                        kinforest.frame_z,
                    ]
                ),
                dim=1,
            ).to(device)
        )

        ordering = KinForestScanOrdering.for_kinforest(kinforest)
        self.nodes_f = _p(ordering.forward_scan_paths.nodes.to(device))
        self.scans_f = _p(ordering.forward_scan_paths.scans.to(device))
        self.gens_f = _p(ordering.forward_scan_paths.gens)  # on cpu
        self.nodes_b = _p(ordering.backward_scan_paths.nodes.to(device))
        self.scans_b = _p(ordering.backward_scan_paths.scans.to(device))
        self.gens_b = _p(ordering.backward_scan_paths.gens)  # on cpu

    @torch.jit.script_method
    def forward(self, dofs):
        return forward_kin_op(
            dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.nodes_b,
            self.scans_b,
            self.gens_b,
            self.kinforest,
        )


class PoseStackKinematicModule(torch.jit.ScriptModule):
    """torch.autograd compatible forward kinematic operator for PoseStack.

    Perform forward (dof to coordinate) kinematics within torch.autograd
    compute graph. Provides support for forward kinematics over of a subset of
    source dofs, as specified by the provided DOFMetadata entries.

    The kinematic system maps between the natm x 9 internal coordinate frame
    and the natm x 3 coordinate frame.  Some of this natm x 9 array is unused
    or is redundant but this is not known by the kinematic module.

    See KinDOF for a description of the internal coordinate representation.
    """

    def __init__(self, pose_stack: PoseStack, fold_forest: FoldForest):
        super().__init__()

        ps = pose_stack
        pbt = pose_stack.packed_block_types
        ff = fold_forest
        device = pose_stack.device

        # Setup: initial annotations of block types and packed block types
        # with the per-block-scan-path segments.
        _annotate_packed_block_type_with_gen_scan_path_segs(pbt)

        n_blocks = torch.sum(ps.block_type_ind != -1, dim=1).cpu().numpy()
        validate_fold_forest(ff.roots, n_blocks, ff.edges)

        pbt_gssps = pbt.gen_seg_scan_path_segs
        ff_edges_cpu = torch.from_numpy(ff.edges).to(torch.int32)
        kmd = construct_kin_module_data_for_pose(ps, ff_edges_cpu)

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        self.kmd = kmd

        self.kinforest = _p(
            torch.stack(
                _tint(
                    [
                        kmd.forest.id,
                        kmd.forest.doftype,
                        kmd.forest.parent,
                        kmd.forest.frame_x,
                        kmd.forest.frame_y,
                        kmd.forest.frame_z,
                    ]
                ),
                dim=1,
            ).to(device)
        )

        self.nodes_f = _p(kmd.scan_data_fw.nodes.to(device))
        self.scans_f = _p(kmd.scan_data_fw.scans.to(device))
        self.gens_f = _p(kmd.scan_data_fw.gens)  # on cpu
        self.nodes_b = _p(kmd.scan_data_bw.nodes.to(device))
        self.scans_b = _p(kmd.scan_data_bw.scans.to(device))
        self.gens_b = _p(kmd.scan_data_bw.gens)  # on cpu

    @torch.jit.script_method
    def forward(self, dofs):
        return forward_kin_op(
            dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.nodes_b,
            self.scans_b,
            self.gens_b,
            self.kinforest,
        )
