import torch

from .datatypes import KinTree

import tmol.kinematics.compiled  # noqa

from tmol.kinematics.scan_ordering import KinTreeScanOrdering

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class KinematicModule(torch.jit.ScriptModule):
    """torch.autograd compatible forward kinematic operator.

    Perform forward (dof to coordinate) kinematics within torch.autograd
    compute graph. Provides support for forward kinematics over of a subset of
    source dofs, as specified by the provided DOFMetadata entries.

    A kinematic system is defined by a combination of mobile and fixed dofs, a
    KinematicOp manages forward kinematics for mobile dofs within a fixed dof
    context. The full mobile & fixed dof set is initialized by backward
    kinematics from a given coordinate state via the `from_coords` factory
    function.
    """

    def __init__(self, kintree: KinTree, device: torch.device):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        self.kintree = _p(
            torch.stack(
                _tint(
                    [
                        kintree.id,
                        kintree.doftype,
                        kintree.parent,
                        kintree.frame_x,
                        kintree.frame_y,
                        kintree.frame_z,
                    ]
                ),
                dim=1,
            ).to(device)
        )

        ordering = KinTreeScanOrdering.for_kintree(kintree)
        self.nodes_f = _p(ordering.forward_scan_paths.nodes.to(device))
        self.scans_f = _p(ordering.forward_scan_paths.scans.to(device))
        self.gens_f = _p(ordering.forward_scan_paths.gens)  # on cpu
        self.nodes_b = _p(ordering.backward_scan_paths.nodes.to(device))
        self.scans_b = _p(ordering.backward_scan_paths.scans.to(device))
        self.gens_b = _p(ordering.backward_scan_paths.gens)  # on cpu

    @torch.jit.script_method
    def forward(self, dofs):
        return torch.ops.tmol.forward_kin_op(
            dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.nodes_b,
            self.scans_b,
            self.gens_b,
            self.kintree,
        )
