import attr
import torch

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from .datatypes import KinTree, KinDOF
from .metadata import DOFMetadata

from .operations import forwardKin, backwardKin, resolveDerivs


@attr.s(auto_attribs=True, frozen=True)
class KinematicOp:
    """torch.autograd compatible forward kinematic operator.

    Perform forward (dof to coordinate) kinematics within torch.autograd
    compute graph. Provides support for forward kinematics over of a subset of
    source dofs, as specified by the provided DOFMetadata entries.

    A kinematic system is defined by a combination of mobile and fixed dofs, a
    KinematicOp manages forward kinematics for mobile dofs within a fixed dof
    context. The full mobile & fixed dof set is initialized by backward
    kinematics from a given coordinate state via the `from_coords` factory
    function.

    KinematicOp provides an interface mirroring an autograd function and can be
    used via either `__call__` or `apply`. Unlike an autograd function, the Op
    is reused during compute graph construction and serves as a factory for
    single use KinematicFun functions.
    """

    kintree: KinTree
    mobile_dofs: DOFMetadata

    src_dofs: KinDOF
    src_mobile_dofs: Tensor("f8")[:]

    @classmethod
    @validate_args
    def from_coords(
            cls,
            kintree: KinTree,
            mobile_dofs: DOFMetadata,
            kin_coords: Tensor("f8")[:, 3],
    ):
        """Construct KinematicOp for given mobile dofs via backward kinematics."""
        bkin = backwardKin(kintree, kin_coords)
        src_mobile_dofs = bkin.dofs.raw[mobile_dofs.node_idx,
                                        mobile_dofs.dof_idx]

        return cls(
            kintree=kintree,
            mobile_dofs=mobile_dofs,
            src_dofs=bkin.dofs,
            src_mobile_dofs=src_mobile_dofs,
        )

    def __call__(
            self,
            dofs: Tensor("f8")[:],
    ) -> Tensor("f8")[:]:
        return self.apply(dofs)

    def apply(
            self,
            dofs: Tensor("f8")[:],
    ) -> Tensor("f8")[:]:
        return KinematicFun(self)(dofs)


class KinematicFun(torch.autograd.Function):
    """Autograd function for KinematicOp.

    A standard, single-use, autograd function implementing the forward &
    backward passes for a given KinematicOp.
    """

    def __init__(self, kinematic_op: KinematicOp):
        self.kinematic_op = kinematic_op
        super().__init__()

    @validate_args
    def forward(
            ctx,
            dofs: Tensor("f8")[:],
    ) -> Tensor("f8")[:, 3]:

        assert len(dofs) == len(ctx.kinematic_op.mobile_dofs)

        working_dofs = ctx.kinematic_op.src_dofs.clone()
        working_dofs.raw[ctx.kinematic_op.mobile_dofs.node_idx,
                         ctx.kinematic_op.mobile_dofs.dof_idx] = dofs

        fkin = forwardKin(ctx.kinematic_op.kintree, working_dofs)

        ctx.save_for_backward(working_dofs.raw, fkin.hts)

        return fkin.coords

    @validate_args
    def backward(
            ctx,
            coord_grads: Tensor("f8")[:, 3],
    ) -> Tensor("f8")[:]:
        working_dofs_raw, hts = ctx.saved_tensors
        working_dofs = KinDOF(raw=working_dofs_raw)

        working_derivs = resolveDerivs(
            ctx.kinematic_op.kintree,
            working_dofs,
            hts,
            coord_grads,
        )

        result_derivs = working_derivs.raw[
            ctx.kinematic_op.mobile_dofs.node_idx,
            ctx.kinematic_op.mobile_dofs.dof_idx
        ] # yapf:disable

        return result_derivs
