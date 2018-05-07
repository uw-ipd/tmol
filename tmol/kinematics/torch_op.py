import attr
import math
import torch

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from .datatypes import KinTree, KinDOF
from .metadata import DOFMetadata

from .operations import forwardKin, backwardKin, resolveDerivs


@attr.s(auto_attribs=True, frozen=True)
class KinematicOp:
    kintree: KinTree
    mobile_dofs: DOFMetadata

    src_dofs: KinDOF
    src_mobile_dofs: Tensor("f8")[:]

    result_size: int
    kin_coord_ind: Tensor(int)[:]
    result_coord_ind: Tensor(int)[:]

    @classmethod
    @validate_args
    def from_src_coords(
            cls,
            kintree: KinTree,
            mobile_dofs: DOFMetadata,
            src_coords: Tensor("f8")[:, 3],
    ):
        result_size = len(src_coords)

        kin_coord_ind = torch.nonzero(kintree.id >= 0)
        result_coord_ind = kintree.id[kin_coord_ind]

        working_coords = torch.full(
            (len(kintree), 3),
            math.nan,
            dtype=src_coords.dtype,
            layout=src_coords.layout,
            device=src_coords.device,
        )
        working_coords[kin_coord_ind] = src_coords[result_coord_ind]
        bkin = backwardKin(kintree, working_coords)
        src_mobile_dofs = bkin.dofs.raw[mobile_dofs.node_idx,
                                        mobile_dofs.dof_idx]

        return cls(
            kintree=kintree,
            mobile_dofs=mobile_dofs,
            src_dofs=bkin.dofs,
            src_mobile_dofs=src_mobile_dofs,
            result_size=result_size,
            kin_coord_ind=kin_coord_ind,
            result_coord_ind=result_coord_ind,
        )

    def apply(
            self,
            dofs: Tensor("f8")[:],
    ) -> Tensor("f8")[:]:
        return KinematicFun().apply(dofs, self)


class KinematicFun(torch.autograd.Function):
    @validate_args
    def forward(
            ctx,
            dofs: Tensor("f8")[:],
            config: KinematicOp,
    ) -> Tensor("f8")[:, 3]:

        assert len(dofs) == len(config.mobile_dofs)

        working_dofs = config.src_dofs.clone()
        working_dofs.raw[config.mobile_dofs.node_idx,
                         config.mobile_dofs.dof_idx] = dofs

        fkin = forwardKin(config.kintree, working_dofs)

        ctx.save_for_backward(config, dofs, fkin.hts)

        result_coords = torch.full(
            (config.result_size, 3),
            math.nan,
            dtype=dofs.dtype,
            layout=dofs.layout,
            device=dofs.device,
        )

        result_coords[config.result_coord_ind] = \
            fkin.coords[config.kin_coord_ind]

        return result_coords

    @validate_args
    def backward(
            ctx,
            coord_grads: Tensor("f8")[:, 3],
            _,
    ) -> Tensor("f8")[:]:
        config, working_dofs, hts = ctx.saved_tensors

        kincoord_derivs = torch.full(
            (len(config.kintree), 3),
            0,
            dtype=coord_grads.dtype,
            layout=coord_grads.layout,
            device=coord_grads.device,
        )

        kincoord_derivs[config.kin_coord_ind] = \
            coord_grads[config.result_coord_ind]

        working_derivs = resolveDerivs(
            config.kintree,
            working_dofs,
            hts,
            kincoord_derivs,
        )

        result_derivs = working_derivs.raw[config.mobile_dofs.node_idx,
                                           config.mobile_dofs.dof_idx]
        return result_derivs
