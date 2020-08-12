import math
import attr
import torch.nn
from typing import Callable, TypeVar

from tmol.extern.singledispatchmethod import singledispatch, singledispatchmethod

from tmol.kinematics.scan_ordering import KinTreeScanOrdering
from tmol.kinematics.compiled import forward_kin_op

from .metadata import DOFMetadata
from .datatypes import KinTree

T = TypeVar("T")


@attr.s(auto_attribs=True, kw_only=True, eq=False)
class CartesianDOFs(torch.nn.Module):
    @staticmethod
    def build_from(val: object) -> "CartesianDOFs":
        return CartesianDOFs(coords=None).get_from(val)

    @singledispatchmethod
    def get_from(self, val: T) -> "CartesianDOFs":
        raise NotImplementedError(f"CartesianDOFs.set_on: {type(val)}")

    @singledispatchmethod
    def set_on(self, val: T) -> T:
        raise NotImplementedError(f"CartesianDOFs.set_on: {type(val)}")

    _module_init: bool = attr.ib(init=False, repr=False)

    @_module_init.default
    def _init_module(self):
        """Call Module.__init__ before other attrs for parameter registration."""
        super().__init__()

    coords: torch.nn.Parameter = attr.ib(converter=torch.nn.Parameter)

    def forward(self) -> torch.Tensor:  # Tensor[float][:, :, 3]
        return self.coords


@CartesianDOFs.get_from.register(CartesianDOFs)
def cartesian_dofs_get_from(self: CartesianDOFs, src: CartesianDOFs) -> CartesianDOFs:
    self.coords = torch.nn.Parameter(src.coords.clone())

    return self


@CartesianDOFs.set_on.register(CartesianDOFs)
def cartesian_dofs_set_on(self: CartesianDOFs, other: CartesianDOFs) -> CartesianDOFs:
    return other.get_from(self)


# mask out relevant dofs to the minimizer
class DOFMaskingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fg, mask, bg):
        ctx.mask = mask
        ctx.fg = fg
        result = bg.clone()
        result[mask] = fg
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad = torch.zeros_like(ctx.fg)
        grad = grad_output[ctx.mask]
        return grad, None, None


# Non-jit version of script module, torch 1.0.1 ScriptModule workaround
class KinematicModule(torch.nn.Module):
    """torch.autograd compatible forward kinematic operator.

    Perform forward (dof to coordinate) kinematics within torch.autograd
    compute graph. Provides support for forward kinematics over of a subset of
    source dofs, as specified by the provided DOFMetadata entries.

    The kinematic system maps between the natm x 9 internal coordinate frame
    and the natm x 3 coordinate frame.  Some of this natm x 9 array is unused
    or is redundant but this is not known by the kinematic module.

    See KinDOF for a description of the internal coordinate representation.
    """

    def __init__(self, kintree: KinTree):
        super().__init__()

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        self.register_buffer("kintree", torch.tensor([]))
        self.kintree = torch.stack(
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
        )

        ordering = KinTreeScanOrdering.for_kintree(kintree)

        self.register_buffer("nodes_f", torch.tensor([]))
        self.register_buffer("scans_f", torch.tensor([]))
        self.nodes_f = ordering.forward_scan_paths.nodes
        self.scans_f = ordering.forward_scan_paths.scans
        self.gens_f = ordering.forward_scan_paths.gens.cpu()  # Remains on CPU

        self.register_buffer("nodes_b", torch.tensor([]))
        self.register_buffer("scans_b", torch.tensor([]))
        self.nodes_b = ordering.backward_scan_paths.nodes
        self.scans_b = ordering.backward_scan_paths.scans
        self.gens_b = ordering.backward_scan_paths.gens.cpu()  # Remains on CPU

    def forward(self, dofs):
        return forward_kin_op(
            dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.nodes_b,
            self.scans_b,
            self.gens_b,
            self.kintree,
        )


@attr.s(auto_attribs=True, kw_only=True, eq=False)
class KinematicOperation(torch.nn.Module):
    @staticmethod
    @singledispatch
    def build_for(val: object) -> "KinematicOperation":
        raise NotImplementedError(f"KinematicOperation.build_for: {type(val)}")

    _module_init: bool = attr.ib(init=False, repr=False)

    @_module_init.default
    def _init_module(self):
        """Call Module.__init__ before other attrs for parameter registration."""
        super().__init__()

    system_size: int
    kintree: KinTree
    dof_metadata: DOFMetadata
    kin_module: KinematicModule = attr.ib(init=False)

    @kin_module.default
    def _init_kin_module(self):
        return KinematicModule(self.kintree)

    # TODO restore type annotations
    def forward(self, dofs: torch.Tensor) -> torch.Tensor:  # Tensor[torch.float][:, 9],

        kincoords = self.kin_module(dofs)

        coords = kincoords.new_full(size=(self.system_size, 3), fill_value=math.nan)

        idIdx = self.kintree.id[1:].to(dtype=torch.long)
        coords[idIdx] = kincoords[1:]

        return coords


@KinematicOperation.build_for.register(KinematicOperation)
def kinematic_operation_clone(src: KinematicOperation) -> KinematicOperation:
    return KinematicOperation(
        system_size=src.system_size, kintree=src.kintree, dof_metadata=src.dof_metadata
    )


@attr.s(auto_attribs=True, kw_only=True, eq=False)
class KinematicDOFs(torch.nn.Module):
    @classmethod
    def build_from(
        cls, val: object, *, dof_filter: Callable[[DOFMetadata], DOFMetadata] = None
    ):
        kinop: KinematicOperation = KinematicOperation.build_for(val)

        if dof_filter is None:
            selected_dofs = kinop.dof_metadata
        else:
            selected_dofs = dof_filter(kinop.dof_metadata)

        dof_mask = torch.stack((selected_dofs.node_idx, selected_dofs.dof_idx))

        return cls(
            kinop=kinop,
            selected_dofs=selected_dofs,
            dof_mask=dof_mask,
            full_dofs=None,
            dofs=None,
        ).get_from(val)

    @singledispatchmethod
    def get_from(self, val: object):
        raise NotImplementedError(f"get_from: {self} {val}")

    @singledispatchmethod
    def set_on(self, val: object):
        raise NotImplementedError(f"set_on: {self} {val}")

    _module_init: bool = attr.ib(init=False, repr=False)

    @_module_init.default
    def _init_module(self):
        """Call Module.__init__ before other attrs for parameter registration."""
        super().__init__()

        self.register_buffer("dof_mask", None)
        self.register_buffer("full_dofs", None)

    kinop: KinematicOperation

    selected_dofs: DOFMetadata
    dof_mask: torch.Tensor  # Tensor[int][2, :]

    full_dofs: torch.Tensor
    dofs: torch.nn.Parameter = attr.ib(converter=torch.nn.Parameter)

    def forward(self):
        return self.kinop(
            DOFMaskingFunc.apply(self.dofs, tuple(self.dof_mask), self.full_dofs)
        )[None, ...]


@KinematicOperation.build_for.register(KinematicDOFs)
def kinematic_operation_clone_dofs(src: KinematicOperation) -> KinematicOperation:
    return KinematicOperation.build_for(src.kinop)


@KinematicDOFs.get_from.register(KinematicDOFs)
def kinematic_dofs_get_from(self: KinematicDOFs, src: KinematicDOFs) -> KinematicDOFs:
    # TODO check kintree equivalency?
    src_dofs = src.full_dofs.clone()
    src_dofs[tuple(src.dof_mask)] = src.dofs

    self.full_dofs = src_dofs.detach()
    self.dofs = torch.nn.Parameter(self.full_dofs[tuple(self.dof_mask)])

    return self


@KinematicDOFs.set_on.register(KinematicDOFs)
def kinematic_dofs_set_on(self: KinematicDOFs, other: KinematicDOFs) -> KinematicDOFs:
    return other.get_from(self)
