import attr

import torch

from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.ljlk.params import LJLKParamResolver, LJLKGlobalParams

from tmol.utility.cpp_extension import load, modulename, relpaths

_compiled = load(modulename(__name__), relpaths(__file__, "water.pybind.cc"))


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AttachedWaters:

    atom_resolver: AtomTypeParamResolver
    global_params: LJLKGlobalParams

    @classmethod
    def from_database(cls, database, torch_device):
        return cls(
            AtomTypeParamResolver.from_database(database.chemical, torch.device("cpu")),
            global_params=LJLKParamResolver.from_database(
                database.chemical, database.scoring.ljlk, torch.device("cpu")
            ).global_params,
        )

    def apply(self, coords, atom_types, indexed_bonds):
        atom_type_params = self.atom_resolver.params[atom_types]

        return AttachedWatersFun(self, indexed_bonds, atom_type_params)(coords)


class AttachedWatersFun(torch.autograd.Function):
    def __init__(self, op, indexed_bonds, atom_type_params):
        self.op = op
        self.atom_type_params = atom_type_params
        self.indexed_bonds = indexed_bonds

        super().__init__()

    def forward(ctx, coords):

        water_locations = _compiled.attached_waters_forward(
            coords, ctx.indexed_bonds, ctx.atom_type_params, ctx.op.global_params
        )

        ctx.inputs = (
            coords,
            ctx.indexed_bonds,
            ctx.atom_type_params,
            ctx.op.global_params,
        )

        return water_locations

    def backward(ctx, dE_dW):
        water_locations = _compiled.attached_waters_backward(dE_dW, *ctx.inputs)

        return water_locations
