import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.score.constraint.params import ConstraintResolver
from tmol.score.constraint.script_modules import ConstraintIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.chemical_database import ChemicalDB
from tmol.score.modules.stacked_system import StackedSystem
from tmol.score.modules.bonded_atom import BondedAtoms


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class ConstraintParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(val, system: ScoreSystem, *, cstdata: Optional[dict] = None, **_):
        """Override constructor.
        """
        spline_xs, spline_ys = None, None
        if cstdata is not None:
            spline_xs = cstdata["dense_cb_spline_xs"]
            spline_ys = cstdata["dense_cb_spline_ys"]
        return ConstraintParameters(
            system=system, spline_xs=spline_xs, spline_ys=spline_ys
        )

    spline_xs: torch.Tensor = attr.ib(validator=type_validator())
    spline_ys: torch.Tensor = attr.ib(validator=type_validator())

    constraint_resolver: ConstraintResolver = attr.ib(init=False)

    @constraint_resolver.default
    def _init_cst_param_resolver(self) -> ConstraintResolver:
        return ConstraintResolver.from_dense_CB_spline_data(
            device=TorchDevice.get(self).device,
            atm_names=BondedAtoms.get(self).atom_names,
            res_indices=BondedAtoms.get(self).res_indices,
            spline_xs=self.spline_xs,
            spline_ys=self.spline_ys,
        )


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class ConstraintScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {ConstraintParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "ConstraintScore":
        return ConstraintScore(system=system)

    cst_intra_module: ConstraintIntraModule = attr.ib(init=False)

    @cst_intra_module.default
    def _init_cst_intra_module(self):
        return ConstraintIntraModule(ConstraintParameters.get(self).constraint_resolver)

    def intra_forward(self, coords: torch.Tensor):
        return {"cst": self.cst_intra_module(coords)}
