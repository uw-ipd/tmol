import attr
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.score.constraint.params import ConstraintResolver
from tmol.score.constraint.script_modules import ConstraintIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
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
        if cstdata is None:
            cstdata = {}

        return ConstraintParameters(system=system, cstdata=cstdata)

    cstdata: dict = attr.ib()
    constraint_resolver: ConstraintResolver = attr.ib(init=False)

    @constraint_resolver.default
    def _init_cst_param_resolver(self) -> ConstraintResolver:
        return ConstraintResolver.from_dense_6D(
            device=TorchDevice.get(self).device,
            atm_names=BondedAtoms.get(self).atom_names,
            res_indices=BondedAtoms.get(self).res_indices,
            csts=self.cstdata,
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
        cst_atompair, cst_dihedral, cst_angle = self.cst_intra_module(coords)
        return {
            "cst_atompair": cst_atompair,
            "cst_dihedral": cst_dihedral,
            "cst_angle": cst_angle,
        }
