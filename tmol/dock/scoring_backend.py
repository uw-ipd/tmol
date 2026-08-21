"""Pluggable scoring and local-minimization backends for docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import LigandDockingDOFs
from tmol.optimization.minimizers import run_kin_min
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.score_function import ScoreFunction


class ScoringBackend(ABC):
    """The scoring contract used by the docking loop.

    A future ``GridBackend`` must implement these same two methods. ``score``
    returns one total beta16-compatible energy per pose, with shape
    ``[n_poses]``. ``minimize`` performs one batched local minimization over
    exactly the freedoms described by ``docking_dofs`` and returns a new
    :class:`PoseStack`. A grid implementation may use a different internal
    objective and gradient representation, but it must preserve these input and
    output semantics so the population loop does not depend on the backend.
    """

    @abstractmethod
    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return the total energy of each pose as a ``[n_poses]`` tensor."""

    @abstractmethod
    def minimize(
        self, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> PoseStack:
        """Locally minimize the supplied batch over its ligand docking DOFs."""


class ContinuousBackend(ScoringBackend):
    """Full continuous beta2016 scoring and kinematic minimization.

    ``param_db`` should be the ligand-extended parameter database returned by
    ligand preparation. If it is omitted, tmol's memoized default beta2016
    score function is used. A score function is cached per device, while the
    whole-pose scoring module is rendered for each input PoseStack.

    Args:
        param_db: Optional ligand-extended scoring parameter database.
        kin_dtype: Internal-coordinate dtype for minimization. By default the
            input coordinate dtype is used.
    """

    def __init__(
        self,
        param_db: Optional[ParameterDatabase] = None,
        *,
        kin_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if kin_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("kin_dtype must be torch.float32 or torch.float64")
        self.param_db = param_db
        self.kin_dtype = kin_dtype
        self._score_functions: dict[torch.device, ScoreFunction] = {}

    def _score_function(self, device: torch.device) -> ScoreFunction:
        device = torch.device(device)
        if device not in self._score_functions:
            self._score_functions[device] = beta2016_score_function(
                device, param_db=self.param_db
            )
        return self._score_functions[device]

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Score a PoseStack with the weighted continuous beta2016 function."""

        sfxn = self._score_function(pose_stack.device)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        with torch.no_grad():
            energies = scorer(pose_stack.coords)
        expected_shape = (pose_stack.n_poses,)
        if energies.shape != expected_shape:
            raise RuntimeError(
                "continuous whole-pose scorer returned shape "
                f"{tuple(energies.shape)}; expected {expected_shape}"
            )
        return energies

    def minimize(
        self, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> PoseStack:
        """Run ligand-only minimization and return detached coordinates."""

        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        minimized = run_kin_min(
            pose_stack,
            self._score_function(pose_stack.device),
            docking_dofs.fold_forest,
            docking_dofs.move_map,
            kin_dtype=kin_dtype,
        )
        minimized.coords = minimized.coords.detach()
        return minimized
