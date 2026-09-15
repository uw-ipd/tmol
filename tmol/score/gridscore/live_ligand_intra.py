"""Live ligand-intramolecular energy for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from tmol.database import ParameterDatabase
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType


LigandIntraWeightProfile = Literal["search", "full"]

_LJLK_ELEC_SCORE_TYPES = (
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
)
_CART_SCORE_TYPES = (
    ScoreType.cart_lengths,
    ScoreType.cart_angles,
    ScoreType.cart_torsions,
    ScoreType.cart_impropers,
    ScoreType.cart_hxltorsions,
)


def _intra_score_function(
    param_db: ParameterDatabase,
    device: torch.device,
    weight_profile: LigandIntraWeightProfile,
    include_cart_bonded: bool,
) -> ScoreFunction:
    """Build the tmol equivalent of Rosetta GridScorer's ``sf1b_``.

    Rosetta's ligand ``fa_intra_atr_xover4``, ``fa_intra_rep_xover4``,
    ``fa_intra_sol_xover4``, and ``fa_intra_elec`` map to tmol's ordinary
    ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``, and ``fa_elec`` score types. Their
    intra-block branches apply the ligand CP_CROSSOVER_3FULL count-pair rule.
    Rosetta ``gen_bonded`` maps to ``gen_torsions``; ligand ``cart_bonded`` is
    represented by tmol's five ``cart_*`` channels for cyclic ligands.

    Rosetta ``GridScorer.cc:338-343`` defines the full one-body terms. Its
    search setup at lines 297-313 replaces intra-repulsion with 0.01 and both
    generic/cartesian bonded weights with 0.1 (the ligand cartesian assignment
    is also visible at line 370). The remaining intra weights stay at beta2016.
    """

    if weight_profile not in ("search", "full"):
        raise ValueError("weight_profile must be 'search' or 'full'")

    beta = beta2016_score_function(device, param_db=param_db)
    result = ScoreFunction(param_db, device)
    score_types = _LJLK_ELEC_SCORE_TYPES + (ScoreType.gen_torsions,)
    if include_cart_bonded:
        score_types += _CART_SCORE_TYPES
    for score_type in score_types:
        weight = float(beta.get_weight(score_type))
        if weight_profile == "search":
            if score_type == ScoreType.fa_ljrep:
                weight = 0.01
            elif (
                score_type == ScoreType.gen_torsions
                or score_type in _CART_SCORE_TYPES
            ):
                weight = 0.1
        result.set_weight(score_type, weight)
    return result


def _ligand_pose_to(
    pose_stack: PoseStack,
    device: torch.device,
    dtype: torch.dtype,
) -> PoseStack:
    """Move a one-block ligand PoseStack and rebuild its device-local PBT."""

    # Resolve to a concrete device (e.g. cuda -> cuda:0) so the PoseStack's
    # declared device matches the device its tensors actually land on; an
    # indexless "cuda" would otherwise mismatch the tensors' "cuda:0".
    device = torch.empty(0, device=device).device

    source_pbt = pose_stack.packed_block_types
    source_type_index = int(pose_stack.block_type_ind64[0, 0])
    block_type = source_pbt.active_block_types[source_type_index]
    pbt = PackedBlockTypes.from_restype_list(
        source_pbt.chem_db,
        source_pbt.restype_set,
        [block_type],
        device=device,
    )

    def moved(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device).contiguous()

    block_type_ind64 = torch.zeros((1, 1), dtype=torch.int64, device=device)
    return PoseStack(
        packed_block_types=pbt,
        coords=pose_stack.coords.to(device=device, dtype=dtype).contiguous(),
        block_coord_offset=moved(pose_stack.block_coord_offset),
        block_coord_offset64=moved(pose_stack.block_coord_offset64),
        inter_residue_connections=moved(pose_stack.inter_residue_connections),
        inter_residue_connections64=moved(
            pose_stack.inter_residue_connections64
        ),
        inter_block_bondsep=moved(pose_stack.inter_block_bondsep),
        inter_block_bondsep64=moved(pose_stack.inter_block_bondsep64),
        block_type_ind=block_type_ind64.to(torch.int32),
        block_type_ind64=block_type_ind64,
        chain_id=moved(pose_stack.chain_id),
        chain_id64=moved(pose_stack.chain_id64),
        pdb_info=pose_stack.pdb_info,
        constraint_set=None,
        device=device,
    )


@dataclass(frozen=True)
class LiveLigandIntraData:
    """A ligand-only topology, weighted ScoreFunction, and compiled-scorer cache."""

    ligand_pose_stack: PoseStack
    score_function: ScoreFunction
    param_db: ParameterDatabase
    weight_profile: LigandIntraWeightProfile
    include_cart_bonded: bool
    _scorer_cache: dict[int, tuple[PoseStack, object]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        pose_stack = self.ligand_pose_stack
        if pose_stack.n_poses != 1 or pose_stack.max_n_blocks != 1:
            raise ValueError("live ligand-intra data requires one one-block pose")
        if pose_stack.max_n_pose_atoms < 1:
            raise ValueError("live ligand-intra data requires ligand atoms")
        if pose_stack.coords.dtype not in (torch.float32, torch.float64):
            raise TypeError("live ligand-intra coordinates must be float32 or float64")
        if pose_stack.device != pose_stack.coords.device:
            raise ValueError("live ligand-intra PoseStack device is inconsistent")
        if pose_stack.packed_block_types.device != pose_stack.device:
            raise ValueError(
                "live ligand-intra packed block types are on another device"
            )
        if self.weight_profile not in ("search", "full"):
            raise ValueError("weight_profile must be 'search' or 'full'")
        if not isinstance(self.include_cart_bonded, bool):
            raise TypeError("include_cart_bonded must be bool")

    @property
    def n_ligand_atoms(self) -> int:
        return self.ligand_pose_stack.max_n_pose_atoms

    def to(
        self, device: torch.device | str, dtype: torch.dtype
    ) -> "LiveLigandIntraData":
        """Return ligand topology and scoring machinery on a device/dtype."""

        if dtype not in (torch.float32, torch.float64):
            raise TypeError("live ligand-intra dtype must be float32 or float64")
        device = torch.device(device)
        ligand_pose_stack = _ligand_pose_to(self.ligand_pose_stack, device, dtype)
        score_function = _intra_score_function(
            self.param_db,
            device,
            self.weight_profile,
            self.include_cart_bonded,
        )
        return LiveLigandIntraData(
            ligand_pose_stack=ligand_pose_stack,
            score_function=score_function,
            param_db=self.param_db,
            weight_profile=self.weight_profile,
            include_cart_bonded=self.include_cart_bonded,
        )


def build_live_ligand_intra_data(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    weight_profile: LigandIntraWeightProfile,
) -> LiveLigandIntraData:
    """Build CPU ligand-only intra scoring data from one reference complex."""

    if reference_pose_stack.n_poses != 1:
        raise ValueError("live ligand-intra construction requires one reference pose")
    if reference_pose_stack.device.type != "cpu":
        raise ValueError("live ligand-intra construction requires a CPU reference pose")

    # Import locally to avoid a gridscore -> dock.scoring_backend -> gridscore
    # import cycle. This helper preserves the grid scorer's block-local atom order.
    from tmol.dock.ga import _ligand_only_pose_stack

    ligand_pose_stack = _ligand_only_pose_stack(reference_pose_stack, ligand_block)
    ligand_pose_stack = _ligand_pose_to(
        ligand_pose_stack, torch.device("cpu"), reference_pose_stack.coords.dtype
    )
    block_type = ligand_pose_stack.packed_block_types.active_block_types[0]
    include_cart_bonded = any(block_type.bond_to_ringness.values())
    score_function = _intra_score_function(
        param_db,
        torch.device("cpu"),
        weight_profile,
        include_cart_bonded,
    )
    return LiveLigandIntraData(
        ligand_pose_stack=ligand_pose_stack,
        score_function=score_function,
        param_db=param_db,
        weight_profile=weight_profile,
        include_cart_bonded=include_cart_bonded,
    )


def _validate_ligand_coords(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> None:
    if ligand_coords.ndim != 3 or ligand_coords.shape[2] != 3:
        raise ValueError("ligand_coords must have shape [n_poses, n_atoms, 3]")
    if ligand_coords.shape[0] == 0:
        raise ValueError("at least one ligand pose is required")
    if ligand_coords.shape[1] != data.n_ligand_atoms:
        raise ValueError("ligand atom count does not match live intra topology")
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if ligand_coords.device != data.ligand_pose_stack.device:
        raise ValueError("live ligand-intra data must match coordinate device")
    if ligand_coords.dtype != data.ligand_pose_stack.coords.dtype:
        raise TypeError("live ligand-intra data must match coordinate dtype")


def _scorer_for(data: LiveLigandIntraData, n_poses: int):
    cached = data._scorer_cache.get(n_poses)
    if cached is None:
        # Runtime-local import avoids importing dock.ga while scoring_backend is
        # importing gridscore. Batch topology is built once per population size.
        from tmol.dock.dock_loop import _repeat_pose_stack

        pose_stack = _repeat_pose_stack(data.ligand_pose_stack, n_poses)
        scorer = data.score_function.render_whole_pose_scoring_module(pose_stack)
        cached = (pose_stack, scorer)
        data._scorer_cache[n_poses] = cached
    return cached[1]


def score_ligand_intra(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return weighted ligand-intramolecular energy for every ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    scorer = _scorer_for(data, ligand_coords.shape[0])
    energies = scorer(ligand_coords.contiguous())
    expected_shape = (ligand_coords.shape[0],)
    if energies.shape != expected_shape:
        raise RuntimeError(
            f"ligand-intra scorer returned {tuple(energies.shape)}; "
            f"expected {expected_shape}"
        )
    return energies


def score_ligand_intra_with_gradients(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached weighted energies and compiled-autograd gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.enable_grad():
        coords_leaf = ligand_coords.detach().requires_grad_(True)
        energies = score_ligand_intra(data, coords_leaf)
        (gradients,) = torch.autograd.grad(energies.sum(), coords_leaf)
    return energies.detach(), gradients.detach()
