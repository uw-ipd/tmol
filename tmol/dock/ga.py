"""Vectorized GALigandDock genetic operators for one ligand population.

The control flow mirrors Rosetta ``GAOptimizer.cc:104-113``: generate an
offspring batch, optimize the whole batch, then merge it into the parent pool.
The gene is exactly the MoveMap-freed raw DOFs: the ligand jump's three
translations and three delta rotations followed by logical ligand chis.
Logical chis use the docking-local sign/offset mapping in ``ligand_dofs.py``.
Genes are only converted to coordinates by forward kinematics; minimized genes
come directly from the optimizer's freed DOFs, never from inverse kinematics of
the minimized pose. Ring and receptor-sidechain genes are out of scope.
Continuous scoring has no grid smoothing or fa_rep ramp, so this module runs
the same schedule un-ramped until a grid backend implements those knobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import attr
import torch

from tmol.dock.dock_loop import DockingPerturbation, _repeat_pose_stack
from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    _inverse_dofs,
    _ligand_kinforest_nodes,
    _wrap_angle,
    build_ligand_docking_dofs,
)
from tmol.dock.scoring_backend import MinimizationResult, ScoringBackend
from tmol.kinematics.datatypes import JumpDOFTypes, NodeType
from tmol.kinematics.move_map import MinimizerMap
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.pose.pose_stack import PoseStack


@dataclass(frozen=True)
class GASchedule:
    """Single-stage VSX schedule for the continuous backend."""

    generations: int = 5
    npool: int = 50
    rmsdthreshold: float = 2.0
    pmut: float = 0.2
    maxiter: int = 50
    initial_perturbation: DockingPerturbation = DockingPerturbation()

    def __post_init__(self) -> None:
        for name in ("generations", "npool", "maxiter"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.rmsdthreshold) or self.rmsdthreshold < 0:
            raise ValueError("rmsdthreshold must be finite and nonnegative")
        if not math.isfinite(self.pmut) or not 0.0 <= self.pmut <= 1.0:
            raise ValueError("pmut must be between zero and one")


@dataclass(frozen=True)
class LigandGeneLayout:
    """Map ``[pose, six raw jump DOFs + logical chis]`` to coordinates."""

    template: PoseStack
    docking_dofs: LigandDockingDOFs
    kin_module: PoseStackKinematicsModule
    base_raw_dofs: torch.Tensor
    jump_nodes: torch.Tensor
    torsion_nodes: torch.Tensor
    torsion_dof_types: torch.Tensor
    torsion_signs: torch.Tensor
    torsion_offsets: torch.Tensor
    ligand_nodes: torch.Tensor
    ligand_local_atoms: torch.Tensor
    heavy_local_atoms: torch.Tensor

    @classmethod
    def from_pose_stack(
        cls, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> "LigandGeneLayout":
        kin_module = PoseStackKinematicsModule(
            pose_stack, docking_dofs.fold_forest
        )
        kmd = kin_module.kmd
        raw_dofs = _inverse_dofs(pose_stack, kin_module)
        minmap = MinimizerMap(pose_stack, kmd, docking_dofs.move_map)

        jump_mask = (
            (kmd.forest.doftype == int(NodeType.jump))
            & torch.all(
                minmap.dof_mask[:, : int(JumpDOFTypes.RBdel_gamma) + 1],
                dim=1,
            )
        )
        jump_node_list = torch.nonzero(jump_mask, as_tuple=False).flatten()
        if jump_node_list.numel() != pose_stack.n_poses:
            raise ValueError("expected exactly one movable ligand jump per pose")
        jump_pose = torch.div(
            kmd.forest.id[jump_node_list],
            pose_stack.max_n_pose_atoms,
            rounding_mode="floor",
        ).to(torch.int64)
        jump_nodes = torch.empty(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        jump_nodes.scatter_(0, jump_pose, jump_node_list)

        block_type_index = int(
            pose_stack.block_type_ind64[0, docking_dofs.ligand_block]
        )
        block_type = pose_stack.packed_block_types.active_block_types[
            block_type_index
        ]
        ligand_nodes = _ligand_kinforest_nodes(
            pose_stack,
            docking_dofs.ligand_block,
            block_type.n_atoms,
            kmd.forest.id,
        )
        raw_atoms = torch.tensor(
            [mapping.raw_atom for mapping in docking_dofs.torsions],
            dtype=torch.int64,
            device=pose_stack.device,
        )
        torsion_dof_types = torch.tensor(
            [int(mapping.dof_type) for mapping in docking_dofs.torsions],
            dtype=torch.int64,
            device=pose_stack.device,
        )
        torsion_signs = torch.tensor(
            [mapping.sign for mapping in docking_dofs.torsions],
            dtype=raw_dofs.dtype,
            device=pose_stack.device,
        )
        if docking_dofs.torsions:
            torsion_offsets = torch.stack(
                [mapping.offset for mapping in docking_dofs.torsions], dim=1
            ).to(dtype=raw_dofs.dtype)
        else:
            torsion_offsets = torch.empty(
                (pose_stack.n_poses, 0),
                dtype=raw_dofs.dtype,
                device=pose_stack.device,
            )
        heavy_local_atoms = torch.nonzero(
            ~pose_stack.packed_block_types.atom_is_hydrogen[
                block_type_index, : block_type.n_atoms
            ].to(torch.bool),
            as_tuple=False,
        ).flatten()

        return cls(
            template=pose_stack,
            docking_dofs=docking_dofs,
            kin_module=kin_module,
            base_raw_dofs=raw_dofs,
            jump_nodes=jump_nodes,
            torsion_nodes=ligand_nodes[:, raw_atoms],
            torsion_dof_types=torsion_dof_types,
            torsion_signs=torsion_signs,
            torsion_offsets=torsion_offsets,
            ligand_nodes=ligand_nodes,
            ligand_local_atoms=torch.arange(
                block_type.n_atoms,
                dtype=torch.int64,
                device=pose_stack.device,
            ),
            heavy_local_atoms=heavy_local_atoms,
        )

    @property
    def n_dof(self) -> int:
        return 6 + self.torsion_nodes.shape[1]

    def genes_from_raw_dofs(self, raw_dofs: torch.Tensor) -> torch.Tensor:
        """Gather the fixed-order gene from a raw kinematic DOF tensor."""

        if raw_dofs.shape != self.base_raw_dofs.shape:
            raise ValueError(
                "raw DOF shape does not match the gene layout: "
                f"{tuple(raw_dofs.shape)} != {tuple(self.base_raw_dofs.shape)}"
            )
        jump_dofs = raw_dofs[
            self.jump_nodes[:, None],
            torch.arange(6, dtype=torch.int64, device=raw_dofs.device)[None, :],
        ]
        raw_torsions = raw_dofs[
            self.torsion_nodes, self.torsion_dof_types[None, :]
        ]
        logical_torsions = _wrap_angle(
            self.torsion_signs[None, :] * raw_torsions
            + self.torsion_offsets
        )
        return torch.cat((jump_dofs, logical_torsions), dim=1)

    def genes_from_pose_stack(self, pose_stack: PoseStack) -> torch.Tensor:
        """Extract the starting template gene from its freed raw DOFs.

        This inverse-kinematics path exists only to seed the initial pool. The
        generation loop receives minimized genes directly from the optimizer.
        """

        if pose_stack.n_poses != self.template.n_poses:
            raise ValueError("pose stack size does not match the gene layout")
        raw_dofs = _inverse_dofs(pose_stack, self.kin_module)
        return self.genes_from_raw_dofs(raw_dofs)

    def raw_dofs_from_genes(self, genes: torch.Tensor) -> torch.Tensor:
        """Write genes into a clone of the template raw-DOF tensor."""

        expected = (self.template.n_poses, self.n_dof)
        if genes.shape != expected:
            raise ValueError(
                f"genes have shape {tuple(genes.shape)}; expected {expected}"
            )
        if genes.device != self.template.device:
            raise ValueError("genes and gene layout must be on the same device")
        if genes.dtype != self.base_raw_dofs.dtype:
            raise ValueError("genes and kinematic DOFs must have the same dtype")

        raw_dofs = self.base_raw_dofs.clone()
        raw_dofs[
            self.jump_nodes[:, None],
            torch.arange(6, dtype=torch.int64, device=genes.device)[None, :],
        ] = genes[:, :6]
        raw_torsions = _wrap_angle(
            self.torsion_signs[None, :]
            * (genes[:, 6:] - self.torsion_offsets)
        )
        raw_dofs[
            self.torsion_nodes, self.torsion_dof_types[None, :]
        ] = raw_torsions
        return raw_dofs

    def pose_stack_from_genes(self, genes: torch.Tensor) -> PoseStack:
        """Rebuild all ligand coordinates without copying or changing receptor atoms."""

        raw_dofs = self.raw_dofs_from_genes(genes)
        kin_coords = self.kin_module(raw_dofs)
        coords = self.template.coords.clone()
        pose_indices = torch.arange(
            self.template.n_poses,
            dtype=torch.int64,
            device=self.template.device,
        )
        atom_indices = (
            self.template.block_coord_offset64[
                :, self.docking_dofs.ligand_block, None
            ]
            + self.ligand_local_atoms[None, :]
        )
        coords[pose_indices[:, None], atom_indices] = kin_coords[
            self.ligand_nodes
        ].to(coords.dtype)
        return attr.evolve(self.template, coords=coords)

    def ligand_heavy_coords(self, pose_stack: PoseStack) -> torch.Tensor:
        """Gather ligand heavy atoms as ``[pose, heavy_atom, xyz]``."""

        pose_indices = torch.arange(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        atom_indices = (
            pose_stack.block_coord_offset64[
                :, self.docking_dofs.ligand_block, None
            ]
            + self.heavy_local_atoms[None, :]
        )
        return pose_stack.coords[pose_indices[:, None], atom_indices]


@dataclass(frozen=True)
class GAPool:
    genes: torch.Tensor
    pose_stack: PoseStack
    energies: torch.Tensor
    n_diverse: int
    selected_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class GAResult:
    pool: GAPool
    best_energy_history: torch.Tensor
    diverse_count_history: tuple[int, ...]


def _unit_vectors(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    vectors = torch.randn(
        (count, 3), dtype=dtype, device=device, generator=generator
    )
    return vectors / torch.linalg.vector_norm(
        vectors, dim=1, keepdim=True
    ).clamp_min(torch.finfo(dtype).tiny)


def _canonical_torsions(torsions: torch.Tensor) -> torch.Tensor:
    """Wrap torsions to the unique interval ``(-pi, pi]``."""

    wrapped = torch.remainder(torsions + math.pi, 2 * math.pi) - math.pi
    return torch.where(
        wrapped == -math.pi,
        torch.full_like(wrapped, math.pi),
        wrapped,
    )


def _canonicalize_genes(genes: torch.Tensor) -> torch.Tensor:
    """Wrap only logical torsions; preserve all six freed jump DOFs."""

    if genes.ndim != 2 or genes.shape[1] < 6:
        raise ValueError("genes must have shape [population, 6+n_torsions]")
    return torch.cat(
        (genes[:, :6], _canonical_torsions(genes[:, 6:])),
        dim=1,
    )


def _zyx_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Build tmol's row-vector ZYX rotation matrix from ``[..., 3]`` angles."""

    alpha, beta, gamma = torch.unbind(euler, dim=-1)
    si, sj, sk = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)
    ci, cj, ck = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    return torch.stack(
        (
            cj * ck,
            cj * sk,
            -sj,
            sj * si * ck - ci * sk,
            sj * si * sk + ci * ck,
            cj * si,
            sj * ci * ck + si * sk,
            sj * ci * sk - si * ck,
            cj * ci,
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))


def _rotation_matrix_to_zyx(rotation: torch.Tensor) -> torch.Tensor:
    """Invert ``_zyx_rotation_matrix`` with common.hh's canonical branch."""

    cy = torch.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 0, 1] ** 2)
    singular = cy < 1.0e-6
    alpha = torch.where(
        singular,
        torch.atan2(-rotation[..., 2, 1], rotation[..., 1, 1]),
        torch.atan2(rotation[..., 1, 2], rotation[..., 2, 2]),
    )
    beta = torch.atan2(-rotation[..., 0, 2], cy)
    gamma = torch.where(
        singular,
        torch.zeros_like(cy),
        torch.atan2(rotation[..., 0, 1], rotation[..., 0, 0]),
    )
    return torch.stack((alpha, beta, gamma), dim=-1)


def _quaternion_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Build Rosetta's column-vector rotation matrix from ``[..., w,x,y,z]``."""

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _apply_rosetta_rotation_mutation(
    parent_euler: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Left-compose Rosetta's ``(cos(a), sin(a)*axis)`` perturbation.

    Rosetta rotation matrices act on column vectors, while tmol's kinematics
    stores their transposes for row-vector homogeneous transforms. Composition
    is therefore performed in Rosetta's convention and transposed only at the
    ZYX conversion boundaries.
    """

    perturbation_quaternion = torch.cat(
        (torch.cos(angle)[..., None], torch.sin(angle)[..., None] * axis),
        dim=-1,
    )
    parent_rotation = _zyx_rotation_matrix(parent_euler).transpose(-1, -2)
    perturbation_rotation = _quaternion_rotation_matrix(
        perturbation_quaternion
    )
    composed_rotation = perturbation_rotation @ parent_rotation
    return _rotation_matrix_to_zyx(composed_rotation.transpose(-1, -2))


def mutate(
    genes: torch.Tensor,
    *,
    generator: torch.Generator,
    rtmutation_rate: float = 0.5,
    torsmutation_rate: float = 0.1,
    translation_width: float = 2.0,
    rotation_width: float = math.radians(60.0),
    torsion_width: float = math.radians(120.0),
) -> torch.Tensor:
    """Mutate a gene batch with Rosetta's rigid-body-or-chi semantics.

    ``LigandConformer.cc:765-820`` selects rigid-body mutation with probability
    0.5; otherwise each chi mutates with probability 0.1 and one chi is forced
    when none was selected. Translation has a uniform radius and chi jitter is
    uniform in its width. Rotation draws Rosetta's Gaussian quaternion
    parameter about a random axis and left-composes the perturbation with the
    parent rotation. The composed orientation is written back as raw ZYX
    delta-Euler DOFs for forward kinematics.
    """

    if genes.ndim != 2 or genes.shape[1] < 6:
        raise ValueError("genes must have shape [population, 6+n_torsions]")
    n_genes, n_dof = genes.shape
    device, dtype = genes.device, genes.dtype
    randomize_rt = (
        torch.rand((n_genes,), device=device, generator=generator)
        < rtmutation_rate
    )
    result = genes.clone()

    translation_axis = _unit_vectors(
        n_genes, dtype=dtype, device=device, generator=generator
    )
    translation_length = (
        torch.rand(
            (n_genes, 1), dtype=dtype, device=device, generator=generator
        )
        * translation_width
    )
    result[:, :3] += (
        randomize_rt[:, None] * translation_axis * translation_length
    )

    if rotation_width > math.pi:
        random_values = torch.rand(
            (n_genes, 3), dtype=dtype, device=device, generator=generator
        )
        u1 = random_values[:, 0]
        u2 = 2 * math.pi * random_values[:, 1]
        u3 = 2 * math.pi * random_values[:, 2]
        random_quaternion = torch.stack(
            (
                torch.sqrt(1 - u1) * torch.sin(u2),
                torch.sqrt(1 - u1) * torch.cos(u2),
                torch.sqrt(u1) * torch.sin(u3),
                torch.sqrt(u1) * torch.cos(u3),
            ),
            dim=1,
        )
        random_rotation = _quaternion_rotation_matrix(random_quaternion)
        mutated_rotation = _rotation_matrix_to_zyx(
            random_rotation.transpose(-1, -2)
        )
    else:
        rotation_axis = _unit_vectors(
            n_genes, dtype=dtype, device=device, generator=generator
        )
        angle = rotation_width * torch.randn(
            (n_genes,), dtype=dtype, device=device, generator=generator
        )
        mutated_rotation = _apply_rosetta_rotation_mutation(
            genes[:, 3:6], rotation_axis, angle
        )
    result[:, 3:6] = torch.where(
        randomize_rt[:, None], mutated_rotation, genes[:, 3:6]
    )

    n_torsions = n_dof - 6
    if n_torsions:
        mutate_torsion = (
            torch.rand(
                (n_genes, n_torsions), device=device, generator=generator
            )
            < torsmutation_rate
        ) & ~randomize_rt[:, None]
        force = ~randomize_rt & ~torch.any(mutate_torsion, dim=1)
        forced_index = torch.randint(
            n_torsions, (n_genes, 1), device=device, generator=generator
        )
        forced = torch.zeros_like(mutate_torsion)
        forced.scatter_(1, forced_index, force[:, None])
        mutate_torsion |= forced

        if torsion_width > math.pi:
            mutated_torsions = 2 * math.pi * torch.rand(
                (n_genes, n_torsions),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        else:
            delta = torsion_width * (
                1
                - 2
                * torch.rand(
                    (n_genes, n_torsions),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
            )
            mutated_torsions = genes[:, 6:] + delta
        result[:, 6:] = torch.where(
            mutate_torsion, mutated_torsions, genes[:, 6:]
        )
    return _canonicalize_genes(result)


def crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniform crossover with rigid body inherited as one indivisible unit.

    Rosetta selects the complete rigid body from one parent, then each chi
    independently, forcing one partner-2 chi when all independent choices took
    parent 1 (``LigandConformer.cc:862-897``).
    """

    if parent1.shape != parent2.shape or parent1.ndim != 2:
        raise ValueError("crossover parents must have matching 2D shapes")
    if parent1.shape[1] < 6:
        raise ValueError("crossover genes need a six-value rigid body")
    n_genes, n_dof = parent1.shape
    take_rb_1 = torch.rand(
        (n_genes, 1), device=parent1.device, generator=generator
    ) < 0.5
    result = torch.empty_like(parent1)
    result[:, :6] = torch.where(take_rb_1, parent1[:, :6], parent2[:, :6])

    n_torsions = n_dof - 6
    if n_torsions:
        take_chi_1 = torch.rand(
            (n_genes, n_torsions),
            device=parent1.device,
            generator=generator,
        ) < 0.5
        force_partner2 = torch.all(take_chi_1, dim=1)
        forced_index = torch.randint(
            n_torsions,
            (n_genes, 1),
            device=parent1.device,
            generator=generator,
        )
        forced_partner2 = torch.zeros_like(take_chi_1)
        forced_partner2.scatter_(1, forced_index, force_partner2[:, None])
        take_chi_1 &= ~forced_partner2
        result[:, 6:] = torch.where(
            take_chi_1, parent1[:, 6:], parent2[:, 6:]
        )
    return _canonicalize_genes(result)


def ligand_heavy_atom_rms(
    coords1: torch.Tensor, coords2: torch.Tensor
) -> torch.Tensor:
    """Unsymmetrized heavy-atom coordinate RMS with broadcasting."""

    if coords1.shape[-2:] != coords2.shape[-2:]:
        raise ValueError("ligand coordinate tensors must have matching atom shapes")
    delta = coords1 - coords2
    return torch.sqrt(torch.mean(torch.sum(delta * delta, dim=-1), dim=-1))


def next_generation(
    genes: torch.Tensor,
    npoolout: int,
    pmut: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create offspring by cyclic parent assignment and mutation/crossover.

    This is the batched equivalent of ``GAOptimizer.cc:220-258``. Parent ``i``
    cycles modulo the input pool; mutation is chosen with ``pmut`` and a
    crossover partner is sampled uniformly from every other parent.
    """

    if genes.ndim != 2 or genes.shape[0] < 1:
        raise ValueError("genes must be a nonempty 2D population")
    if not isinstance(npoolout, int) or isinstance(npoolout, bool) or npoolout < 1:
        raise ValueError("npoolout must be a positive integer")
    if not math.isfinite(pmut) or not 0.0 <= pmut <= 1.0:
        raise ValueError("pmut must be between zero and one")
    npoolin = genes.shape[0]
    parent_indices = torch.arange(
        npoolout, dtype=torch.int64, device=genes.device
    ) % npoolin
    parents = genes[parent_indices]
    mutate_choice = torch.rand(
        (npoolout,), device=genes.device, generator=generator
    ) <= pmut
    if npoolin == 1:
        mutate_choice = torch.ones_like(mutate_choice)
        partners = parents
    else:
        partner_indices = torch.randint(
            npoolin - 1,
            (npoolout,),
            device=genes.device,
            generator=generator,
        )
        partner_indices += (partner_indices >= parent_indices).to(torch.int64)
        partners = genes[partner_indices]
    mutated = mutate(parents, generator=generator)
    crossed = crossover(parents, partners, generator=generator)
    return torch.where(mutate_choice[:, None], mutated, crossed)


def optimize_generation(
    genes: torch.Tensor,
    layout: LigandGeneLayout,
    backend: ScoringBackend,
    *,
    maxiter: int,
) -> GAPool:
    """Score and locally minimize one whole population as a batch.

    This mirrors the score-then-optimize operation at
    ``GAOptimizer.cc:195-208`` while delegating the batched objective to the
    Phase 2 backend.
    """

    proposed = layout.pose_stack_from_genes(genes)
    proposed_raw_dofs = layout.raw_dofs_from_genes(genes)
    premin = backend.score(proposed)
    expected = (genes.shape[0],)
    if premin.shape != expected:
        raise ValueError(
            f"backend.score returned {tuple(premin.shape)}; expected {expected}"
        )
    minimize_result = backend.minimize(
        proposed,
        layout.docking_dofs,
        maxiter=maxiter,
        initial_raw_dofs=proposed_raw_dofs,
    )
    if not isinstance(minimize_result, MinimizationResult):
        raise TypeError(
            "GA backends must return MinimizationResult with optimizer genes"
        )
    minimized = minimize_result.pose_stack
    minimized_genes = minimize_result.genes
    if minimized_genes is None:
        raise ValueError("GA minimization did not return a freed-DOF gene")
    if minimized_genes.shape != genes.shape:
        raise ValueError(
            "backend.minimize returned genes with shape "
            f"{tuple(minimized_genes.shape)}; expected {tuple(genes.shape)}"
        )
    if minimized_genes.device != genes.device:
        raise ValueError("minimized genes moved to a different device")
    if minimized_genes.dtype != genes.dtype:
        raise ValueError("minimized genes changed dtype")
    energies = backend.score(minimized)
    if energies.shape != expected:
        raise ValueError(
            f"backend.score returned {tuple(energies.shape)}; expected {expected}"
        )
    return GAPool(
        genes=minimized_genes.detach(),
        pose_stack=minimized,
        energies=energies.detach(),
        n_diverse=genes.shape[0],
    )


def update_pool(
    parents: GAPool,
    offspring: GAPool,
    layout: LigandGeneLayout,
    *,
    npool: int,
    rmsdthreshold: float,
) -> GAPool:
    """Greedily retain energy-ranked RMS-diverse genes, then fill if needed.

    Rosetta concatenates offspring and parents, sorts by energy, greedily keeps
    candidates whose heavy-atom RMS is not below the cutoff, and finally fills
    any missing slots with rejected candidates (``GAOptimizer.cc:268-313``).
    Consequently ``n_diverse < npool`` documents a filled pool that cannot
    satisfy the cutoff, matching Rosetta rather than returning a short pool.
    """

    genes = torch.cat((offspring.genes, parents.genes), dim=0)
    energies = torch.cat((offspring.energies, parents.energies), dim=0)
    # RMS niching is defined by the genes themselves. Rebuild both populations
    # with forward kinematics so no coordinate-to-gene inversion participates
    # in the distance calculation.
    offspring_from_genes = layout.pose_stack_from_genes(offspring.genes)
    parents_from_genes = layout.pose_stack_from_genes(parents.genes)
    coords = torch.cat(
        (
            layout.ligand_heavy_coords(offspring_from_genes),
            layout.ligand_heavy_coords(parents_from_genes),
        ),
        dim=0,
    )
    if genes.shape[0] <= npool:
        raise ValueError("combined parent and offspring pools must exceed npool")
    order = torch.argsort(energies)
    distances = ligand_heavy_atom_rms(coords[:, None], coords[None, :])

    selected: list[int] = []
    selected_mask = torch.zeros(
        genes.shape[0], dtype=torch.bool, device=genes.device
    )
    for candidate in order.detach().cpu().tolist():
        if not selected:
            selected.append(candidate)
            selected_mask[candidate] = True
        else:
            sufficiently_distant = (
                ~selected_mask
                | (distances[candidate] >= rmsdthreshold)
            )
            if bool(torch.all(sufficiently_distant)):
                selected.append(candidate)
                selected_mask[candidate] = True
        if len(selected) == npool:
            break
    n_diverse = len(selected)

    selected_tensor = torch.tensor(
        selected, dtype=torch.int64, device=genes.device
    )
    if n_diverse < npool:
        rejected_in_energy_order = order[~selected_mask[order]]
        selected_tensor = torch.cat(
            (selected_tensor, rejected_in_energy_order[: npool - n_diverse])
        )

    candidate_coords = torch.cat(
        (offspring.pose_stack.coords, parents.pose_stack.coords), dim=0
    )
    selected_pose_stack = attr.evolve(
        parents.pose_stack, coords=candidate_coords[selected_tensor]
    )
    return GAPool(
        genes=genes[selected_tensor],
        pose_stack=selected_pose_stack,
        energies=energies[selected_tensor],
        n_diverse=n_diverse,
        selected_indices=selected_tensor,
    )


def _initial_genes(
    layout: LigandGeneLayout,
    perturbation: DockingPerturbation,
    generator: torch.Generator,
) -> torch.Tensor:
    genes = layout.genes_from_pose_stack(layout.template)
    unit_jitter = 2 * torch.rand(
        genes.shape,
        dtype=genes.dtype,
        device=genes.device,
        generator=generator,
    ) - 1
    jitter_rows = torch.arange(genes.shape[0], device=genes.device) > 0
    result = genes.clone()
    result[:, :3] += (
        jitter_rows[:, None]
        * unit_jitter[:, :3]
        * perturbation.translation_angstrom
    )
    result[:, 3:6] += (
        jitter_rows[:, None]
        * unit_jitter[:, 3:6]
        * perturbation.rotation_radians
    )
    result[:, 6:] += (
        jitter_rows[:, None]
        * unit_jitter[:, 6:]
        * perturbation.torsion_radians
    )
    return _canonicalize_genes(result)


def run_ga_docking(
    pose_stack: PoseStack,
    ligand_block: int,
    backend: ScoringBackend,
    *,
    schedule: GASchedule = GASchedule(),
    seed: int = 0,
) -> GAResult:
    """Run the vectorized continuous GA from an initially perturbed pool."""

    if pose_stack.n_poses != 1:
        raise ValueError("pose_stack must contain one reference pose")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    population = _repeat_pose_stack(pose_stack, schedule.npool)
    docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    generator = torch.Generator(device=pose_stack.device)
    generator.manual_seed(seed)

    genes = _initial_genes(
        layout, schedule.initial_perturbation, generator
    )
    pool = optimize_generation(
        genes, layout, backend, maxiter=schedule.maxiter
    )
    best_history = [torch.min(pool.energies)]
    diverse_history: list[int] = []

    # Rosetta GAOptimizer.cc:104-113.
    for _ in range(schedule.generations):
        offspring_genes = next_generation(
            pool.genes,
            schedule.npool,
            schedule.pmut,
            generator=generator,
        )
        offspring = optimize_generation(
            offspring_genes, layout, backend, maxiter=schedule.maxiter
        )
        pool = update_pool(
            pool,
            offspring,
            layout,
            npool=schedule.npool,
            rmsdthreshold=schedule.rmsdthreshold,
        )
        best_history.append(torch.min(pool.energies))
        diverse_history.append(pool.n_diverse)

    return GAResult(
        pool=pool,
        best_energy_history=torch.stack(best_history),
        diverse_count_history=tuple(diverse_history),
    )


__all__ = [
    "GAPool",
    "GAResult",
    "GASchedule",
    "LigandGeneLayout",
    "crossover",
    "ligand_heavy_atom_rms",
    "mutate",
    "next_generation",
    "optimize_generation",
    "run_ga_docking",
    "update_pool",
]
