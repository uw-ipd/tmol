"""Grid-guided rigid-body ligand placement for GA initialization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tmol.dock._diversity import select_diverse_low_energy

if TYPE_CHECKING:
    from tmol.dock.ga import LigandGeneLayout
    from tmol.dock.scoring_backend import GridBackend


@dataclass(frozen=True)
class LigandAlignment:
    """Diverse, energy-ranked placements of one fixed ligand conformer.

    Coordinates have shape ``[placement, ligand_atom, 3]``; energies and
    candidate indices have shape ``[placement]``.
    """

    coordinates: torch.Tensor
    energies: torch.Tensor
    candidate_indices: torch.Tensor
    rotations: torch.Tensor
    centroids: torch.Tensor


@dataclass(frozen=True)
class LigandAligner:
    """Sample and rank fixed-conformer rigid placements against a receptor grid.

    Candidate centroids and uniform SO(3) rotations are sampled for one ligand,
    then scored and filtered for placement diversity.
    """

    n_candidates: int = 4000
    seed_fraction: float = 0.5
    cluster_rmsd: float = 2.0
    diversity_pool_size: int = 512
    score_batch_size: int = 4096
    refinement_steps: int = 0
    refinement_top_k: int = 64
    refinement_learning_rate: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "n_candidates",
            "diversity_pool_size",
            "score_batch_size",
            "refinement_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.refinement_steps, int)
            or isinstance(self.refinement_steps, bool)
            or self.refinement_steps < 0
        ):
            raise ValueError("refinement_steps must be a nonnegative integer")
        if (
            not math.isfinite(self.seed_fraction)
            or not 0.0 <= self.seed_fraction <= 1.0
        ):
            raise ValueError("seed_fraction must be between zero and one")
        if not math.isfinite(self.cluster_rmsd) or self.cluster_rmsd < 0.0:
            raise ValueError("cluster_rmsd must be finite and nonnegative")
        if (
            not math.isfinite(self.refinement_learning_rate)
            or self.refinement_learning_rate <= 0.0
        ):
            raise ValueError("refinement_learning_rate must be finite and positive")

    def align(
        self,
        layout: "LigandGeneLayout",
        backend: "GridBackend",
        *,
        n_placements: int,
        pocket_bounds: torch.Tensor | None,
        generator: torch.Generator,
    ) -> LigandAlignment:
        """Return ``n_placements`` rigid placements of the reference ligand."""

        if (
            not isinstance(n_placements, int)
            or isinstance(n_placements, bool)
            or n_placements < 1
        ):
            raise ValueError("n_placements must be a positive integer")
        if n_placements > self.n_candidates:
            raise ValueError("n_placements cannot exceed n_candidates")
        template_coordinates = layout.template.coords[0]
        ligand_atom_indices = (
            layout.template.block_coord_offset64[0, layout.docking_dofs.ligand_block]
            + layout.ligand_local_atoms
        )
        heavy_atom_indices = layout.heavy_local_atoms
        bounds = self._pocket_bounds(
            backend,
            template_coordinates,
            ligand_atom_indices,
            heavy_atom_indices,
            pocket_bounds,
        )
        preselect_count = min(
            self.n_candidates,
            max(
                n_placements,
                self.diversity_pool_size,
                self.refinement_top_k if self.refinement_steps else 0,
            ),
        )
        unit_positions, rotations = self._sample_transforms(bounds, generator)
        (
            preselected_centroids,
            preselected_rotations,
            preselected_energies,
            preselect_indices,
        ) = self._score_and_preselect(
            backend,
            template_coordinates,
            ligand_atom_indices,
            heavy_atom_indices,
            bounds,
            unit_positions,
            rotations,
            preselect_count,
        )
        preselected_ligand = self._placed_ligand_coordinates(
            template_coordinates,
            ligand_atom_indices,
            heavy_atom_indices,
            preselected_centroids,
            preselected_rotations,
        )

        if self.refinement_steps:
            n_refine = min(self.refinement_top_k, preselect_count)
            refinement_coordinates = self._candidate_coordinates(
                template_coordinates,
                ligand_atom_indices,
                preselected_ligand[:n_refine],
            )
            refined = self._refine_candidates(
                backend, refinement_coordinates, ligand_atom_indices
            )
            refined_energies = self._score_candidates(backend, refined)
            refined_ligand = _gather_ligand_coordinates(refined, ligand_atom_indices)
            template_ligand = template_coordinates[ligand_atom_indices]
            refined_rotations = _fit_rigid_transforms(
                template_ligand[None], refined_ligand
            )
            refined_centroids = refined_ligand.index_select(1, heavy_atom_indices).mean(
                dim=1
            )
            preselected_ligand = torch.cat((refined_ligand, preselected_ligand), dim=0)
            preselected_rotations = torch.cat(
                (refined_rotations, preselected_rotations), dim=0
            )
            preselected_centroids = torch.cat(
                (refined_centroids, preselected_centroids), dim=0
            )
            preselected_energies = torch.cat(
                (refined_energies, preselected_energies), dim=0
            )
            preselect_indices = torch.cat(
                (preselect_indices[:n_refine], preselect_indices), dim=0
            )

        selected = _select_diverse_indices(
            preselected_ligand.index_select(1, heavy_atom_indices),
            preselected_energies,
            n_placements,
            self.cluster_rmsd,
        )
        return LigandAlignment(
            coordinates=preselected_ligand[selected],
            energies=preselected_energies[selected],
            candidate_indices=preselect_indices[selected],
            rotations=preselected_rotations[selected],
            centroids=preselected_centroids[selected],
        )

    def seed_genes(
        self,
        layout: "LigandGeneLayout",
        backend: "GridBackend",
        random_genes: torch.Tensor,
        *,
        pocket_bounds: torch.Tensor | None,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Replace a fraction of a blind population with rigid aligner seeds.

        Row zero remains the reference state. Aligner placements replace
        the complete fixed-template gene for those rows. Unseeded rows retain
        their blind-random rigid, torsional, and receptor-sidechain genes.
        """

        if random_genes.shape != (layout.template.n_poses, layout.n_dof):
            raise ValueError("random genes do not match the gene layout")
        if random_genes.device != layout.template.device:
            raise ValueError("random genes and gene layout must share a device")
        if random_genes.dtype != layout.base_raw_dofs.dtype:
            raise ValueError("random genes and kinematic DOFs must share a dtype")
        npool = random_genes.shape[0]
        n_seeded = min(int(math.floor(self.seed_fraction * npool)), npool - 1)
        result = random_genes.clone()
        reference = layout.genes_from_pose_stack(layout.template)
        result[0] = reference[0]
        if n_seeded == 0:
            return result

        alignment = self.align(
            layout,
            backend,
            n_placements=n_seeded,
            pocket_bounds=pocket_bounds,
            generator=generator,
        )
        seeded_rows = torch.arange(
            1, n_seeded + 1, dtype=torch.int64, device=result.device
        )
        from tmol.dock.ga import _rotation_matrix_to_zyx

        result[seeded_rows] = reference[seeded_rows]
        result[seeded_rows, 3:6] = _rotation_matrix_to_zyx(
            alignment.rotations.transpose(-1, -2)
        )
        oriented = layout.pose_stack_from_genes(result)
        current_centroids = (
            layout.ligand_heavy_coords(oriented)
            .index_select(0, seeded_rows)
            .mean(dim=1)
        )
        result[seeded_rows, :3] += alignment.centroids - current_centroids
        return result

    def _pocket_bounds(
        self,
        backend: "GridBackend",
        template_coordinates: torch.Tensor,
        ligand_atom_indices: torch.Tensor,
        heavy_atom_indices: torch.Tensor,
        pocket_bounds: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype, device = template_coordinates.dtype, template_coordinates.device
        if pocket_bounds is not None:
            bounds = torch.as_tensor(pocket_bounds, dtype=dtype, device=device)
            if bounds.shape != (2, 3):
                raise ValueError("pocket_bounds must have shape [2, 3]")
        else:
            geometry = backend._scorer(device, dtype).grids.geometry
            grid_max = (
                geometry.origin
                + geometry.origin.new_tensor(geometry.dims).sub(1)
                * geometry.voxel_spacing
            )
            ligand = template_coordinates[ligand_atom_indices]
            heavy = ligand.index_select(0, heavy_atom_indices)
            centroid = heavy.mean(dim=0)
            radius = torch.linalg.vector_norm(ligand - centroid, dim=1).amax()
            lower = geometry.origin + radius
            upper = grid_max - radius
            grid_center = 0.5 * (geometry.origin + grid_max)
            invalid = lower > upper
            lower = torch.where(invalid, grid_center, lower)
            upper = torch.where(invalid, grid_center, upper)
            bounds = torch.stack((lower, upper))
        if not bool(torch.all(torch.isfinite(bounds))) or not bool(
            torch.all(bounds[1] >= bounds[0])
        ):
            raise ValueError("pocket bounds must be finite with max >= min")
        return bounds

    def _sample_transforms(
        self,
        bounds: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tmol.dock.ga import _quaternion_rotation_matrix

        dtype, device = bounds.dtype, bounds.device
        unit_position = torch.rand(
            (self.n_candidates, 3),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        random_values = torch.rand(
            (self.n_candidates, 3),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        u1, u2, u3 = torch.unbind(random_values, dim=1)
        u2 = 2 * math.pi * u2
        u3 = 2 * math.pi * u3
        quaternion = torch.stack(
            (
                torch.sqrt(u1) * torch.cos(u3),
                torch.sqrt(1 - u1) * torch.sin(u2),
                torch.sqrt(1 - u1) * torch.cos(u2),
                torch.sqrt(u1) * torch.sin(u3),
            ),
            dim=1,
        )
        rotation = _quaternion_rotation_matrix(quaternion)
        return unit_position, rotation

    def _placed_ligand_coordinates(
        self,
        template_coordinates: torch.Tensor,
        ligand_atom_indices: torch.Tensor,
        heavy_atom_indices: torch.Tensor,
        target_centroids: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        ligand = template_coordinates[ligand_atom_indices]
        heavy = ligand.index_select(0, heavy_atom_indices)
        centroid = heavy.mean(dim=0)
        centered = ligand - centroid
        return (
            torch.einsum("ae,mde->mad", centered, rotation) + target_centroids[:, None]
        )

    def _candidate_coordinates(
        self,
        template_coordinates: torch.Tensor,
        ligand_atom_indices: torch.Tensor,
        placed_ligand: torch.Tensor,
    ) -> torch.Tensor:
        candidates = (
            template_coordinates[None].expand(placed_ligand.shape[0], -1, -1).clone()
        )
        candidates[:, ligand_atom_indices] = placed_ligand
        return candidates

    def _score_and_preselect(
        self,
        backend: "GridBackend",
        template_coordinates: torch.Tensor,
        ligand_atom_indices: torch.Tensor,
        heavy_atom_indices: torch.Tensor,
        bounds: torch.Tensor,
        unit_positions: torch.Tensor,
        rotations: torch.Tensor,
        preselect_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_batch_size = self.score_batch_size
        selected_centroids = None
        selected_rotations = None
        selected_energies = None
        selected_indices = None
        for start in range(0, self.n_candidates, candidate_batch_size):
            end = min(start + candidate_batch_size, self.n_candidates)
            target_centroids = bounds[0] + unit_positions[start:end] * (
                bounds[1] - bounds[0]
            )
            batch_rotations = rotations[start:end]
            placed_ligand = self._placed_ligand_coordinates(
                template_coordinates,
                ligand_atom_indices,
                heavy_atom_indices,
                target_centroids,
                batch_rotations,
            )
            coordinates = self._candidate_coordinates(
                template_coordinates,
                ligand_atom_indices,
                placed_ligand,
            )
            energies = self._score_candidates(backend, coordinates)
            indices = torch.arange(
                start, end, dtype=torch.int64, device=energies.device
            )
            if selected_centroids is not None:
                target_centroids = torch.cat(
                    (selected_centroids, target_centroids), dim=0
                )
                batch_rotations = torch.cat(
                    (selected_rotations, batch_rotations), dim=0
                )
                energies = torch.cat((selected_energies, energies), dim=0)
                indices = torch.cat((selected_indices, indices), dim=0)
            keep_count = min(preselect_count, energies.shape[0])
            keep = torch.argsort(energies, stable=True)[:keep_count]
            selected_centroids = target_centroids[keep]
            selected_rotations = batch_rotations[keep]
            selected_energies = energies[keep]
            selected_indices = indices[keep]
        if (
            selected_centroids is None
            or selected_rotations is None
            or selected_energies is None
            or selected_indices is None
        ):
            raise RuntimeError("aligner candidate sampling produced no placements")
        return (
            selected_centroids,
            selected_rotations,
            selected_energies,
            selected_indices,
        )

    def _score_candidates(
        self, backend: "GridBackend", coordinates: torch.Tensor
    ) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, coordinates.shape[0], self.score_batch_size):
                chunks.append(
                    backend._score_ligand_coordinates(
                        coordinates[start : start + self.score_batch_size]
                    )
                )
        return torch.cat(chunks)

    def _refine_candidates(
        self,
        backend: "GridBackend",
        coordinates: torch.Tensor,
        ligand_atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        flat = coordinates.detach()
        flat_ligand_indices = ligand_atom_indices[None].expand(flat.shape[0], -1)
        base_ligand = _gather_ligand_coordinates(flat, ligand_atom_indices)
        center = base_ligand.mean(dim=1)
        centered = base_ligand - center[:, None]
        translation = torch.zeros_like(center, requires_grad=True)
        rotation_vector = torch.zeros_like(center, requires_grad=True)
        optimizer = torch.optim.Adam(
            (translation, rotation_vector), lr=self.refinement_learning_rate
        )
        for _ in range(self.refinement_steps):
            optimizer.zero_grad(set_to_none=True)
            rotation = _axis_angle_rotation_matrix(rotation_vector)
            placed = torch.einsum("nae,nde->nad", centered, rotation)
            placed = placed + center[:, None] + translation[:, None]
            refined = flat.clone()
            refined.scatter_(
                1,
                flat_ligand_indices[:, :, None].expand(-1, -1, 3),
                placed,
            )
            backend._score_ligand_coordinates(refined).sum().backward()
            optimizer.step()
        with torch.no_grad():
            rotation = _axis_angle_rotation_matrix(rotation_vector)
            placed = torch.einsum("nae,nde->nad", centered, rotation)
            placed = placed + center[:, None] + translation[:, None]
            refined = flat.clone()
            refined.scatter_(
                1,
                flat_ligand_indices[:, :, None].expand(-1, -1, 3),
                placed,
            )
        return refined.reshape_as(coordinates)


def _axis_angle_rotation_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Return column-vector Rodrigues matrices for axis-angle vectors."""

    theta_squared = torch.sum(rotation_vector * rotation_vector, dim=-1)
    small = theta_squared < 1.0e-8
    safe_theta_squared = theta_squared.clamp_min(1.0e-16)
    theta = torch.sqrt(safe_theta_squared)
    sine_scale = torch.where(
        small,
        1 - theta_squared / 6 + theta_squared * theta_squared / 120,
        torch.sin(theta) / theta,
    )
    cosine_scale = torch.where(
        small,
        0.5 - theta_squared / 24 + theta_squared * theta_squared / 720,
        (1 - torch.cos(theta)) / safe_theta_squared,
    )
    x, y, z = torch.unbind(rotation_vector, dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        rotation_vector.shape[:-1] + (3, 3)
    )
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device
    ).expand_as(skew)
    return (
        identity
        + sine_scale[..., None, None] * skew
        + cosine_scale[..., None, None] * (skew @ skew)
    )


def _gather_ligand_coordinates(
    coordinates: torch.Tensor,
    ligand_atom_indices: torch.Tensor,
) -> torch.Tensor:
    return coordinates.index_select(1, ligand_atom_indices)


def _fit_rigid_transforms(
    source: torch.Tensor, destination: torch.Tensor
) -> torch.Tensor:
    """Return column-vector Kabsch rotations mapping source to destination."""

    source_centered = source - source.mean(dim=-2, keepdim=True)
    destination_centered = destination - destination.mean(dim=-2, keepdim=True)
    covariance = source_centered.transpose(-1, -2) @ destination_centered
    u, _, vh = torch.linalg.svd(covariance)
    determinant = torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    correction = torch.ones(
        determinant.shape + (3,), dtype=source.dtype, device=source.device
    )
    correction[..., -1] = torch.sign(determinant)
    return vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)


def _select_diverse_indices(
    heavy_coordinates: torch.Tensor,
    energies: torch.Tensor,
    n_select: int,
    rmsd_threshold: float,
) -> torch.Tensor:
    """Energy-ranked RMS-diverse placements, returned lowest energy first."""

    selected, _ = select_diverse_low_energy(
        energies, heavy_coordinates, npool=n_select, rmsdthreshold=rmsd_threshold
    )
    return selected[torch.argsort(energies[selected], stable=True)]


__all__ = ["LigandAligner", "LigandAlignment"]
