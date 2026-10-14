"""Live ligand--receptor LK-ball terms for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from tmol.database import ParameterDatabase
from tmol.pose.pose_stack import PoseStack
from tmol.score.common.convert_float64 import convert_float64
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm

MAX_N_WATER = 4


@dataclass(frozen=True)
class LiveLKBallData:
    """One shared rigid-receptor LK-ball structure and ligand topology."""

    n_ligand_atoms: int
    ligand_atom_params: torch.Tensor
    ligand_heavy_atoms: torch.Tensor
    ligand_polar_atoms: torch.Tensor
    ligand_atom_to_polar: torch.Tensor
    ligand_water_kinds: torch.Tensor
    ligand_water_atoms: torch.Tensor
    ligand_water_params: torch.Tensor
    ligand_water_is_ring: torch.Tensor
    receptor_coords: torch.Tensor
    receptor_params: torch.Tensor
    receptor_waters: torch.Tensor
    receptor_polar_atoms: torch.Tensor
    cell_origin: torch.Tensor
    cell_dims: torch.Tensor
    receptor_cell_offsets: torch.Tensor
    receptor_cell_indices: torch.Tensor
    receptor_polar_cell_offsets: torch.Tensor
    receptor_polar_cell_indices: torch.Tensor
    global_params: torch.Tensor

    def __post_init__(self):
        if (
            not isinstance(self.n_ligand_atoms, int)
            or isinstance(self.n_ligand_atoms, bool)
            or self.n_ligand_atoms < 1
        ):
            raise ValueError("n_ligand_atoms must be a positive integer")
        tensors = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "n_ligand_atoms"
        }
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("all live LK-ball arrays must be tensors")
        if len({value.device for value in tensors.values()}) != 1:
            raise ValueError("all live LK-ball arrays must share one device")
        if not all(value.is_contiguous() for value in tensors.values()):
            raise ValueError("all live LK-ball arrays must be contiguous")

        real_dtype = self.receptor_coords.dtype
        if real_dtype not in (torch.float32, torch.float64):
            raise TypeError("live LK-ball coordinates must be float32 or float64")
        real_names = {
            "ligand_atom_params",
            "ligand_water_params",
            "receptor_coords",
            "receptor_params",
            "receptor_waters",
            "cell_origin",
            "global_params",
        }
        if any(tensors[name].dtype != real_dtype for name in real_names):
            raise TypeError("live LK-ball real arrays must share one dtype")
        if any(
            value.dtype != torch.int64
            for name, value in tensors.items()
            if name not in real_names
        ):
            raise TypeError("live LK-ball index arrays must be int64")

        n_heavy = self.receptor_coords.shape[0]
        n_polar = self.receptor_polar_atoms.shape[0]
        n_ligand_polar = self.ligand_polar_atoms.shape[0]
        expected_shapes = {
            "ligand_atom_params": (self.n_ligand_atoms, 9),
            "ligand_heavy_atoms": (self.ligand_heavy_atoms.shape[0],),
            "ligand_polar_atoms": (n_ligand_polar,),
            "ligand_atom_to_polar": (self.n_ligand_atoms,),
            "ligand_water_kinds": (n_ligand_polar, MAX_N_WATER),
            "ligand_water_atoms": (n_ligand_polar, MAX_N_WATER, 3),
            "ligand_water_params": (n_ligand_polar, MAX_N_WATER, 3),
            "ligand_water_is_ring": (n_ligand_polar, MAX_N_WATER),
            "receptor_coords": (n_heavy, 3),
            "receptor_params": (n_heavy, 9),
            "receptor_waters": (n_heavy, MAX_N_WATER, 3),
            "receptor_polar_atoms": (n_polar,),
            "cell_origin": (3,),
            "cell_dims": (3,),
            "global_params": (1, 5),
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")

        if self.receptor_cell_offsets.ndim != 1:
            raise ValueError("receptor cell offsets must be one-dimensional")
        n_cells = self.receptor_cell_offsets.shape[0] - 1
        cell_shapes = {
            "receptor_cell_offsets": (n_cells + 1,),
            "receptor_cell_indices": (n_heavy,),
            "receptor_polar_cell_offsets": (n_cells + 1,),
            "receptor_polar_cell_indices": (n_polar,),
        }
        for name, expected in cell_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")

        if self.receptor_coords.device.type != "cpu":
            return
        if bool(
            torch.any(self.cell_dims <= 0) or int(torch.prod(self.cell_dims)) != n_cells
        ):
            raise ValueError("cell_dims must be positive and match cell offsets")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                self.ligand_atom_params,
                self.ligand_water_params,
                self.receptor_coords,
                self.receptor_params,
                self.cell_origin,
                self.global_params,
            )
        ):
            raise ValueError("live LK-ball coordinates and parameters must be finite")
        cutoff = float(self.global_params[0, 4])
        if not cutoff > 0:
            raise ValueError("live LK-ball cutoff must be positive")

        def check_range(value: torch.Tensor, upper: int, name: str) -> None:
            if value.numel() and bool(
                torch.any(value < 0) or torch.any(value >= upper)
            ):
                raise ValueError(f"{name} contains an out-of-range index")

        check_range(self.ligand_heavy_atoms, self.n_ligand_atoms, "ligand_heavy_atoms")
        check_range(self.ligand_polar_atoms, self.n_ligand_atoms, "ligand_polar_atoms")
        check_range(self.receptor_polar_atoms, n_heavy, "receptor_polar_atoms")
        if bool(torch.any((self.ligand_atom_to_polar < -1))):
            raise ValueError("ligand_atom_to_polar contains an invalid index")
        active_polar = self.ligand_atom_to_polar >= 0
        check_range(
            self.ligand_atom_to_polar[active_polar],
            n_ligand_polar,
            "ligand_atom_to_polar",
        )
        if n_ligand_polar and not torch.equal(
            self.ligand_atom_to_polar[self.ligand_polar_atoms],
            torch.arange(n_ligand_polar, dtype=torch.int64),
        ):
            raise ValueError("ligand polar lookup does not match ligand_polar_atoms")
        if bool(
            torch.any((self.ligand_water_kinds < 0) | (self.ligand_water_kinds > 2))
        ):
            raise ValueError("ligand_water_kinds must be 0, 1, or 2")
        active_water = self.ligand_water_kinds != 0
        if bool(torch.any(self.ligand_water_atoms[~active_water] != -1)):
            raise ValueError("inactive ligand water atoms must be -1")
        check_range(
            self.ligand_water_atoms[active_water][:, :2],
            self.n_ligand_atoms,
            "ligand_water_atoms",
        )
        acceptor_water = self.ligand_water_kinds == 2
        check_range(
            self.ligand_water_atoms[acceptor_water][:, 2],
            self.n_ligand_atoms,
            "ligand acceptor water atoms",
        )

        for kind, offsets, indices, expected_count, upper in (
            (
                "receptor",
                self.receptor_cell_offsets,
                self.receptor_cell_indices,
                n_heavy,
                n_heavy,
            ),
            (
                "receptor polar",
                self.receptor_polar_cell_offsets,
                self.receptor_polar_cell_indices,
                n_polar,
                n_heavy,
            ),
        ):
            if (
                int(offsets[0]) != 0
                or int(offsets[-1]) != expected_count
                or bool(torch.any(offsets[1:] < offsets[:-1]))
            ):
                raise ValueError(f"{kind} cell offsets are not valid CSR")
            check_range(indices, upper, f"{kind}_cell_indices")
        if n_heavy and not torch.equal(
            torch.sort(self.receptor_cell_indices).values,
            torch.arange(n_heavy, dtype=torch.int64),
        ):
            raise ValueError("receptor cell indices must be a permutation")
        if n_polar and not torch.equal(
            torch.sort(self.receptor_polar_cell_indices).values,
            torch.sort(self.receptor_polar_atoms).values,
        ):
            raise ValueError("receptor polar cell indices must match polar atoms")

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "LiveLKBallData":
        """Return the structure on a scoring device and coordinate dtype."""

        if dtype not in (torch.float32, torch.float64):
            raise TypeError("live LK-ball dtype must be float32 or float64")
        device = torch.device(device)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, torch.Tensor):
                values[field.name] = value
            elif value.dtype.is_floating_point:
                values[field.name] = value.to(device=device, dtype=dtype).contiguous()
            else:
                values[field.name] = value.to(device=device).contiguous()
        return LiveLKBallData(**values)


def _build_cell_list(
    anchors: torch.Tensor,
    source_indices: torch.Tensor,
    origin: torch.Tensor,
    dims: torch.Tensor,
    cell_width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_cells = int(torch.prod(dims))
    if anchors.shape[0] == 0:
        return (
            torch.zeros(n_cells + 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    cells = torch.floor((anchors - origin) / cell_width).to(torch.int64)
    if bool(torch.any(cells < 0) or torch.any(cells >= dims)):
        raise RuntimeError("receptor LK-ball atom fell outside its cell structure")
    linear = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    order = torch.argsort(linear, stable=True)
    counts = torch.bincount(linear, minlength=n_cells)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return offsets.contiguous(), source_indices[order].to(torch.int64).contiguous()


def _pack_type_params(term: LKBallEnergyTerm) -> torch.Tensor:
    params = term.ljlk_param_resolver.type_params
    # Match LKBallTypeParams: these columns hold precomputed factors, not the
    # raw dgfree/lambda values. Compute in the live scorer's float64 precision.
    lk_dgfree = params.lk_dgfree.to(dtype=torch.float64)
    lk_lambda = params.lk_lambda.to(dtype=torch.float64)
    return torch.stack(
        tuple(
            value.to(dtype=torch.float64)
            for value in (
                params.lj_radius,
                -lk_dgfree / (2 * 5.56832799683 * lk_lambda),
                1 / (lk_lambda * lk_lambda),
                params.lk_volume,
                params.is_donor,
                params.is_hydroxyl,
                params.is_polarh,
                params.is_acceptor,
                params.is_carbon_lk,
            )
        ),
        dim=1,
    ).contiguous()


def _generate_reference_waters(
    pose_stack: PoseStack,
    term: LKBallEnergyTerm,
    *,
    water_distance: float | None = None,
) -> torch.Tensor:
    from tmol.score.lk_ball.potentials.compiled import gen_pose_waters

    pbt = pose_stack.packed_block_types
    water_gen_global_params = term._lk_ball_water_gen_global_params
    if water_distance is not None:
        # Assign the Rosetta grid distance after promotion so fp64 grid builds
        # do not inherit a rounded float32 representation of 2.65 A.
        water_gen_global_params = water_gen_global_params.to(torch.float64).clone()
        water_gen_global_params[:, 0] = water_distance
    args = [
        pose_stack.coords.reshape(-1, 3).to(torch.float64).contiguous(),
        pose_stack.rot_coord_offset,
        pose_stack.pose_ind_for_atom,
        pose_stack.first_rot_for_block,
        pose_stack.first_rot_block_type,
        pose_stack.block_ind_for_rot,
        pose_stack.pose_ind_for_rot,
        pose_stack.block_type_ind_for_rot,
        pose_stack.n_rots_for_pose,
        pose_stack.rot_offset_for_pose,
        pose_stack.n_rots_for_block,
        pose_stack.rot_offset_for_block,
        torch.full_like(pose_stack.block_type_ind, -1),
        pose_stack.max_n_rots_per_pose,
        pose_stack.inter_residue_connections,
        pbt.n_atoms,
        pbt.n_conn,
        pbt.conn_atom,
        pbt.n_all_bonds,
        pbt.all_bonds,
        pbt.atom_all_bond_ranges,
        pbt.hbpbt_params.tile_n_donH,
        pbt.hbpbt_params.tile_n_acc,
        pbt.hbpbt_params.tile_donH_inds,
        pbt.hbpbt_params.tile_donH_hvy_inds,
        pbt.hbpbt_params.tile_which_donH_of_donH_hvy,
        pbt.hbpbt_params.tile_acc_inds,
        pbt.hbpbt_params.tile_acceptor_hybridization,
        pbt.hbpbt_params.tile_acceptor_n_attached_H,
        pbt.hbpbt_params.is_hydrogen,
        water_gen_global_params,
        term.ljlk_param_resolver.global_params.lkb_water_tors_sp2,
        term.ljlk_param_resolver.global_params.lkb_water_tors_sp3,
        term.ljlk_param_resolver.global_params.lkb_water_tors_ring,
    ]
    convert_float64(args)
    with torch.no_grad():
        return gen_pose_waters(*args).detach().to(torch.float64).contiguous()


def _derived_atom_indices(pose_stack: PoseStack) -> torch.Tensor:
    from tmol.score.hbond.potentials.compiled import gen_hbond_bases

    pbt = pose_stack.packed_block_types
    with torch.no_grad():
        _, derived = gen_hbond_bases(
            pose_stack.coords.reshape(-1, 3).contiguous(),
            pose_stack.rot_coord_offset,
            pose_stack.first_rot_for_block,
            pose_stack.first_rot_block_type,
            pose_stack.block_ind_for_rot,
            pose_stack.pose_ind_for_rot,
            pose_stack.block_type_ind_for_rot,
            pose_stack.inter_residue_connections,
            pbt.n_atoms,
            pbt.n_conn,
            pbt.conn_atom,
            pbt.n_all_bonds,
            pbt.all_bonds,
            pbt.atom_all_bond_ranges,
            pbt.hbpbt_params.tile_n_donH,
            pbt.hbpbt_params.tile_n_acc,
            pbt.hbpbt_params.tile_donH_inds,
            pbt.hbpbt_params.tile_acc_inds,
            pbt.hbpbt_params.tile_acceptor_hybridization,
            pbt.hbpbt_params.is_hydrogen,
        )
    return derived


def _ligand_water_topology(
    pose_stack: PoseStack,
    ligand_block: int,
    term: LKBallEnergyTerm,
    derived_atom_inds: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    pbt = pose_stack.packed_block_types
    type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pbt.active_block_types[type_index]
    hb = block_type.hbbt_params
    n_atoms = block_type.n_atoms
    offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    end = offset + n_atoms
    polar_atoms = sorted(set(hb.don_hvy_inds.tolist()) | set(hb.acc_inds.tolist()))
    polar = torch.tensor(polar_atoms, dtype=torch.int64)
    atom_to_polar = torch.full((n_atoms,), -1, dtype=torch.int64)
    if polar.numel():
        atom_to_polar[polar] = torch.arange(polar.shape[0], dtype=torch.int64)
    kinds = torch.zeros((polar.shape[0], MAX_N_WATER), dtype=torch.int64)
    atoms = torch.full((polar.shape[0], MAX_N_WATER, 3), -1, dtype=torch.int64)
    params = torch.zeros((polar.shape[0], MAX_N_WATER, 3), dtype=torch.float64)
    rings = torch.zeros((polar.shape[0], MAX_N_WATER), dtype=torch.int64)

    water_globals = term._lk_ball_water_gen_global_params[0].to(torch.float64)
    water_dist = float(water_globals[0])
    for tile in range(hb.tile_n_donH.shape[0]):
        for site in range(int(hb.tile_n_donH[tile])):
            hydrogen = tile * term.tile_size + int(hb.tile_donH_inds[tile, site])
            donor = int(hb.tile_donH_hvy_inds[tile, site])
            slot = int(hb.tile_which_donH_of_donH_hvy[tile, site])
            if not (0 <= donor < n_atoms and 0 <= hydrogen < n_atoms):
                raise RuntimeError("ligand donor water geometry is not block-local")
            if not 0 <= slot < MAX_N_WATER:
                raise RuntimeError("ligand donor water slot exceeds MAX_N_WATER")
            polar_site = int(atom_to_polar[donor])
            if polar_site < 0 or int(kinds[polar_site, slot]) != 0:
                raise RuntimeError("ligand donor water topology is inconsistent")
            kinds[polar_site, slot] = 1
            atoms[polar_site, slot, :2] = torch.tensor((donor, hydrogen))
            params[polar_site, slot, 0] = water_dist

    torsions = {
        1: term.ljlk_param_resolver.global_params.lkb_water_tors_sp2,
        2: term.ljlk_param_resolver.global_params.lkb_water_tors_sp3,
        3: term.ljlk_param_resolver.global_params.lkb_water_tors_ring,
    }
    acceptors = {}
    for tile in range(hb.tile_n_acc.shape[0]):
        for site in range(int(hb.tile_n_acc[tile])):
            acceptor = tile * term.tile_size + int(hb.tile_acc_inds[tile, site])
            acceptors[acceptor] = (
                int(hb.tile_acceptor_hybridization[tile, site]),
                int(hb.tile_acceptor_n_attached_H[tile, site]),
            )
    for acceptor in hb.acc_inds.tolist():
        hybridization, water_offset = acceptors[int(acceptor)]
        global_acceptor = offset + int(acceptor)
        base = int(derived_atom_inds[global_acceptor, 1])
        base0 = int(derived_atom_inds[global_acceptor, 2])
        if base < 0 or base0 < 0 or base == base0:
            continue
        if not (offset <= base < end and offset <= base0 < end):
            raise ValueError(
                "ligand acceptor water geometry crosses a block boundary; "
                "live grid LK-ball cannot represent it"
            )
        if hybridization not in torsions:
            raise RuntimeError("ligand acceptor has unsupported hybridization")
        polar_site = int(atom_to_polar[int(acceptor)])
        for water_index, torsion in enumerate(torsions[hybridization]):
            slot = water_offset + water_index
            if not 0 <= slot < MAX_N_WATER:
                raise RuntimeError("ligand acceptor water slot exceeds MAX_N_WATER")
            if int(kinds[polar_site, slot]) != 0:
                raise RuntimeError("ligand acceptor water topology is inconsistent")
            kinds[polar_site, slot] = 2
            atoms[polar_site, slot] = torch.tensor(
                (int(acceptor), base - offset, base0 - offset), dtype=torch.int64
            )
            params[polar_site, slot] = torch.tensor(
                (water_dist, float(water_globals[hybridization]), float(torsion)),
                dtype=torch.float64,
            )
            rings[polar_site, slot] = int(hybridization == 3)
    return polar, atom_to_polar, kinds, atoms, params, rings


def build_live_lk_ball_data(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    context_blocks: tuple[int, ...] = (),
) -> LiveLKBallData:
    """Build movable water topology and one fixed-receptor cell structure."""

    if pose_stack.n_poses != 1 or pose_stack.device.type != "cpu":
        raise ValueError("live LK-ball construction requires one CPU pose")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError("ligand_block is out of range")
    if int(pose_stack.block_type_ind64[0, ligand_block]) < 0:
        raise ValueError("ligand_block is absent from the reference pose")
    ligand_connection_partners = pose_stack.inter_residue_connections64[
        0, ligand_block, :, 0
    ]
    if bool(torch.any(ligand_connection_partners >= 0)):
        raise ValueError(
            "live grid LK-ball requires a ligand block with no covalent "
            "receptor connections"
        )

    term = LKBallEnergyTerm(param_db, torch.device("cpu"))
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    type_params = _pack_type_params(term)
    reference_waters = _generate_reference_waters(pose_stack, term)
    derived_atom_inds = _derived_atom_indices(pose_stack)
    pbt = pose_stack.packed_block_types

    ligand_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    ligand_type = pbt.active_block_types[ligand_type_index]
    ligand_n_atoms = ligand_type.n_atoms
    if sidechain_blocks:
        from tmol.dock.ga import (
            _movable_block_mapping,
            _movable_pose_stack,
            _sidechain_context_blocks,
        )
        from tmol.dock.ligand_dofs import (
            _sidechain_coordinate_indices,
            _validated_sidechain_blocks,
            build_ligand_docking_dofs,
        )
        from tmol.kinematics.script_modules import PoseStackKinematicsModule

        sidechain_blocks = _validated_sidechain_blocks(
            pose_stack, ligand_block, sidechain_blocks
        )
        expected_context = _sidechain_context_blocks(
            pose_stack, ligand_block, sidechain_blocks
        )
        if context_blocks and tuple(context_blocks) != expected_context:
            raise ValueError("context_blocks do not match sidechain peptide neighbors")
        context_blocks = expected_context
        movable_pose = _movable_pose_stack(
            pose_stack, ligand_block, sidechain_blocks
        )
        source_blocks, compact_sidechain_blocks, _ = _movable_block_mapping(
            pose_stack, ligand_block, sidechain_blocks
        )
        docking_dofs = build_ligand_docking_dofs(
            movable_pose, 0, sidechain_blocks=compact_sidechain_blocks
        )
        kin_module = PoseStackKinematicsModule(
            movable_pose, docking_dofs.fold_forest
        )
        _, sidechain_atom_indices = _sidechain_coordinate_indices(
            movable_pose, docking_dofs, kin_module.kmd.forest
        )
        query_atoms = set(range(ligand_n_atoms))
        query_atoms.update(sidechain_atom_indices[0].tolist())
        compact_atom_types = []
        compact_is_hydrogen = []
        polar_parts = []
        water_kind_parts = []
        water_atom_parts = []
        water_param_parts = []
        water_ring_parts = []
        for compact_block, source_block in enumerate(source_blocks):
            type_index = int(pose_stack.block_type_ind64[0, source_block])
            block_type = pbt.active_block_types[type_index]
            n_atoms = block_type.n_atoms
            compact_offset = int(
                movable_pose.block_coord_offset64[0, compact_block]
            )
            compact_atom_types.append(
                pbt.atom_types[type_index, :n_atoms].to(torch.int64)
            )
            compact_is_hydrogen.append(
                pbt.hbpbt_params.is_hydrogen[type_index, :n_atoms].to(torch.bool)
            )
            (
                block_polar,
                _,
                block_kinds,
                block_atoms,
                block_params,
                block_rings,
            ) = _ligand_water_topology(
                pose_stack, source_block, term, derived_atom_inds
            )
            compact_polar = block_polar + compact_offset
            keep_polar = torch.tensor(
                [int(atom) in query_atoms for atom in compact_polar],
                dtype=torch.bool,
            )
            if not bool(torch.any(keep_polar)):
                continue
            selected_atoms = block_atoms[keep_polar].clone()
            active_atoms = selected_atoms >= 0
            selected_atoms[active_atoms] += compact_offset
            polar_parts.append(compact_polar[keep_polar])
            water_kind_parts.append(block_kinds[keep_polar])
            water_atom_parts.append(selected_atoms)
            water_param_parts.append(block_params[keep_polar])
            water_ring_parts.append(block_rings[keep_polar])

        ligand_n_atoms = movable_pose.max_n_pose_atoms
        ligand_atom_types = torch.cat(compact_atom_types)
        ligand_atom_params = type_params[ligand_atom_types].contiguous()
        is_hydrogen = torch.cat(compact_is_hydrogen)
        ligand_heavy_atoms = torch.tensor(
            sorted(atom for atom in query_atoms if not bool(is_hydrogen[atom])),
            dtype=torch.int64,
        ).contiguous()
        ligand_polar_atoms = (
            torch.cat(polar_parts).to(torch.int64).contiguous()
            if polar_parts
            else torch.empty(0, dtype=torch.int64)
        )
        ligand_atom_to_polar = torch.full(
            (ligand_n_atoms,), -1, dtype=torch.int64
        )
        if ligand_polar_atoms.numel():
            ligand_atom_to_polar[ligand_polar_atoms] = torch.arange(
                ligand_polar_atoms.shape[0], dtype=torch.int64
            )
        ligand_water_kinds = (
            torch.cat(water_kind_parts).contiguous()
            if water_kind_parts
            else torch.empty((0, MAX_N_WATER), dtype=torch.int64)
        )
        ligand_water_atoms = (
            torch.cat(water_atom_parts).contiguous()
            if water_atom_parts
            else torch.empty((0, MAX_N_WATER, 3), dtype=torch.int64)
        )
        ligand_water_params = (
            torch.cat(water_param_parts).contiguous()
            if water_param_parts
            else torch.empty((0, MAX_N_WATER, 3), dtype=torch.float64)
        )
        ligand_water_is_ring = (
            torch.cat(water_ring_parts).contiguous()
            if water_ring_parts
            else torch.empty((0, MAX_N_WATER), dtype=torch.int64)
        )
    else:
        sidechain_blocks = ()
        if context_blocks:
            raise ValueError("context_blocks require sidechain_blocks")
        context_blocks = ()
        ligand_atom_types = pbt.atom_types[
            ligand_type_index, :ligand_n_atoms
        ].to(torch.int64)
        ligand_atom_params = type_params[ligand_atom_types].contiguous()
        ligand_heavy_atoms = (
            torch.nonzero(
                pbt.hbpbt_params.is_hydrogen[
                    ligand_type_index, :ligand_n_atoms
                ]
                == 0
            )
            .reshape(-1)
            .to(torch.int64)
            .contiguous()
        )
        (
            ligand_polar_atoms,
            ligand_atom_to_polar,
            ligand_water_kinds,
            ligand_water_atoms,
            ligand_water_params,
            ligand_water_is_ring,
        ) = _ligand_water_topology(
            pose_stack, ligand_block, term, derived_atom_inds
        )

    receptor_coords = []
    receptor_params = []
    receptor_waters = []
    receptor_polar_atoms = []
    for block_index in range(pose_stack.max_n_blocks):
        type_index = int(pose_stack.block_type_ind64[0, block_index])
        if (
            type_index < 0
            or block_index == ligand_block
            or block_index in sidechain_blocks
            or block_index in context_blocks
        ):
            continue
        block_type = pbt.active_block_types[type_index]
        offset = int(pose_stack.block_coord_offset64[0, block_index])
        local_polar = set(block_type.hbbt_params.don_hvy_inds.tolist()) | set(
            block_type.hbbt_params.acc_inds.tolist()
        )
        for atom in range(block_type.n_atoms):
            if int(pbt.hbpbt_params.is_hydrogen[type_index, atom]) != 0:
                continue
            receptor_index = len(receptor_coords)
            global_atom = offset + atom
            atom_type = int(pbt.atom_types[type_index, atom])
            receptor_coords.append(
                pose_stack.coords.reshape(-1, 3)[global_atom].to(torch.float64)
            )
            receptor_params.append(type_params[atom_type])
            receptor_waters.append(reference_waters[global_atom])
            if atom in local_polar:
                receptor_polar_atoms.append(receptor_index)
    if not receptor_coords:
        raise ValueError("pose contains no receptor heavy atoms")
    receptor_coords_t = torch.stack(receptor_coords).contiguous()
    receptor_params_t = torch.stack(receptor_params).contiguous()
    receptor_waters_t = torch.stack(receptor_waters).contiguous()
    receptor_polar_atoms_t = torch.tensor(
        receptor_polar_atoms, dtype=torch.int64
    ).contiguous()

    global_params = term._lk_ball_global_params.to(torch.float64).contiguous()
    cutoff = float(global_params[0, 4])
    cell_origin = torch.floor(receptor_coords_t.min(dim=0).values / cutoff) * cutoff
    cell_coords = torch.floor((receptor_coords_t - cell_origin) / cutoff).to(
        torch.int64
    )
    cell_dims = (cell_coords.max(dim=0).values + 1).contiguous()
    receptor_indices = torch.arange(receptor_coords_t.shape[0], dtype=torch.int64)
    receptor_cell_offsets, receptor_cell_indices = _build_cell_list(
        receptor_coords_t, receptor_indices, cell_origin, cell_dims, cutoff
    )
    receptor_polar_cell_offsets, receptor_polar_cell_indices = _build_cell_list(
        receptor_coords_t[receptor_polar_atoms_t],
        receptor_polar_atoms_t,
        cell_origin,
        cell_dims,
        cutoff,
    )
    return LiveLKBallData(
        n_ligand_atoms=ligand_n_atoms,
        ligand_atom_params=ligand_atom_params,
        ligand_heavy_atoms=ligand_heavy_atoms,
        ligand_polar_atoms=ligand_polar_atoms.contiguous(),
        ligand_atom_to_polar=ligand_atom_to_polar.contiguous(),
        ligand_water_kinds=ligand_water_kinds.contiguous(),
        ligand_water_atoms=ligand_water_atoms.contiguous(),
        ligand_water_params=ligand_water_params.contiguous(),
        ligand_water_is_ring=ligand_water_is_ring.contiguous(),
        receptor_coords=receptor_coords_t,
        receptor_params=receptor_params_t,
        receptor_waters=receptor_waters_t,
        receptor_polar_atoms=receptor_polar_atoms_t,
        cell_origin=cell_origin.contiguous(),
        cell_dims=cell_dims,
        receptor_cell_offsets=receptor_cell_offsets,
        receptor_cell_indices=receptor_cell_indices,
        receptor_polar_cell_offsets=receptor_polar_cell_offsets,
        receptor_polar_cell_indices=receptor_polar_cell_indices,
        global_params=global_params,
    )


def _validate_ligand_coords(data: LiveLKBallData, ligand_coords: torch.Tensor) -> None:
    if ligand_coords.ndim != 3 or ligand_coords.shape[2] != 3:
        raise ValueError("ligand_coords must have shape [n_poses, n_atoms, 3]")
    if ligand_coords.shape[0] == 0:
        raise ValueError("at least one ligand pose is required")
    if ligand_coords.shape[1] != data.n_ligand_atoms:
        raise ValueError("ligand atom count does not match live LK-ball topology")
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if ligand_coords.device != data.receptor_coords.device:
        raise ValueError("live LK-ball data must match coordinate device")
    if ligand_coords.dtype != data.receptor_coords.dtype:
        raise TypeError("live LK-ball data must match coordinate dtype")


def _live_lk_ball_op(
    data: LiveLKBallData,
    ligand_coords: torch.Tensor,
    *,
    use_cell_list: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .potentials.compiled import live_lk_ball

    return live_lk_ball(
        ligand_coords.contiguous(),
        data.ligand_atom_params,
        data.ligand_heavy_atoms,
        data.ligand_polar_atoms,
        data.ligand_atom_to_polar,
        data.ligand_water_kinds,
        data.ligand_water_atoms,
        data.ligand_water_params,
        data.ligand_water_is_ring,
        data.receptor_coords,
        data.receptor_params,
        data.receptor_waters,
        data.receptor_polar_atoms,
        data.cell_origin,
        data.cell_dims,
        data.receptor_cell_offsets,
        data.receptor_cell_indices,
        data.receptor_polar_cell_offsets,
        data.receptor_polar_cell_indices,
        data.global_params,
        use_cell_list,
    )


class _LiveLKBallAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ligand_coords: torch.Tensor, data: LiveLKBallData):
        energies, gradients = _live_lk_ball_op(data, ligand_coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):
        (gradients,) = ctx.saved_tensors
        expanded_grad_energies = grad_energies[:, :, None, None]
        coordinate_gradients = torch.sum(
            torch.where(
                expanded_grad_energies != 0,
                gradients * expanded_grad_energies,
                torch.zeros_like(gradients),
            ),
            dim=1,
        )
        return coordinate_gradients, None


def score_ligand_lk_ball(
    data: LiveLKBallData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return four unweighted live LK-ball channels for each ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if ligand_coords.requires_grad:
        return _LiveLKBallAutograd.apply(ligand_coords, data)
    return _live_lk_ball_op(data, ligand_coords)[0]


def score_ligand_lk_ball_with_gradients(
    data: LiveLKBallData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted channels and analytic per-channel Cartesian gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.no_grad():
        return _live_lk_ball_op(data, ligand_coords.detach())
