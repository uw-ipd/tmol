"""Live ligand--receptor hydrogen bonds for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from tmol.database import ParameterDatabase
from tmol.pose.pose_stack import PoseStack
from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm


@dataclass(frozen=True)
class LiveHBondData:
    """One shared rigid-receptor hbond structure and one ligand topology.

    Receptor coordinates and cell lists occur once, independently of the
    number of ligand poses. Ligand tensors contain only block-local atom
    indices, so every pose supplies its own current base geometry.
    """

    n_ligand_atoms: int
    receptor_donors: torch.Tensor
    receptor_donor_types: torch.Tensor
    receptor_acceptors: torch.Tensor
    receptor_acceptor_types: torch.Tensor
    ligand_donor_atoms: torch.Tensor
    ligand_donor_types: torch.Tensor
    ligand_acceptor_atoms: torch.Tensor
    ligand_acceptor_types: torch.Tensor
    cell_origin: torch.Tensor
    cell_dims: torch.Tensor
    receptor_donor_cell_offsets: torch.Tensor
    receptor_donor_cell_indices: torch.Tensor
    receptor_acceptor_cell_offsets: torch.Tensor
    receptor_acceptor_cell_indices: torch.Tensor
    pair_params: torch.Tensor
    pair_polynomials: torch.Tensor
    global_params: torch.Tensor

    def __post_init__(self):
        if (
            not isinstance(self.n_ligand_atoms, int)
            or isinstance(self.n_ligand_atoms, bool)
            or self.n_ligand_atoms < 1
        ):
            raise ValueError("n_ligand_atoms must be a positive integer")
        tensor_values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "n_ligand_atoms"
        }
        if not all(isinstance(value, torch.Tensor) for value in tensor_values.values()):
            raise TypeError("all live hbond arrays must be tensors")
        devices = {value.device for value in tensor_values.values()}
        if len(devices) != 1:
            raise ValueError("all live hbond arrays must share one device")
        if not all(value.is_contiguous() for value in tensor_values.values()):
            raise ValueError("all live hbond arrays must be contiguous")

        real_dtype = self.receptor_donors.dtype
        if real_dtype not in (torch.float32, torch.float64):
            raise TypeError("live hbond coordinates must be float32 or float64")
        real_names = (
            "receptor_donors",
            "receptor_acceptors",
            "cell_origin",
            "pair_params",
            "global_params",
        )
        if any(tensor_values[name].dtype != real_dtype for name in real_names):
            raise TypeError("live hbond real arrays must share one dtype")
        if self.pair_polynomials.dtype != torch.float64:
            raise TypeError("hbond polynomial parameters must be float64")
        integer_names = set(tensor_values) - set(real_names) - {"pair_polynomials"}
        if any(tensor_values[name].dtype != torch.int64 for name in integer_names):
            raise TypeError("live hbond index arrays must be int64")

        n_receptor_donors = self.receptor_donors.shape[0]
        n_receptor_acceptors = self.receptor_acceptors.shape[0]
        n_ligand_donors = self.ligand_donor_atoms.shape[0]
        n_ligand_acceptors = self.ligand_acceptor_atoms.shape[0]
        expected_shapes = {
            "receptor_donors": (n_receptor_donors, 2, 3),
            "receptor_donor_types": (n_receptor_donors,),
            "receptor_acceptors": (n_receptor_acceptors, 3, 3),
            "receptor_acceptor_types": (n_receptor_acceptors,),
            "ligand_donor_atoms": (n_ligand_donors, 2),
            "ligand_donor_types": (n_ligand_donors,),
            "ligand_acceptor_atoms": (n_ligand_acceptors, 3),
            "ligand_acceptor_types": (n_ligand_acceptors,),
            "cell_origin": (3,),
            "cell_dims": (3,),
            "pair_params": (
                self.pair_params.shape[0],
                self.pair_params.shape[1],
                3,
            ),
            "pair_polynomials": (
                self.pair_params.shape[0],
                self.pair_params.shape[1],
                48,
            ),
            "global_params": (1, 6),
        }
        for name, expected in expected_shapes.items():
            if tuple(tensor_values[name].shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if self.receptor_donor_cell_offsets.ndim != 1:
            raise ValueError("receptor donor cell offsets must be one-dimensional")
        n_cells = self.receptor_donor_cell_offsets.shape[0] - 1
        cell_shapes = {
            "receptor_donor_cell_offsets": (n_cells + 1,),
            "receptor_donor_cell_indices": (n_receptor_donors,),
            "receptor_acceptor_cell_offsets": (n_cells + 1,),
            "receptor_acceptor_cell_indices": (n_receptor_acceptors,),
        }
        for name, expected in cell_shapes.items():
            if tuple(tensor_values[name].shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")

        # The CPU constructor is the trust boundary. Device copies originate
        # from this validated object; the kernel additionally guards every
        # dynamic index so malformed device data yields NaN instead of OOB.
        if self.receptor_donors.device.type != "cpu":
            return
        if bool(
            torch.any(self.cell_dims <= 0) or int(torch.prod(self.cell_dims)) != n_cells
        ):
            raise ValueError("cell_dims must be positive and match cell offsets")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                self.receptor_donors,
                self.receptor_acceptors,
                self.cell_origin,
                self.global_params,
            )
        ):
            raise ValueError("live hbond coordinates and global params must be finite")
        if not float(self.global_params[0, 5]) > 0:
            raise ValueError("live hbond cutoff must be positive")

        def check_range(value: torch.Tensor, upper: int, name: str) -> None:
            if value.numel() and bool(
                torch.any(value < 0) or torch.any(value >= upper)
            ):
                raise ValueError(f"{name} contains an out-of-range index")

        check_range(
            self.ligand_donor_atoms,
            self.n_ligand_atoms,
            "ligand_donor_atoms",
        )
        check_range(
            self.ligand_acceptor_atoms,
            self.n_ligand_atoms,
            "ligand_acceptor_atoms",
        )
        check_range(
            self.ligand_donor_types,
            self.pair_params.shape[0],
            "ligand_donor_types",
        )
        check_range(
            self.receptor_donor_types,
            self.pair_params.shape[0],
            "receptor_donor_types",
        )
        check_range(
            self.ligand_acceptor_types,
            self.pair_params.shape[1],
            "ligand_acceptor_types",
        )
        check_range(
            self.receptor_acceptor_types,
            self.pair_params.shape[1],
            "receptor_acceptor_types",
        )

        for kind, offsets, indices, n_sites in (
            (
                "donor",
                self.receptor_donor_cell_offsets,
                self.receptor_donor_cell_indices,
                n_receptor_donors,
            ),
            (
                "acceptor",
                self.receptor_acceptor_cell_offsets,
                self.receptor_acceptor_cell_indices,
                n_receptor_acceptors,
            ),
        ):
            if (
                int(offsets[0]) != 0
                or int(offsets[-1]) != n_sites
                or bool(torch.any(offsets[1:] < offsets[:-1]))
            ):
                raise ValueError(f"receptor {kind} cell offsets are not valid CSR")
            check_range(indices, n_sites, f"receptor_{kind}_cell_indices")
            if n_sites and not torch.equal(
                torch.sort(indices).values, torch.arange(n_sites, dtype=torch.int64)
            ):
                raise ValueError(f"receptor {kind} cell indices must be a permutation")

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "LiveHBondData":
        """Return the structure on a scoring device and coordinate dtype."""

        if dtype not in (torch.float32, torch.float64):
            raise TypeError("live hbond dtype must be float32 or float64")
        device = torch.device(device)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, torch.Tensor):
                values[field.name] = value
                continue
            if value.dtype.is_floating_point:
                target_dtype = (
                    torch.float64 if field.name == "pair_polynomials" else dtype
                )
                value = value.to(device=device, dtype=target_dtype)
            else:
                value = value.to(device=device)
            values[field.name] = value.contiguous()
        return LiveHBondData(**values)


def _empty_coords(n_sites: int, n_points: int) -> torch.Tensor:
    return torch.empty((n_sites, n_points, 3), dtype=torch.float64)


def _stack_coords(values: list[torch.Tensor], n_points: int) -> torch.Tensor:
    if not values:
        return _empty_coords(0, n_points)
    return torch.stack(values).to(dtype=torch.float64).contiguous()


def _stack_indices(values: list[tuple[int, ...]], width: int) -> torch.Tensor:
    if not values:
        return torch.empty((0, width), dtype=torch.int64)
    return torch.tensor(values, dtype=torch.int64).contiguous()


def _build_cell_list(
    anchors: torch.Tensor,
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
        raise RuntimeError("receptor hbond site fell outside its cell structure")
    linear = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    order = torch.argsort(linear, stable=True)
    counts = torch.bincount(linear, minlength=n_cells)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return offsets.contiguous(), order.to(torch.int64).contiguous()


def _typed_sites_for_block(
    pose_stack: PoseStack,
    block_index: int,
    derived_atom_inds: torch.Tensor,
) -> tuple[
    list[tuple[int, int, int]],
    list[int],
    list[tuple[int, int, int, int]],
    list[int],
]:
    """Return global donor/acceptor source indices and hbond types."""

    pbt = pose_stack.packed_block_types
    block_type_index = int(pose_stack.block_type_ind64[0, block_index])
    block_type = pbt.active_block_types[block_type_index]
    n_atoms = block_type.n_atoms
    offset = int(pose_stack.block_coord_offset64[0, block_index])
    params = pbt.hbpbt_params
    donors: list[tuple[int, int, int]] = []
    donor_types: list[int] = []
    acceptors: list[tuple[int, int, int, int]] = []
    acceptor_types: list[int] = []
    n_tiles = (n_atoms - 1) // HBondEnergyTerm.tile_size + 1
    for tile in range(n_tiles):
        tile_start = tile * HBondEnergyTerm.tile_size
        n_donors = int(params.tile_n_donH[block_type_index, tile])
        for site in range(n_donors):
            hydrogen = (
                offset
                + tile_start
                + int(params.tile_donH_inds[block_type_index, tile, site])
            )
            donor = int(derived_atom_inds[hydrogen, 0])
            if donor < 0:
                raise RuntimeError("hbond donor base generation returned no heavy atom")
            donors.append((donor, hydrogen, offset))
            donor_types.append(
                int(params.tile_donorH_type[block_type_index, tile, site])
            )

        n_acceptors = int(params.tile_n_acc[block_type_index, tile])
        for site in range(n_acceptors):
            acceptor = (
                offset
                + tile_start
                + int(params.tile_acc_inds[block_type_index, tile, site])
            )
            base = int(derived_atom_inds[acceptor, 1])
            base0 = int(derived_atom_inds[acceptor, 2])
            if base < 0 or base0 < 0:
                raise RuntimeError(
                    "hbond acceptor base generation returned no base atom"
                )
            acceptors.append((acceptor, base, base0, offset))
            acceptor_types.append(
                int(params.tile_acceptor_type[block_type_index, tile, site])
            )
    return donors, donor_types, acceptors, acceptor_types


def build_live_hbond_data(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    context_blocks: tuple[int, ...] = (),
) -> LiveHBondData:
    """Build movable-atom topology and shared fixed-receptor hbond cells."""

    if pose_stack.n_poses != 1 or pose_stack.device.type != "cpu":
        raise ValueError("live hbond construction requires one CPU pose")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError("ligand_block is out of range")
    if int(pose_stack.block_type_ind64[0, ligand_block]) < 0:
        raise ValueError("ligand_block is absent from the reference pose")

    term = HBondEnergyTerm(param_db, torch.device("cpu"))
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)

    from tmol.score.hbond.potentials.compiled import gen_hbond_bases

    pbt = pose_stack.packed_block_types
    with torch.no_grad():
        _, derived_atom_inds = gen_hbond_bases(
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

    flat_coords = pose_stack.coords.reshape(-1, 3).detach().to(torch.float64)
    receptor_donor_coords: list[torch.Tensor] = []
    receptor_donor_types: list[int] = []
    receptor_acceptor_coords: list[torch.Tensor] = []
    receptor_acceptor_types: list[int] = []
    ligand_donor_atoms: list[tuple[int, int]] = []
    ligand_donor_types: list[int] = []
    ligand_acceptor_atoms: list[tuple[int, int, int]] = []
    ligand_acceptor_types: list[int] = []

    ligand_offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    ligand_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    ligand_n_atoms = pbt.active_block_types[ligand_type_index].n_atoms
    ligand_end = ligand_offset + ligand_n_atoms

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
        source_blocks, compact_sidechain_blocks, compact_context_blocks = (
            _movable_block_mapping(
                pose_stack, ligand_block, sidechain_blocks
            )
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
        source_to_compact_block = {
            source_block: compact_block
            for compact_block, source_block in enumerate(source_blocks)
        }
        if set(compact_context_blocks) & set(compact_sidechain_blocks):
            raise RuntimeError("movable and context block mappings overlap")
        movable_n_atoms = movable_pose.max_n_pose_atoms
    else:
        sidechain_blocks = ()
        if context_blocks:
            raise ValueError("context_blocks require sidechain_blocks")
        query_atoms = set(range(ligand_n_atoms))
        source_to_compact_block = {ligand_block: 0}
        movable_n_atoms = ligand_n_atoms

    for block_index in range(pose_stack.max_n_blocks):
        if int(pose_stack.block_type_ind64[0, block_index]) < 0:
            continue
        donors, donor_types, acceptors, acceptor_types = _typed_sites_for_block(
            pose_stack, block_index, derived_atom_inds
        )
        if block_index in source_to_compact_block:
            source_offset = int(pose_stack.block_coord_offset64[0, block_index])
            compact_block = source_to_compact_block[block_index]
            compact_offset = (
                0
                if compact_block == 0 and not sidechain_blocks
                else int(movable_pose.block_coord_offset64[0, compact_block])
            )
            block_n_atoms = pose_stack.block_type(0, block_index).n_atoms

            def compact_atom(global_atom: int) -> int:
                local_atom = global_atom - source_offset
                if not 0 <= local_atom < block_n_atoms:
                    raise ValueError(
                        "movable hbond geometry crosses a block boundary"
                    )
                return compact_offset + local_atom

            for (donor, hydrogen, _), donor_type in zip(donors, donor_types):
                compact_donor = compact_atom(donor)
                if compact_donor not in query_atoms:
                    continue
                compact_hydrogen = compact_atom(hydrogen)
                ligand_donor_atoms.append((compact_donor, compact_hydrogen))
                ligand_donor_types.append(donor_type)
            for (acceptor, base, base0, _), acceptor_type in zip(
                acceptors, acceptor_types
            ):
                compact_acceptor = compact_atom(acceptor)
                if compact_acceptor not in query_atoms:
                    continue
                compact_base = compact_atom(base)
                compact_base0 = compact_atom(base0)
                ligand_acceptor_atoms.append(
                    (compact_acceptor, compact_base, compact_base0)
                )
                ligand_acceptor_types.append(acceptor_type)
        else:
            for (donor, hydrogen, _), donor_type in zip(donors, donor_types):
                if ligand_offset <= donor < ligand_end:
                    raise ValueError("receptor donor base points into the ligand block")
                receptor_donor_coords.append(flat_coords[[donor, hydrogen]])
                receptor_donor_types.append(donor_type)
            for (acceptor, base, base0, _), acceptor_type in zip(
                acceptors, acceptor_types
            ):
                if any(
                    ligand_offset <= atom < ligand_end
                    for atom in (acceptor, base, base0)
                ):
                    raise ValueError(
                        "receptor acceptor base points into the ligand block"
                    )
                receptor_acceptor_coords.append(flat_coords[[acceptor, base, base0]])
                receptor_acceptor_types.append(acceptor_type)

    pair_params, pair_polynomials, global_params = term._param_tables(torch.float64)
    max_ha_dis = float(global_params[0, 5])
    if not (max_ha_dis > 0):
        raise RuntimeError("hbond max H--A distance must be positive")

    receptor_donors = _stack_coords(receptor_donor_coords, 2)
    receptor_acceptors = _stack_coords(receptor_acceptor_coords, 3)
    anchors = []
    if receptor_donors.shape[0]:
        anchors.append(receptor_donors[:, 1])
    if receptor_acceptors.shape[0]:
        anchors.append(receptor_acceptors[:, 0])
    if anchors:
        all_anchors = torch.cat(anchors)
        cell_origin = (
            torch.floor(all_anchors.min(dim=0).values / max_ha_dis) * max_ha_dis
        )
        cell_coords = torch.floor((all_anchors - cell_origin) / max_ha_dis).to(
            torch.int64
        )
        cell_dims = cell_coords.max(dim=0).values + 1
    else:
        cell_origin = torch.zeros(3, dtype=torch.float64)
        cell_dims = torch.ones(3, dtype=torch.int64)

    donor_offsets, donor_cell_indices = _build_cell_list(
        receptor_donors[:, 1], cell_origin, cell_dims, max_ha_dis
    )
    acceptor_offsets, acceptor_cell_indices = _build_cell_list(
        receptor_acceptors[:, 0], cell_origin, cell_dims, max_ha_dis
    )
    return LiveHBondData(
        n_ligand_atoms=movable_n_atoms,
        receptor_donors=receptor_donors,
        receptor_donor_types=torch.tensor(receptor_donor_types, dtype=torch.int64),
        receptor_acceptors=receptor_acceptors,
        receptor_acceptor_types=torch.tensor(
            receptor_acceptor_types, dtype=torch.int64
        ),
        ligand_donor_atoms=_stack_indices(ligand_donor_atoms, 2),
        ligand_donor_types=torch.tensor(ligand_donor_types, dtype=torch.int64),
        ligand_acceptor_atoms=_stack_indices(ligand_acceptor_atoms, 3),
        ligand_acceptor_types=torch.tensor(ligand_acceptor_types, dtype=torch.int64),
        cell_origin=cell_origin.to(torch.float64).contiguous(),
        cell_dims=cell_dims.contiguous(),
        receptor_donor_cell_offsets=donor_offsets,
        receptor_donor_cell_indices=donor_cell_indices,
        receptor_acceptor_cell_offsets=acceptor_offsets,
        receptor_acceptor_cell_indices=acceptor_cell_indices,
        pair_params=pair_params.detach().clone().contiguous(),
        pair_polynomials=pair_polynomials.detach().clone().contiguous(),
        global_params=global_params.detach().clone().contiguous(),
    )


def _validate_ligand_coords(data: LiveHBondData, ligand_coords: torch.Tensor) -> None:
    if ligand_coords.ndim != 3 or ligand_coords.shape[2] != 3:
        raise ValueError("ligand_coords must have shape [n_poses, n_atoms, 3]")
    if ligand_coords.shape[0] == 0 or ligand_coords.shape[1] == 0:
        raise ValueError("live hbond scoring requires poses and ligand atoms")
    if ligand_coords.shape[1] != data.n_ligand_atoms:
        raise ValueError(
            "ligand coordinate atom count does not match live hbond topology"
        )
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if data.receptor_donors.device != ligand_coords.device:
        raise ValueError("live hbond data and ligand coordinates must share a device")
    if data.receptor_donors.dtype != ligand_coords.dtype:
        raise TypeError("live hbond data and ligand coordinates must share a dtype")


def _live_hbond_op(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from .potentials.compiled import live_hbond

    return live_hbond(
        ligand_coords.contiguous(),
        data.ligand_donor_atoms,
        data.ligand_donor_types,
        data.ligand_acceptor_atoms,
        data.ligand_acceptor_types,
        data.receptor_donors,
        data.receptor_donor_types,
        data.receptor_acceptors,
        data.receptor_acceptor_types,
        data.cell_origin,
        data.cell_dims,
        data.receptor_donor_cell_offsets,
        data.receptor_donor_cell_indices,
        data.receptor_acceptor_cell_offsets,
        data.receptor_acceptor_cell_indices,
        data.pair_params,
        data.pair_polynomials,
        data.global_params,
    )


class _LiveHBondAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ligand_coords: torch.Tensor, data: LiveHBondData):
        energies, gradients = _live_hbond_op(data, ligand_coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies):
        (gradients,) = ctx.saved_tensors
        return gradients * grad_energies[:, None, None], None


def score_ligand_hbond(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return unweighted live interface hbond energy for every ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if ligand_coords.requires_grad:
        return _LiveHBondAutograd.apply(ligand_coords, data)
    return _live_hbond_op(data, ligand_coords)[0]


def score_ligand_hbond_with_gradients(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted live hbond energies and analytic ligand gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.no_grad():
        return _live_hbond_op(data, ligand_coords.detach())
