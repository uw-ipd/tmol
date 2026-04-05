from tmol.pack.datatypes import PackerEnergyTables

# Import compiled components to load torch_ops
from tmol.pack.compiled.compiled import pack_anneal


def run_simulated_annealing(
    energy_tables: PackerEnergyTables,
):
    # The target device is inferred from the pose-level tables (which may be
    # on MPS) even when the energy tables themselves ended up on CPU.
    target_device = energy_tables.pose_n_res.device
    scores, rotamer_assignments = pack_anneal(
        energy_tables.max_n_rotamers_per_pose,
        energy_tables.pose_n_res,
        energy_tables.pose_n_rotamers,
        energy_tables.pose_rotamer_offset,
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.chunk_size,
        energy_tables.chunk_offset_offsets,
        energy_tables.chunk_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b,
    )
    return scores.to(target_device), rotamer_assignments.to(target_device)
