import torch

from tmol.pack.datatypes import PackerEnergyTables

# Import compiled components to load torch_ops
import tmol.pack.compiled.compiled


def run_simulated_annealing(
    energy_tables: PackerEnergyTables,
):
    return torch.ops.tmol.pack_anneal(
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.nenergies,
        energy_tables.twob_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b
    )
    