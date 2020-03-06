import torch

from tmol.types.torch import Tensor
from tmol.pack.datatypes import PackerEnergyTables

# Import compiled components to load torch_ops
import tmol.pack.compiled.compiled


def run_one_stage_simulated_annealing(
    simA_params: Tensor(torch.float)[:],
    energy_tables: PackerEnergyTables
):
    return torch.ops.tmol.one_stage_anneal(
        simA_params,
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.nenergies,
        energy_tables.twob_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b
    )
    
def run_multi_stage_simulated_annealing(
    simA_params: Tensor(torch.float)[:],
    energy_tables: PackerEnergyTables
):
    return torch.ops.tmol.multi_stage_anneal(
        simA_params,
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.nenergies,
        energy_tables.twob_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b
    )
    
