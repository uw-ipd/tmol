import toolz
import attr

import numpy
import torch
import sparse

from tmol.score.elec.torch_op import ElecScoreFun
from tmol.score.elec.params import ElecParamResolver
from tmol.score.bonded_atom import bonded_path_length

import tmol.database

from tmol.utility.args import ignore_unused_kwargs


def test_elec_torch(default_database, ubq_system, torch_device):
    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    res_names = ubq_system.atom_metadata["residue_name"].copy()
    # res_indices = system.atom_metadata["residue_index"].copy()

    coords = ubq_system.coords.copy()
    bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)
    tcoords = torch.from_numpy(coords).to(device=torch_device).requires_grad_(True)
    tbpl = torch.from_numpy(bpl).to(device=torch_device)

    epr = ElecParamResolver.from_database(default_database.scoring.elec, torch_device)
    pcs = epr.resolve_partial_charge(res_names, atom_names)

    import tmol.score.elec.potentials.compiled as compiled

    pairs, batch_scores, *batch_derivs = compiled.elec(
        tcoords, pcs, tcoords, pcs, tbpl, **attr.asdict(epr.global_params)
    )

    mask = torch.abs(batch_scores) > 1e-6
    print(pairs[mask])
    print(batch_scores[mask])

    assert False


def test_elec_deriv(default_database, ubq_system, torch_device):
    pass
