import pytest
import torch
import attr

from tmol.types.functional import convert_args

from tmol.score.lkball.potentials import (render_waters)

from tmol.score.lkball.params import (WaterBuildingParams)


def test_render_waters():
    # eventually from a database
    waterparams = WaterBuildingParams(
        max_acc_wat=2,
        dists_sp2=torch.tensor([2.65, 2.65]),
        angles_sp2=torch.tensor([109.5, 109.5]),
        tors_sp2=torch.tensor([0.0, 180.0]),
        dists_sp3=torch.tensor([2.65, 2.65]),
        angles_sp3=torch.tensor([120.0, 120.0]),
        tors_sp3=torch.tensor([120.0, 240.0]),
        dists_ring=torch.tensor([2.65]),
        angles_ring=torch.tensor([180.0]),
        tors_ring=torch.tensor([0.0]),
        dist_donor=torch.tensor([2.65])
    )

    heavyatoms = torch.tensor([[0.0, 0.0, 0.0]])
    #acc_base = torch.tensor( [[ float('nan'), float('nan'), float('nan') ]] )
    #acc_base0 = torch.tensor( [[ float('nan'), float('nan'), float('nan') ]] )
    don_H = torch.full([1, 4, 3], float('nan'))
    #don_H[0,0,:] = torch.tensor([0,0,1]);
    #don_H[0,1,:] = torch.tensor([0,1,1]);
    #don_H[0,2,:] = torch.tensor([0,-1,1]);

    acc_base = torch.tensor([[0.0, 0.0, -1.0]])
    acc_base0 = torch.tensor([[0.0, -1.0, -1.0]])

    is_sp2_acceptor = torch.tensor([True])
    is_sp3_acceptor = torch.tensor([False])
    is_ring_acceptor = torch.tensor([False])

    waters = convert_args(render_waters)(
        heavyatoms, acc_base, acc_base0, don_H, is_sp2_acceptor,
        is_sp3_acceptor, is_ring_acceptor, waterparams
    )
