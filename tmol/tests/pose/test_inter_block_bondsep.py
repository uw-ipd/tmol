from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION as CAP
from tmol.io import (
    atom_array_from_cif,
    build_context_from_biotite,
    pose_stack_from_biotite,
    pose_stack_from_pdb,
)
from tmol.pose import PoseStackBuilder

DATA = Path(__file__).parents[1] / "data"


def heavy_pose_stack(array, device):
    array = array[np.char.upper(array.element.astype(str)) != "H"]
    context = build_context_from_biotite(array, device)
    return pose_stack_from_biotite(array, device, context=context, no_optH=True)


def oex_site_5xnl(device):
    """5XNL's Mn4CaO5 cluster (OEX A 401, 12 connections) and residues within 3 A."""
    path = DATA / "atomworks_regressions" / "decreasing_water_author_ids_5xnl.cif.zst"
    array = atom_array_from_cif(path)
    cluster = array.coord[(array.res_name == "OEX") & (array.chain_id == "A")]
    distance = np.linalg.norm(array.coord[:, None] - cluster[None], axis=-1)
    residue = struc.get_all_residue_positions(array)
    return heavy_pose_stack(
        array[np.isin(residue, residue[distance.min(axis=1) < 3.0])], device
    )


def floyd_warshall_bondsep(pose_stack):
    """Dense ``[pose, block1, block2, conn1, conn2]`` capped separations."""
    pbt = pose_stack.packed_block_types
    n_poses, n_blocks = pose_stack.block_type_ind64.shape
    dense = torch.full(
        (n_poses, n_blocks, n_blocks, pbt.max_n_conn, pbt.max_n_conn), CAP
    )
    partners = pose_stack.inter_residue_connections64.cpu().tolist()
    for pose, block_types in enumerate(pose_stack.block_type_ind64.cpu().tolist()):
        bts = {
            b: pbt.active_block_types[t] for b, t in enumerate(block_types) if t >= 0
        }
        nodes = [(b, c) for b, bt in bts.items() for c in range(len(bt.connections))]
        index = {node: i for i, node in enumerate(nodes)}
        dist = torch.full((len(nodes), len(nodes)), CAP)
        for i, (block, conn) in enumerate(nodes):
            atoms = bts[block].ordered_connection_atoms
            for conn2, atom2 in enumerate(atoms):
                sep = bts[block].path_distance[atoms[conn], atom2]
                dist[i, index[block, conn2]] = min(int(sep), CAP)
            if partners[pose][block][conn][0] >= 0:
                dist[i, index[tuple(partners[pose][block][conn])]] = 1
        for node in range(len(nodes)):
            dist = torch.minimum(dist, dist[:, node, None] + dist[None, node, :])
        block, conn = torch.tensor(nodes, dtype=torch.int64).reshape(-1, 2).T
        dense[pose, block[:, None], block, conn[:, None], conn] = dist.clamp(max=CAP)
    return dense.to(torch.int8)


@pytest.mark.parametrize(
    "structure",
    [
        "ubq_and_pertuzumab_stack",
        "oex_5xnl",
        "clf_nitrogenase_7adr",
        "sf4_ferredoxin_1fdn",
        "zn_tetrahedral_3ks3",
        "nco_zdna_1dn8",
        "heme_myoglobin_5yce",
    ],
)
def test_the_sparse_table_holds_every_short_path_and_only_near_blocks(
    structure, ubq_pdb, pertuzumab_pdb, torch_device
):
    if structure == "ubq_and_pertuzumab_stack":
        # the second pose has more blocks and more near blocks (its disulfide)
        pose_stack = PoseStackBuilder.from_poses(
            [
                pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=9),
                pose_stack_from_pdb(pertuzumab_pdb, torch_device, residue_end=100),
            ],
            torch_device,
        )
    elif structure == "oex_5xnl":
        pose_stack = oex_site_5xnl(torch_device)
    else:
        path = DATA / "metal_fixtures" / f"{structure}.cif.zst"
        pose_stack = heavy_pose_stack(atom_array_from_cif(path), torch_device)
    ibb = pose_stack.inter_block_bondsep
    dense = floyd_warshall_bondsep(pose_stack)

    assert ibb.bondsep.dtype == torch.int8
    torch.testing.assert_close(ibb.to_dense().cpu(), dense)
    # rows list their near blocks, then empty slots holding the cap
    near_blocks, bondsep = ibb.near_blocks.cpu(), ibb.bondsep.cpu()
    stored = near_blocks[..., 0] >= 0
    assert (stored[..., :-1] | ~stored[..., 1:]).all() and not stored[..., -1].any()
    assert (bondsep[~stored] == CAP).all() and (near_blocks[~stored][:, 1] == CAP).all()
    slab_min = torch.amin(bondsep, dim=(3, 4)).to(torch.int32)
    torch.testing.assert_close(near_blocks[..., 1], slab_min)
    near = torch.amin(dense, dim=(3, 4)) < CAP
    assert ibb.n_slots == int(near.sum(dim=2).max()) + 1
