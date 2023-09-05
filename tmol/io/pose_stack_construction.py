import torch
from tmol.types.torch import Tensor
from typing import Optional
from tmol.pose.pose_stack import PoseStack


def pose_stack_from_canonical_form(
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Optional[Tensor[torch.int32][:, :, :]] = None,
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    return_chain_ind: bool = False,
) -> PoseStack:
    """Create a PoseStack given atom coordinates in canonical ordering

    Arguments:
    chain_id: an n-pose x max-n-residue tensor with an index for which chain
        each residue belongs to. Residues belonging to the same chain must be
        consecutive.
    res_types: an n-pose x max-n-residue tensor with the canonically
        ordered amino acid index for each residue. A sentinel value of "-1"
        should be used to indicate that a given position does not contain
        a residue (perhaps because it belongs to a pose with fewer than
        max-n-residue residues).
    coords: an n-pose x max-n-residue x max-n-atoms-per-residue tensor
        providing the coordinates of some or all of the atoms. The order
        in which atoms should appear in this tensor is given in
        tmol.io.canonical_ordering.ordered_canonical_aa_atoms (see also
        tmol.io.canonical_ordering.ordered_canonical_aa_atoms_v2 for
        the older PDB v2 atom naming scheme). It is recommended that
        atoms whose coordinates are not being provided to this function
        have their coordinates marked as NaN to ensure that tmol is not
        reading from positions in this array that it should not be and
        thus delivering inaccurate energies.
    atom_is_present: an n-pose x max-n-residue x max-n-atoms-per-residue
        tensor answering the yes/no question of whether an atom's
        coordinate is being provided; 0 for no, 1 for yes. If this
        tensor is not provided, then any coordinates in the "coords"
        tensor with a coordinate of NaN will be taken as "not present"
        and all others (including any coordinates at the origin, e.g.)
        will be treated as if they "are present." Note: "present" here
        means "the coordinate is being provided" and not "this atom
        should be modeled;" conversely, there is no way to say "do not
        include a particular atom in tmol calculations."

        Currently, all heavy atoms must be provided to tmol except
        leaf atoms. A "leaf atom" is one that has no atoms that use it
        as a parent or grand parent when describing their icoors.
        Hydrogen atoms are all leaf atoms. Backbone carbonyl oxygens
        are also leaf atoms. Even thoguh hydrogen atoms are optional,
        the hydroxyl hydrogens on SER, THR, and TYR are recommended
        as tmol will build them suboptimally: at a dihedral of 180
        regardless of the presence of nearby hydrogen-bond acceptors

    disulfides: an optional n-total-disulfides x 3 tensor. If this argument
        is not provided, then the coordinates of SG atoms on CYS residues
        will be used to determine which pairs are closest to each other and
        within a cutoff distance of 2.5A and declare those SGs as forming
        disulfide bonds. This means that SGs slightly longer than 2.5A
        will not be detected. If you should know which pairs of cysteines
        form disulfide bonds, then you can provide their pairing:
        [ [pose_ind, cys1_ind, cys2_ind], ...]. If you provide this argument
        then no other disulfides will be sought and the disulfide-detection
        step will be skipped.

    res_not_connected: an optional input used to indicate that a given (polymeric)
        residue is not connected to either its previous or next residue; for
        termini residues, they will not be built with their termini-variant
        types. The purpose is to allow the user to include a subset of the
        residues in a protein where a series of "gap" residues can be omitted
        between i and i+1 without those two residues being treated as if they
        are chemically bonded. This will keep the Ramachandran term from scoring
        nonsense dihdral angles and will keep the cart-bonded term from scoring
        nonsense bond lengths and angles.

    return_chain_ind: return the chain-index tensor that has been "left-justified"
        from the chain
    """

    from tmol.io.details.canonical_packed_block_types import (
        default_canonical_packed_block_types,
    )
    from tmol.io.details.left_justify_canonical_form import left_justify_canonical_form
    from tmol.io.details.disulfide_search import find_disulfides
    from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
    from tmol.io.details.select_from_canonical import (
        assign_block_types,
        take_block_type_atoms_from_canonical,
    )
    from tmol.io.details.build_missing_leaf_atoms import build_missing_leaf_atoms

    assert chain_id.device == res_types.device
    assert chain_id.device == coords.device

    # step 1: retrieve the global packed_block_types object with the 66
    #         canonical residue types
    # step 2: resolve disulfides
    # step 3: resolve his tautomer
    # step 4: resolve termini variants, assign block-types to each input
    #         residue, and populate the inter-block connectivity tensors
    # step 5: select the atoms from the canonically-ordered input tensors
    #         (the coords and atom_is_present tensors) that belong to the
    #         now-assigned block types, discarding/ignoring
    #         any others that may have been provided
    # step 6: if any atoms missing, build them
    # step 7: construct PoseStack object

    if atom_is_present is None:
        atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    # 1
    # this will return the same object each time to minimize the number
    # of calls to the setup_packed_block_types annotation functions
    pbt, atom_type_resolver = default_canonical_packed_block_types(chain_id.device)

    # 2
    # "left justify" the input canonical-form residues: residues that are
    (
        chain_id,
        res_types,
        coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = left_justify_canonical_form(
        chain_id, res_types, coords, atom_is_present, disulfides, res_not_connected
    )

    # 2
    found_disulfides, res_type_variants = find_disulfides(res_types, coords, disulfides)

    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    # 4
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep,
    ) = assign_block_types(
        pbt, chain_id, res_types, res_type_variants, found_disulfides, res_not_connected
    )

    # 5
    block_coords, missing_atoms, real_atoms = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, atom_is_present
    )

    # 6
    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    pose_stack_coords, block_coord_offset = build_missing_leaf_atoms(
        pbt,
        atom_type_resolver,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
    )

    def i64(x):
        return x.to(torch.int64)

    def i32(x):
        return x.to(torch.int32)

    # 7
    ps = PoseStack(
        packed_block_types=pbt,
        coords=pose_stack_coords,
        block_coord_offset=block_coord_offset,
        block_coord_offset64=i64(block_coord_offset),
        inter_residue_connections=inter_residue_connections,
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=inter_block_bondsep,
        inter_block_bondsep64=i64(inter_block_bondsep),
        block_type_ind=i32(block_types64),
        block_type_ind64=block_types64,
        device=pbt.device,
    )
    # if return_chain_ind:
    #     ps_chain_ind = torch.full_like(block_types64, -1)
    #     ps_chain_ind[block_type_ind64 != -1] =
    #     return (ps,
    return ps
