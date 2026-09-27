import torch
import numpy

from tmol.types import (
    Tensor,
    NDArray,
    validate_args,
)
from typing import Optional
from tmol.database.chemical import GEOMETRY_NAMES
from tmol.pose import (
    PDBInfo,
    DEFAULT_ATOM_OCCUPANCY,
    DEFAULT_ATOM_B_FACTOR,
    PoseStack,
    PackedBlockTypes,
)
from tmol.io import CanonicalOrdering


@validate_args
def pose_stack_from_canonical_form(  # noqa: C901
    canonical_ordering: CanonicalOrdering,
    pbt: PackedBlockTypes,
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    res_labels: Optional[NDArray[int][:, :]],
    res_ins_codes: Optional[NDArray[object][:, :]],
    chain_labels: Optional[NDArray[object][:, :]],
    atom_occupancy: Optional[NDArray[numpy.float32][:, :, :]] = None,
    atom_b_factor: Optional[NDArray[numpy.float32][:, :, :]] = None,
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    cyclic_bonds: Optional[Tensor[torch.int64][:, 3]] = None,
    covalent_bonds: Optional[Tensor[torch.int64][:, 5]] = None,
    metal_sites: Optional[Tensor[torch.int64][:, 3]] = None,
    metal_coordination: Optional[Tensor[torch.int64][:, 5]] = None,
    metal_origins: Optional[NDArray[object][:, :]] = None,
    *,
    trust_hydrogen_names: bool = False,
    find_additional_disulfides: Optional[bool] = True,
    find_additional_cyclic_closures: Optional[bool] = True,
    find_additional_metal_coordination: bool = True,
    return_chain_ind: bool = False,
    return_atom_mapping: bool = False,
    return_block_has_missing_atoms: bool = False,
    packer_atoms=None,
):
    """Build a pose stack from tensors in canonical atom ordering.

    Residue variants are selected from the atoms present in ``coords``. A NaN
    coordinate marks an absent input atom; TMol builds missing leaf atoms, but
    requires non-leaf atoms unless ``return_block_has_missing_atoms`` is set.

    Args:
        canonical_ordering: Residue and atom ordering used by the input tensors.
        pbt: Packed residue types and score-term annotations for the new pose.
        chain_id: Chain index for each ``[pose, residue]``; residues in a chain
            must be consecutive.
        res_types: Canonical residue-type index for each ``[pose, residue]``;
            ``-1`` marks padding.
        coords: Coordinates shaped ``[pose, residue, canonical_atom, xyz]``.
        res_labels: Optional source residue numbers shaped ``[pose, residue]``.
        res_ins_codes: Optional source insertion codes shaped ``[pose, residue]``.
        chain_labels: Optional source chain labels shaped ``[pose, residue]``.
        atom_occupancy: Optional source occupancies in canonical atom ordering.
        atom_b_factor: Optional source B-factors in canonical atom ordering.
        disulfides: Explicit ``[pose, cys1, cys2]`` rows. If omitted, nearby
            cysteine SG atoms are paired geometrically.
        res_not_connected: Flags shaped ``[pose, residue, direction]`` for
            polymer connections absent before or after each residue.
        find_additional_disulfides: Detect geometrically plausible disulfides
            not included in ``disulfides``.
        covalent_bonds: Explicit ``[pose, res1, atom1, res2, atom2]`` rows in
            canonical ordering, with matching ports on prepared block types.
        cyclic_bonds: Explicit ``[pose, up_res, down_res]`` polymer closures.
            If omitted, terminal connection atoms within 2 A imply a closure.
        find_additional_cyclic_closures: Detect closures absent from cyclic_bonds.
        metal_sites: Explicit ``[pose, metal, geometry]`` rows, geometry an index
            into GEOMETRY_NAMES. A listed metal is not detected.
        metal_coordination: Explicit ``[pose, metal, site, donor, atom]`` rows
            for listed metals, atom in canonical ordering.
        metal_origins: Per residue, where a metal split out of a component
            sat, kept in pdb_info so export can put it back.
        find_additional_metal_coordination: Detect the geometry and donors of
            metals absent from metal_sites; otherwise they take their default
            geometry with every site open.
        trust_hydrogen_names: Preserve generated-residue hydrogens whose names
            match the prepared database, for example in canonical roundtrips.
        return_chain_ind: Include the left-justified ``chain_ind`` tensor.
        return_atom_mapping: Include ``can_atom_mapping`` and
            ``ps_atom_mapping`` tensors between canonical and pose atom order.
        return_block_has_missing_atoms: Include a ``[pose, residue]`` mask for
            blocks missing non-leaf input atoms instead of rejecting them.
        packer_atoms: Given a block type, the atoms rotamer packing will place;
            missing ones are left for the packer instead of built here.

    Returns:
        The pose stack. If any return flag is set, returns ``(pose_stack,
        metadata)`` with the requested tensors in ``metadata``.
    """

    from tmol.io.details import left_justify_canonical_form
    from tmol.io.details import find_cyclic_closures, find_disulfides
    from tmol.io.details._metal_detection import (
        donor_patches,
        find_metal_geometries,
        metal_connection_rows,
        place_site_virtuals,
        select_donor_variants,
        with_donor_patches,
    )
    from tmol.io.details import resolve_his_tautomerization
    from tmol.io.details import (
        assign_block_types,
        take_block_type_atoms_from_canonical,
    )
    from tmol.io.details import build_missing_leaf_atoms

    assert chain_id.device == res_types.device
    assert chain_id.device == coords.device

    assert chain_id.shape[0] == res_types.shape[0]
    assert chain_id.shape[1] == res_types.shape[1]
    assert chain_id.shape[0] == coords.shape[0]
    assert chain_id.shape[1] == coords.shape[1]
    assert coords.shape[2] == canonical_ordering.max_n_canonical_atoms
    assert res_labels is None or res_labels.shape[0] == chain_id.shape[0]
    assert res_labels is None or res_labels.shape[1] == chain_id.shape[1]
    assert res_ins_codes is None or res_ins_codes.shape[0] == chain_id.shape[0]
    assert res_ins_codes is None or res_ins_codes.shape[1] == chain_id.shape[1]
    assert chain_labels is None or chain_labels.shape[0] == chain_id.shape[0]
    assert chain_labels is None or chain_labels.shape[1] == chain_id.shape[1]
    assert atom_occupancy is None or atom_occupancy.shape[0] == chain_id.shape[0]
    assert atom_occupancy is None or atom_occupancy.shape[1] == chain_id.shape[1]
    assert atom_b_factor is None or atom_b_factor.shape[0] == chain_id.shape[0]
    assert atom_b_factor is None or atom_b_factor.shape[1] == chain_id.shape[1]
    assert res_not_connected is None or res_not_connected.shape[0] == chain_id.shape[0]
    assert res_not_connected is None or res_not_connected.shape[1] == chain_id.shape[1]

    # step 1: record which atoms the user has given us by looking for NaNs
    #         in the input coordinate tensor.
    # step 2: remove any "virtual residues," marked with a res-type ind of -1
    #         by shifting all of the residues in each Pose "to the left"
    # step 3: resolve disulfides and cyclic-chain closures
    # step 4: resolve his tautomer, then coordinating variants and the donor
    #         forms metal connections need
    # step 5: resolve termini variants, assign block-types to each input
    #         residue, and populate the inter-block connectivity tensors
    # step 6: select the atoms from the canonically-ordered input tensors
    #         (the coords and atom_is_present tensors) that belong to the
    #         now-assigned block types, discarding/ignoring
    #         any others that may have been provided
    # step 7: if any atoms missing, build them
    # step 8: construct PoseStack object
    # step 9: construct the forward/reverse atom mapping indices if required

    # 1: look for NaNs in the input coordinates tensor
    atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    declared_metal_sites, required_metal_donors = _declared_metal_sites(
        res_types, metal_sites, metal_coordination
    )

    # 2
    # "left justify" the input canonical-form residues: residues that are
    # given with a "-1" residue-type should be excised from the center of
    # their Poses to ensure that the polymeric-bond-detection logic
    # downstream will work properly. This effectively means "shifting left"
    # all the other residues in the Pose to fill the vacated slots.
    # Single-slot poses are already left-justified.
    if res_types.shape[1] != 1:
        (
            chain_id,
            res_types,
            coords,
            atom_is_present,
            disulfides,
            cyclic_bonds,
            covalent_bonds,
            res_not_connected,
            res_labels,
            res_ins_codes,
            chain_labels,
            atom_occupancy,
            atom_b_factor,
        ) = left_justify_canonical_form(
            chain_id,
            res_types,
            coords,
            atom_is_present,
            disulfides,
            cyclic_bonds,
            covalent_bonds,
            res_not_connected,
            res_labels,
            res_ins_codes,
            chain_labels,
            atom_occupancy,
            atom_b_factor,
        )

    if res_types.shape[1] == 0:
        raise ValueError(
            "pose_stack_from_canonical_form: no recognized residues found in input. "
            "Check that residue names match entries in the CanonicalOrdering and that "
            "the structure was parsed via biotite (HETATM records are supported) rather "
            "than the internal PDB parser (ATOM records only)."
        )

    # 3
    if res_types.shape[1] == 1 and disulfides is None:
        found_disulfides = torch.zeros(
            (0, 3), dtype=torch.int64, device=res_types.device
        )
        res_type_variants = torch.zeros_like(res_types)
    else:
        found_disulfides, res_type_variants = find_disulfides(
            canonical_ordering,
            res_types,
            coords,
            disulfides,
            find_additional_disulfides,
        )

    # 3a: a metal's geometry cannot be read from the atoms it presents -- every
    #     geometry of an ion shows the same single atom -- so it rides the same
    #     variant axis a disulfide does. Metals and cysteines never collide.
    #     A disulfide cysteine's state is already fixed, so it cannot donate.
    metal_variants, metal_assignments = find_metal_geometries(
        canonical_ordering,
        pbt.chem_db,
        res_types,
        coords,
        excluded_donor_residues=res_type_variants != 0,
        declared_sites=declared_metal_sites,
        find_additional=find_additional_metal_coordination,
        required_donors=required_metal_donors,
    )
    res_type_variants = torch.where(
        metal_variants != 0, metal_variants, res_type_variants
    )

    # 3b
    cyclic_closures = find_cyclic_closures(
        canonical_ordering,
        chain_id,
        res_types,
        coords,
        cyclic_bonds,
        find_additional_cyclic_closures,
    )

    # 4
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        canonical_ordering, res_types, res_type_variants, coords, atom_is_present
    )

    # 4a: a donor must be able to donate: coordination selects the deprotonated
    #     form or the other histidine tautomer. After 4, which rewrites every
    #     histidine's variant.
    res_type_variants = select_donor_variants(
        canonical_ordering, pbt.chem_db, res_types, res_type_variants, metal_assignments
    )

    # 4b: a coordinating atom needs a connection its metal can fill; those
    #     forms are made for the donors this input has, not ahead of time
    pbt = with_donor_patches(
        pbt,
        donor_patches(canonical_ordering, pbt.chem_db, res_types, metal_assignments),
    )

    # 5
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep,
    ) = assign_block_types(
        canonical_ordering,
        pbt,
        resolved_atom_is_present,
        chain_id,
        res_types,
        res_type_variants,
        found_disulfides,
        res_not_connected,
        cyclic_closures,
        covalent_bonds,
        metal_connection_rows(metal_assignments),
    )

    # 6
    (
        block_coords,
        missing_atoms,
        real_atoms,
        real_canonical_atom_inds,
        atom_occupancy,
        atom_b_factor,
    ) = take_block_type_atoms_from_canonical(
        pbt,
        block_types64,
        coords,
        atom_is_present,
        atom_occupancy,
        atom_b_factor,
        trust_hydrogen_names=trust_hydrogen_names,
    )

    # 6a: a free ion's site virtuals have no frame to build from; orient them
    #     from the fitted geometry
    block_coords, missing_atoms = place_site_virtuals(
        pbt, block_types64, block_coords, missing_atoms, metal_assignments
    )

    # 7
    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    (
        pose_stack_coords,
        block_coord_offset,
        real_block_atoms,
        pose_at_is_real,
        block_has_missing_atoms,
    ) = build_missing_leaf_atoms(
        pbt,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
        fail_on_missing_nonleaf_atoms=not return_block_has_missing_atoms,
        packer_atoms=packer_atoms,
    )

    def i64(x):
        return x.to(torch.int64)

    def i32(x):
        return x.to(torch.int32)

    # 8
    if atom_occupancy is not None or atom_b_factor is not None:
        real_block_atoms = real_block_atoms.cpu().numpy()
        pose_at_is_real = pose_at_is_real.cpu().numpy()
    atom_occupancy_pose_layout = numpy.full(
        pose_stack_coords.shape[:2], DEFAULT_ATOM_OCCUPANCY, dtype=numpy.float32
    )
    if atom_occupancy is not None:
        atom_occupancy_pose_layout[pose_at_is_real] = atom_occupancy[real_block_atoms]
    atom_b_factor_pose_layout = numpy.full(
        pose_stack_coords.shape[:2], DEFAULT_ATOM_B_FACTOR, dtype=numpy.float32
    )
    if atom_b_factor is not None:
        atom_b_factor_pose_layout[pose_at_is_real] = atom_b_factor[real_block_atoms]

    pdb_info = PDBInfo(
        residue_labels=res_labels,
        residue_insertion_codes=res_ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=atom_occupancy_pose_layout,
        atom_b_factor=atom_b_factor_pose_layout,
        metal_origins=metal_origins,
    )

    block_coord_offset64 = i64(block_coord_offset)
    ps = PoseStack(
        packed_block_types=pbt,
        coords=pose_stack_coords,
        block_coord_offset=block_coord_offset,
        block_coord_offset64=block_coord_offset64,
        inter_residue_connections=inter_residue_connections,
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=inter_block_bondsep,
        block_type_ind=i32(block_types64),
        block_type_ind64=block_types64,
        chain_id=chain_id,
        chain_id64=i64(chain_id),
        pdb_info=pdb_info,
        constraint_set=None,
        device=pbt.device,
    )

    # 9
    if return_atom_mapping:
        (
            nz_block_layout_pose_ind,
            nz_block_layout_block_ind,
            nz_block_at_ind,
        ) = torch.nonzero(real_atoms, as_tuple=True)
        pose_atom_ind = (
            block_coord_offset64[nz_block_layout_pose_ind, nz_block_layout_block_ind]
            + nz_block_at_ind
        )

        def _u1(x):
            return x.unsqueeze(1)

        can_atom_mapping = torch.cat(
            (
                _u1(nz_block_layout_pose_ind),
                _u1(nz_block_layout_block_ind),
                _u1(real_canonical_atom_inds),
            ),
            dim=1,
        )
        ps_atom_mapping = torch.cat(
            (
                _u1(nz_block_layout_pose_ind),
                _u1(pose_atom_ind),
            ),
            dim=1,
        )

    # return the optional arguments in a dictionary
    opt_return_vals = {}
    if return_chain_ind:
        opt_return_vals["chain_ind"] = chain_id
    if return_atom_mapping:
        opt_return_vals["can_atom_mapping"] = can_atom_mapping
        opt_return_vals["ps_atom_mapping"] = ps_atom_mapping
    if return_block_has_missing_atoms:
        opt_return_vals["block_has_missing_atoms"] = block_has_missing_atoms

    if len(opt_return_vals) > 0:
        return ps, opt_return_vals
    return ps


def _declared_metal_sites(res_types, metal_sites, metal_coordination):
    """Declared metal sites and declared metal bonds, left-justified.

    Returns ({(pose, metal): (geometry, ((site, donor, atom), ...))},
    {(pose, metal): {(donor, atom), ...}}). A bond is placed at its site only
    when the metal's geometry is declared too; otherwise it is a required donor
    and detection chooses the site.
    """
    # tmol.io.details imports this module's package, as the builder above does
    from tmol.io.details import left_justify_residue_indices

    if res_types.shape[1] != 1:
        if metal_sites is not None:
            metal_sites = left_justify_residue_indices(res_types, metal_sites, (1,))
        if metal_coordination is not None:
            metal_coordination = left_justify_residue_indices(
                res_types, metal_coordination, (1, 3)
            )
    geometry_for = {
        (pose, metal): GEOMETRY_NAMES[geometry]
        for pose, metal, geometry in (
            metal_sites.tolist() if metal_sites is not None else []
        )
    }
    filled, required = {}, {}
    for pose, metal, site, donor, atom in (
        metal_coordination.tolist() if metal_coordination is not None else []
    ):
        if (pose, metal) in geometry_for and site >= 0:
            filled.setdefault((pose, metal), []).append((site, donor, atom))
        else:
            required.setdefault((pose, metal), set()).add((donor, atom))
    declared = {
        key: (geometry, tuple(filled.get(key, ())))
        for key, geometry in geometry_for.items()
    }
    return declared, required
