import copy

import attr
import torch
import numpy
import toolz
import biotite
import biotite.structure
import logging

from tmol.types import validate_args
from tmol.chemical import ResidueTypeSet, get_element_from_atom_name
from tmol.database import ParameterDatabase
from tmol.io import (
    CanonicalForm,
    CanonicalOrdering,
    canonical_form_from_pose_stack,
    PoseBuildContext,
)
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
    DEFAULT_ATOM_B_FACTOR,
    DEFAULT_ATOM_OCCUPANCY,
)
from tmol.utility import (
    get_all_residue_positions,
    resolve_device,
)

logger = logging.getLogger(__name__)

_MAX_PREPARED_BATCH_SIZES = 4


class Atom37MappingError(ValueError):
    """An AtomArray cannot be routed unambiguously into an Atom37 tensor."""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _PreparedAtom37PoseTopology:
    """Fixed pose layout plus the masks needed to rebuild missing leaf atoms."""

    pose_stack: PoseStack
    canonical_atom_mapping: torch.Tensor
    pose_atom_mapping: torch.Tensor
    block_leaf_atom_is_missing: torch.Tensor
    pose_atom_is_missing: torch.Tensor
    block_has_missing_atoms: torch.Tensor
    real_atoms: torch.Tensor

    @classmethod
    def from_pose(
        cls,
        pose_stack: PoseStack,
        canonical_coords: torch.Tensor,
        canonical_atom_mapping: torch.Tensor,
        pose_atom_mapping: torch.Tensor,
        block_has_missing_atoms: torch.Tensor,
    ) -> "_PreparedAtom37PoseTopology":
        canonical_atom_mapping = canonical_atom_mapping.to(torch.int64)
        pose_atom_mapping = pose_atom_mapping.to(torch.int64)
        source_coords = canonical_coords[
            canonical_atom_mapping[:, 0],
            canonical_atom_mapping[:, 1],
            canonical_atom_mapping[:, 2],
        ]
        missing = ~torch.isfinite(source_coords).all(dim=-1)
        pose_ind = pose_atom_mapping[:, 0]
        pose_atom = pose_atom_mapping[:, 1]
        block_ind = canonical_atom_mapping[:, 1]
        block_atom = pose_atom - pose_stack.block_coord_offset64[pose_ind, block_ind]

        block_leaf_atom_is_missing = torch.zeros(
            (
                pose_stack.n_poses,
                pose_stack.max_n_blocks,
                pose_stack.max_n_block_atoms,
            ),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        block_leaf_atom_is_missing[pose_ind, block_ind, block_atom] = missing
        pose_atom_is_missing = torch.zeros(
            pose_stack.coords.shape[:2], dtype=torch.bool, device=pose_stack.device
        )
        pose_atom_is_missing[pose_ind, pose_atom] = missing
        canonical_atom_mapping = canonical_atom_mapping[~missing]
        pose_atom_mapping = pose_atom_mapping[~missing]
        pose_template = pose_stack.clone()
        return cls(
            pose_stack=pose_template,
            canonical_atom_mapping=canonical_atom_mapping,
            pose_atom_mapping=pose_atom_mapping,
            block_leaf_atom_is_missing=block_leaf_atom_is_missing,
            pose_atom_is_missing=pose_atom_is_missing,
            block_has_missing_atoms=block_has_missing_atoms,
            real_atoms=pose_stack.real_atoms,
        )

    def pose_from_canonical(self, canonical_coords: torch.Tensor) -> PoseStack:
        """Rebind coordinates and rebuild leaf atoms in the prepared pose layout."""
        mapping = self.canonical_atom_mapping
        pose_mapping = self.pose_atom_mapping
        source_coords = canonical_coords[mapping[:, 0], mapping[:, 1], mapping[:, 2]]
        coords = torch.zeros_like(self.pose_stack.coords)
        coords[pose_mapping[:, 0], pose_mapping[:, 1]] = source_coords

        pbt = self.pose_stack.packed_block_types
        from tmol.io.details._build_missing_leaf_atoms import (
            _apply_h_geometric_completion,
        )
        from tmol.io.details.compiled import gen_pose_leaf_atoms

        coords = gen_pose_leaf_atoms(
            coords,
            self.block_leaf_atom_is_missing,
            self.pose_atom_is_missing,
            self.pose_stack.block_coord_offset,
            self.pose_stack.block_type_ind,
            self.pose_stack.inter_residue_connections,
            pbt.n_atoms,
            pbt.atom_downstream_of_conn,
            pbt.build_missing_leaf_atom_icoor_ann.anc_uaids,
            pbt.build_missing_leaf_atom_icoor_ann.geom,
            pbt.build_missing_leaf_atom_icoor_ann.anc_uaids_backup,
            pbt.build_missing_leaf_atom_icoor_ann.geom_backup,
        )
        coords = _apply_h_geometric_completion(
            pbt,
            coords,
            self.block_leaf_atom_is_missing,
            self.pose_stack.block_coord_offset,
            self.pose_stack.block_type_ind,
        )
        pose_stack = copy.copy(self.pose_stack)
        pose_stack.coords = coords
        return pose_stack


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PreparedAtom37PoseBuilder:
    """Bind immutable Biotite topology for repeated Atom37 pose construction."""

    context: PoseBuildContext
    canonical_template: CanonicalForm
    mapped_token_id: torch.Tensor
    mapped_slot: torch.Tensor
    mapped_residue: torch.Tensor
    mapped_atom: torch.Tensor
    max_token_id: int
    fragment_mapping: object | None = None
    _topology_cache_safe: bool = True
    _pose_topologies: dict[int, _PreparedAtom37PoseTopology] = attr.ib(
        factory=dict, eq=False, repr=False
    )

    def __call__(
        self,
        atom37_coords: torch.Tensor,
        *,
        opt_h: bool = True,
    ) -> PoseStack:
        """Build a differentiable pose batch; optimize hydrogens by default."""
        cf = self._canonical_form(atom37_coords)
        if not self._topology_cache_safe:
            return _pose_stack_from_canonical_and_context(
                cf,
                self.context,
                no_optH=not opt_h,
                atom37_coords=atom37_coords,
                fragment_mapping=self.fragment_mapping,
            )
        n_poses = atom37_coords.shape[0]
        topology = self._pose_topologies.pop(n_poses, None)
        topology_was_cached = topology is not None
        if topology is None:
            pose_stack, details = _pose_stack_from_canonical_and_context(
                cf,
                self.context,
                no_optH=True,
                atom37_coords=atom37_coords,
                fragment_mapping=self.fragment_mapping,
                return_atom_mapping=True,
            )
            block_has_missing_atoms = details["block_has_missing_atoms"]
            if bool(torch.any(block_has_missing_atoms)):
                if not opt_h:
                    return pose_stack
                return _pose_stack_from_canonical_and_context(
                    cf,
                    self.context,
                    no_optH=False,
                    atom37_coords=atom37_coords,
                    fragment_mapping=self.fragment_mapping,
                )
            topology = _PreparedAtom37PoseTopology.from_pose(
                pose_stack,
                cf.coords,
                details["can_atom_mapping"],
                details["ps_atom_mapping"],
                block_has_missing_atoms,
            )
        else:
            pose_stack = topology.pose_from_canonical(cf.coords)
        if len(self._pose_topologies) == _MAX_PREPARED_BATCH_SIZES:
            self._pose_topologies.pop(next(iter(self._pose_topologies)))
        self._pose_topologies[n_poses] = topology

        if opt_h:
            from tmol.pack import build_missing_sidechains

            pose_stack = build_missing_sidechains(
                pose_stack,
                self.context._opth_score_function,
                self.context._dunbrack_sampler,
                topology.block_has_missing_atoms,
                no_optH=False,
            )
            pose_stack = _restore_canonical_input_coords(
                pose_stack,
                cf.coords,
                topology.canonical_atom_mapping,
                topology.pose_atom_mapping,
                mappings_are_finite=True,
            )
        # Pose construction validates the initial topology. Check the initial
        # packed result too, but do not force a CUDA-to-host synchronization on
        # every replay of an already validated fixed topology.
        if opt_h and not topology_was_cached:
            _assert_no_nan_coords(pose_stack, topology.real_atoms)
        return pose_stack

    def _canonical_form(self, atom37_coords: torch.Tensor) -> CanonicalForm:
        device = self.context.packed_block_types.device
        _validate_atom37_coords(atom37_coords, device)
        if self.max_token_id >= atom37_coords.shape[1]:
            raise Atom37MappingError(
                f"token_id {self.max_token_id} exceeds atom37_coords token count "
                f"{atom37_coords.shape[1]}"
            )

        template = self.canonical_template
        n_poses = atom37_coords.shape[0]
        template_n_poses = template.coords.shape[0]
        if template_n_poses not in (1, n_poses):
            raise ValueError(
                f"Biotite structure has {template_n_poses} poses but "
                f"atom37_coords has {n_poses}"
            )

        def tensor_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return value.expand(n_poses, *value.shape[1:]).clone()

        def array_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return numpy.repeat(value, n_poses, axis=0)

        coords = template.coords
        if template_n_poses == n_poses:
            coords = coords.clone()
        else:
            coords = coords.expand(n_poses, *coords.shape[1:]).clone()
        source_coords = atom37_coords[:, self.mapped_token_id, self.mapped_slot]
        finite = torch.isfinite(source_coords).all(dim=-1)
        coords[:, self.mapped_residue, self.mapped_atom] = torch.where(
            finite.unsqueeze(-1),
            source_coords,
            coords[:, self.mapped_residue, self.mapped_atom],
        )
        return CanonicalForm(
            chain_id=tensor_for_poses(template.chain_id),
            res_types=tensor_for_poses(template.res_types),
            coords=coords,
            res_labels=array_for_poses(template.res_labels),
            residue_insertion_codes=array_for_poses(template.residue_insertion_codes),
            chain_labels=array_for_poses(template.chain_labels),
            atom_occupancy=array_for_poses(template.atom_occupancy),
            atom_b_factor=array_for_poses(template.atom_b_factor),
            disulfides=template.disulfides,
            res_not_connected=tensor_for_poses(template.res_not_connected),
        )


@validate_args
def prepare_pose_stack_from_atom37(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    context: PoseBuildContext,
) -> PreparedAtom37PoseBuilder:
    """Prepare a callable for repeatedly binding Atom37 coordinates to topology.

    This is the campaign-oriented counterpart to
    :func:`pose_stack_from_atom37_and_biotite`: immutable residue identity,
    connectivity, fragmentation, and Atom37 routing are resolved once. Calling
    the returned builder with a coordinate tensor constructs a differentiable
    pose while retaining TMol's usual missing-atom behavior. The returned
    builder optimizes hydrogens by default; pass ``opt_h=False`` to disable it.
    """
    device = context.packed_block_types.device
    fragment_mapping = None
    if context.fragment_definitions:
        from tmol.ligand import expand_fragmented_ligands

        biotite_structure, fragment_mapping = expand_fragmented_ligands(
            biotite_structure, context.fragment_definitions
        )

    canonical_template = canonical_form_from_biotite(
        biotite_structure,
        device,
        co=context.canonical_ordering,
        missing_density_distance_threshold=0.0,
    )
    filtered, _ = _filter_supported_atoms_and_connectivity(
        biotite_structure, context.canonical_ordering
    )
    atom_residue = get_all_residue_positions(filtered)
    valid_mask, valid_atom, valid_residue = _map_atoms_to_canonical(
        context.canonical_ordering,
        atom_residue,
        filtered.res_name,
        filtered.atom_name,
    )
    token_id, slot, mapped_residue, mapped_atom = _atom37_mapping(
        filtered, valid_mask, valid_residue, valid_atom
    )
    mapped_token_id = torch.as_tensor(token_id, device=device)
    mapped_slot = torch.as_tensor(slot, device=device)
    mapped_residue = torch.as_tensor(mapped_residue, device=device)
    mapped_atom = torch.as_tensor(mapped_atom, device=device)
    mapped_reference_coords = canonical_template.coords[
        :,
        mapped_residue,
        mapped_atom,
    ]
    his_inds = context.canonical_ordering.his_inds
    ambiguous_his = False
    if his_inds.his_co_aa_ind >= 0:
        ambiguous_atom_inds = torch.tensor(
            [his_inds.his_HN_in_co, his_inds.his_NH_in_co, his_inds.his_NN_in_co],
            device=device,
        )
        is_his = canonical_template.res_types == his_inds.his_co_aa_ind
        ambiguous_his = bool(
            torch.any(
                is_his.unsqueeze(-1)
                & torch.isfinite(
                    canonical_template.coords[:, :, ambiguous_atom_inds]
                ).all(dim=-1)
            )
        )
    return PreparedAtom37PoseBuilder(
        context=context,
        canonical_template=canonical_template,
        mapped_token_id=mapped_token_id,
        mapped_slot=mapped_slot,
        mapped_residue=mapped_residue,
        mapped_atom=mapped_atom,
        max_token_id=int(token_id.max()),
        fragment_mapping=fragment_mapping,
        topology_cache_safe=(
            bool(torch.isfinite(mapped_reference_coords).all()) and not ambiguous_his
        ),
    )


@validate_args
def build_context_from_biotite(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    param_db: ParameterDatabase | None = None,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    ligand_params_files: list[str] | None = None,
    sample_proton_chi: bool = True,
) -> PoseBuildContext:
    """Build the structure-independent construction context.

    The returned context holds only database/ligand-derived pieces (canonical
    ordering, residue-type set, packed block types, parameter database); it does
    not depend on the input structure's coordinates and can be reused across
    structures sharing the same ligand(s). ``biotite_structure`` is used only to
    detect and prepare ligands (when ``prepare_ligands=True``).

    Args:
        biotite_structure: Input AtomArray or AtomArrayStack. Used only for
            ligand detection/preparation when ``prepare_ligands=True``.
        torch_device: Target torch device.
        param_db: Optional parameter database. When provided, canonical ordering,
            residue types, and packed block types are built from this database.
            If prepare_ligands=True, it is extended with ligand data. If None,
            defaults are used.
        prepare_ligands: If True, detect and prepare non-standard residues
            (via ``tmol.ligand``, which uses RDKit for atom typing and
            residue-type construction).
        ligand_ph: Target pH for ligand protonation (default 7.4, only used when
            prepare_ligands=True).
        strict_atom_types: If True, unknown ligand atom types raise errors
            instead of using a fallback element heuristic.
        strict_ligands: If True (default), raise when a detected ligand cannot
            be prepared and registered (instead of silently dropping it during
            pose construction). Pass False to fall back to warn-and-skip. Only
            used when prepare_ligands=True.
        ligand_params_files: Optional list of tmol YAML params file paths.
            Residues defined in these files skip the RDKit/OB pipeline.
        sample_proton_chi: If True, prepared ligands emit PROTON_CHI
            ``chi_samples`` for polar-hydrogen rotations (driving OptHSampler).
            Enabled by default; pass False to suppress proton-chi samples. Only
            used when prepare_ligands=True.

    Returns:
        PoseBuildContext containing canonical ordering, packed block
        types, parameter database, and residue type set.
    """
    torch_device = resolve_device(torch_device)
    if prepare_ligands:
        from tmol.ligand import prepare_ligands as _prepare_ligands

        using_default_database = param_db is None
        if param_db is None:
            param_db = ParameterDatabase.get_default()

        param_db, co, fragment_definitions = _prepare_ligands(
            biotite_structure,
            param_db=param_db,
            ph=ligand_ph,
            strict_atom_types=strict_atom_types,
            params_files=ligand_params_files,
            sample_proton_chi=sample_proton_chi,
            strict_ligands=strict_ligands,
            return_fragment_definitions=True,
        )
        if (
            using_default_database
            and param_db is _paramdb_for_biotite()
            and not fragment_definitions
        ):
            return _default_pose_build_context(torch_device)

        rts = ResidueTypeSet.from_database(param_db.chemical)
        pbt = PackedBlockTypes.from_restype_list(
            rts.chem_db, rts, rts.residue_types, torch_device
        )
        return PoseBuildContext(
            canonical_ordering=co,
            packed_block_types=pbt,
            parameter_database=param_db,
            restype_set=rts,
            fragment_definitions=fragment_definitions,
        )

    if param_db is None:
        return _default_pose_build_context(torch_device)

    db = param_db
    co, rts, pbt = _derived_types_for_param_db(db, torch_device)
    return PoseBuildContext(
        canonical_ordering=co,
        packed_block_types=pbt,
        parameter_database=db,
        restype_set=rts,
    )


@validate_args
def pose_stack_from_biotite(  # noqa: C901
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    param_db: ParameterDatabase | None = None,
    missing_density_distance_threshold: float = 2.4,
    no_optH: bool = False,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    ligand_params_files: list[str] | None = None,
    sample_proton_chi: bool = True,
    return_context: bool = False,
    context: PoseBuildContext | None = None,
    atom37_coords: torch.Tensor | None = None,
    **kwargs: object,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Build a PoseStack from the output generated by Biotite.

    To score many structures that share the same ligand(s) efficiently, build
    the (expensive, structure-independent) context once and reuse it::

        context = build_context_from_biotite(struct0, dev, prepare_ligands=True)
        for struct in structures:
            pose_stack = pose_stack_from_biotite(struct, dev, context=context)

    Reusing a context skips rebuilding the parameter database, canonical
    ordering, residue-type set, and packed block types; only the per-structure
    canonical form is recomputed (see the ``context`` arg).

    Args:
        biotite_structure: A Biotite AtomArray or AtomArrayStack.
        torch_device: Target PyTorch device.
        param_db: Optional ParameterDatabase. When provided, conversion and pose
            construction use this database. If prepare_ligands=True, it is
            extended with ligand data. Mutually exclusive with ``context``.
        missing_density_distance_threshold: Distance threshold in Angstroms.
            Adjacent residues whose closest inter-atom distance exceeds this
            value are treated as disconnected (upper/lower connects broken).
            Set to 0 to disable. Default is 2.4.
        no_optH: When False (default), all residues with complete heavy atoms
            are packed with OptHSampler to place and optimize hydrogen positions
            and NHQ flips, while residues with missing heavy atoms are rebuilt
            with DunbrackChiSampler.  When True, only missing heavy-atom
            sidechains are rebuilt with Dunbrack; hydrogens are left at the
            kinematically ideal positions produced during pose construction.
        prepare_ligands: If True, detect and prepare non-standard residues
            (see ``build_context_from_biotite`` for details).
        ligand_ph: Target pH for ligand protonation (default 7.4, only used when
            prepare_ligands=True).
        strict_atom_types: If True, unknown ligand atom types raise errors
            instead of using a fallback element heuristic.
        strict_ligands: If True (default), raise when a detected ligand cannot
            be prepared and registered, instead of silently dropping it. Pass
            False to warn-and-skip. Only used when prepare_ligands=True.
        ligand_params_files: Optional list of tmol YAML params file paths.
        sample_proton_chi: If True, prepared ligands emit PROTON_CHI
            ``chi_samples`` so OptHSampler samples ligand polar-H rotamers
            (enabled by default; pass False to disable). Only used when
            prepare_ligands=True.
        return_context: If True, return ``(pose_stack, PoseBuildContext)``.
        context: Reusable context from ``build_context_from_biotite``. It must
            be on ``torch_device`` and is mutually exclusive with ``param_db``
            and ``prepare_ligands=True``.
        atom37_coords: Optional coordinates shaped ``[pose, token, 37, xyz]``.
            When supplied, mapped finite coordinates are read from this tensor
            using the input structure's integer ``token_id`` and
            ``atom37_slot`` annotations. Unmapped atoms and non-finite entries
            retain the Biotite coordinates, allowing TMol to build absent leaf
            atoms normally. The resulting pose coordinates remain connected to
            this tensor for autograd. Geometry-based missing-density and
            additional-disulfide detection are disabled so topology is fixed.
        **kwargs: Additional arguments passed to pose_stack_from_canonical_form.

    Returns:
        PoseStack when no optional values requested and return_context is False.
        ``(PoseStack, PoseBuildContext)`` when return_context is True.
        ``(PoseStack, dict)`` when optional return values were requested via kwargs.
        Fragmented poses expose their block mapping as
        ``pose_stack.split_block_mapping``.
    """
    torch_device = resolve_device(torch_device)

    if context is not None:
        if param_db is not None:
            raise ValueError(
                "Pass either context= or param_db=, not both; the context "
                "already carries its parameter database."
            )
        if prepare_ligands:
            raise ValueError(
                "context= already contains prepared ligands; do not also pass "
                "prepare_ligands=True."
            )
        context_device = context.packed_block_types.device
        if context_device.type != torch_device.type or (
            context_device.type == "cuda" and context_device.index != torch_device.index
        ):
            raise ValueError(
                "context was built for device "
                f"'{context.packed_block_types.device}' but torch_device is "
                f"'{torch_device}'; they must match."
            )
    else:
        context = build_context_from_biotite(
            biotite_structure,
            torch_device,
            param_db=param_db,
            prepare_ligands=prepare_ligands,
            ligand_ph=ligand_ph,
            strict_atom_types=strict_atom_types,
            strict_ligands=strict_ligands,
            ligand_params_files=ligand_params_files,
            sample_proton_chi=sample_proton_chi,
        )

    fragment_mapping = None
    if context.fragment_definitions:
        from tmol.ligand import expand_fragmented_ligands

        biotite_structure, fragment_mapping = expand_fragmented_ligands(
            biotite_structure, context.fragment_definitions
        )

    # The canonical form is per-structure, so it is always computed here for the
    # given structure (never carried in the reusable context).
    cf = canonical_form_from_biotite(
        biotite_structure,
        torch_device,
        co=context.canonical_ordering,
        missing_density_distance_threshold=missing_density_distance_threshold,
        atom37_coords=atom37_coords,
    )

    return _pose_stack_from_canonical_and_context(
        cf,
        context,
        no_optH=no_optH,
        atom37_coords=atom37_coords,
        fragment_mapping=fragment_mapping,
        return_context=return_context,
        **kwargs,
    )


def _pose_stack_from_canonical_and_context(
    cf: CanonicalForm,
    context: PoseBuildContext,
    *,
    no_optH: bool,
    atom37_coords: torch.Tensor | None,
    fragment_mapping=None,
    return_context: bool = False,
    **kwargs: object,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Finish pose construction from a canonical form and reusable context."""
    from tmol.io import pose_stack_from_canonical_form
    from tmol.pack import build_missing_sidechains

    if atom37_coords is not None:
        kwargs.setdefault("find_additional_disulfides", False)

    caller_requested_atom_mapping = bool(kwargs.get("return_atom_mapping", False))
    if atom37_coords is not None:
        # Coordinate rebuilding / hydrogen optimization may return a detached
        # coordinate tensor. Keep the canonical-to-pose mapping so the finite
        # input coordinates can be restored afterward without name matching.
        kwargs["return_atom_mapping"] = True

    result = pose_stack_from_canonical_form(
        context.canonical_ordering,
        context.packed_block_types,
        *cf,
        return_block_has_missing_atoms=True,
        **kwargs,
    )

    pose_stack, opt_return_vals = result
    if fragment_mapping is not None:
        from tmol.ligand import apply_fragment_connections

        pose_stack = apply_fragment_connections(pose_stack, fragment_mapping)
        fragment_mapping = pose_stack.split_block_mapping
    block_has_missing_atoms = opt_return_vals["block_has_missing_atoms"]

    has_missing_atoms = block_has_missing_atoms is not None and bool(
        torch.any(block_has_missing_atoms)
    )
    if has_missing_atoms:
        _assert_no_ligand_with_missing_atoms(pose_stack, block_has_missing_atoms)

    needs_packing = block_has_missing_atoms is not None and (
        has_missing_atoms or not no_optH
    )
    if needs_packing:
        sfxn = (
            context._packing_score_function
            if has_missing_atoms
            else context._opth_score_function
        )
        dunbrack_sampler = context._dunbrack_sampler
        na_sampler = context._na_sampler if has_missing_atoms else None

        if has_missing_atoms:
            logger.info(
                "%i blocks with missing heavy atoms",
                torch.count_nonzero(block_has_missing_atoms),
            )
        pose_stack = build_missing_sidechains(
            pose_stack,
            sfxn,
            dunbrack_sampler,
            block_has_missing_atoms,
            no_optH=no_optH,
            na_sampler=na_sampler,
        )

    if atom37_coords is not None and needs_packing:
        pose_stack = _restore_canonical_input_coords(
            pose_stack,
            cf.coords,
            opt_return_vals["can_atom_mapping"],
            opt_return_vals["ps_atom_mapping"],
        )
    if atom37_coords is not None and not caller_requested_atom_mapping:
        del opt_return_vals["can_atom_mapping"]
        del opt_return_vals["ps_atom_mapping"]

    if fragment_mapping is not None:
        pose_stack.split_block_mapping = fragment_mapping
    _assert_no_nan_coords(pose_stack)

    # This code tries to faithfully return what the caller expects based on the optional
    # return values that they requested. Since we override the return_block_has_missing_atoms
    # bool to True, we cannot just count on the existence or absence of optional returned vals
    return_block_has_missing_atoms = (
        kwargs.get("return_block_has_missing_atoms")
        if ("return_block_has_missing_atoms" in kwargs)
        else False
    )
    if return_context:
        return pose_stack, context
    if len(opt_return_vals) > (0 if return_block_has_missing_atoms else 1):
        return pose_stack, opt_return_vals
    return pose_stack


def _restore_canonical_input_coords(
    pose_stack: PoseStack,
    canonical_coords: torch.Tensor,
    canonical_atom_mapping: torch.Tensor,
    pose_atom_mapping: torch.Tensor,
    mappings_are_finite: bool = False,
) -> PoseStack:
    """Restore finite canonical inputs after coordinate rebuilding or packing.

    TMol's packing pipeline deliberately treats coordinates as values rather
    than as an autograd graph. Atom37 callers need the reverse behavior: keep
    rebuilt/optimized coordinates for missing atoms and hydrogens, but route
    every finite input atom back to the differentiable canonical tensor. The
    atom mapping returned by pose construction makes this a pair of indexed
    tensor operations and avoids matching residue or atom names.
    """
    canonical_atom_mapping = canonical_atom_mapping.to(torch.int64)
    pose_atom_mapping = pose_atom_mapping.to(torch.int64)
    source_coords = canonical_coords[
        canonical_atom_mapping[:, 0],
        canonical_atom_mapping[:, 1],
        canonical_atom_mapping[:, 2],
    ]
    coords = pose_stack.coords.clone()
    if mappings_are_finite:
        coords[pose_atom_mapping[:, 0], pose_atom_mapping[:, 1]] = source_coords
    else:
        finite = torch.isfinite(source_coords).all(dim=-1)
        coords[
            pose_atom_mapping[finite, 0],
            pose_atom_mapping[finite, 1],
        ] = source_coords[finite]
    result = copy.copy(pose_stack)
    result.coords = coords
    return result


def _assert_no_ligand_with_missing_atoms(
    pose_stack: PoseStack, block_has_missing_atoms: "torch.Tensor"
) -> None:
    """Raise RuntimeError if a non-polymer block is flagged with missing atoms.

    The sidechain-rebuild pipeline (DunbrackChiSampler + FixedAAChiSampler)
    only handles polymer residues; if a ligand reaches it with missing heavy
    atoms the sampler silently produces no rotamer and the block's coords
    stay NaN.  Catch that here with a clear, actionable error.
    """
    pbt = pose_stack.packed_block_types
    block_type_ind = pose_stack.block_type_ind
    block_coord_offset = pose_stack.block_coord_offset
    coords = pose_stack.coords
    pdb_info = getattr(pose_stack, "pdb_info", None)

    flagged = torch.nonzero(block_has_missing_atoms, as_tuple=False).cpu().tolist()
    bad: list[str] = []
    for pi, bi in flagged:
        bt_ind = int(block_type_ind[pi, bi].item())
        if bt_ind < 0:
            continue
        bt = pbt.active_block_types[bt_ind]
        if bt.properties.polymer.is_polymer:
            continue  # protein/nucleic — handled by sidechain rebuild

        n_ats = len(bt.atoms)
        atom_start = int(block_coord_offset[pi, bi].item())
        block_coords = coords[pi, atom_start : atom_start + n_ats]
        missing_mask = torch.isnan(block_coords).any(dim=-1)
        missing_names = [
            bt.atoms[ai].name
            for ai in torch.nonzero(missing_mask, as_tuple=False).flatten().tolist()
        ]

        label = ""
        if pdb_info is not None and pdb_info.residue_labels is not None:
            chain = pdb_info.chain_labels[pi, bi]
            resid = pdb_info.residue_labels[pi, bi]
            label = f" chain={chain} resid={resid}"
        bad.append(
            f"pose={pi} block={bi} bt={bt.name}{label} "
            f"missing_atoms={missing_names}"
        )

    if bad:
        raise RuntimeError(
            "Ligand (non-polymer) block(s) have missing heavy atoms; "
            "tmol's sidechain rebuild only supports polymer residues. "
            "Provide a complete ligand structure (or remove the ligand) "
            "before calling pose_stack_from_biotite:\n  " + "\n  ".join(bad)
        )


def _assert_no_nan_coords(
    pose_stack: PoseStack, real_atoms: torch.Tensor | None = None
) -> None:
    """Raise a descriptive error if any real atom in the PoseStack has NaN coords.

    Reports the offending pose, residue label/chain, block-type name, and atom
    name so failures in the auto-parsing pipeline (ligand prep, leaf-atom
    rebuild, sidechain build) can be traced to a specific residue.
    """
    coords = pose_stack.coords
    real = pose_stack.real_atoms if real_atoms is None else real_atoms
    nan_atom_mask = torch.isnan(coords).any(dim=-1) & real
    if not torch.any(nan_atom_mask):
        return

    pbt = pose_stack.packed_block_types
    block_coord_offset = pose_stack.block_coord_offset
    block_type_ind = pose_stack.block_type_ind
    pdb_info = getattr(pose_stack, "pdb_info", None)

    bad: list[str] = []
    nan_idxs = torch.nonzero(nan_atom_mask, as_tuple=False).cpu().tolist()
    for pi, at_idx in nan_idxs:
        valid_block_mask = block_type_ind[pi] >= 0
        valid_block_inds = torch.nonzero(valid_block_mask, as_tuple=False).flatten()
        offsets = block_coord_offset[pi, valid_block_inds]
        sel = torch.nonzero(offsets <= at_idx, as_tuple=False).flatten()
        if sel.numel() == 0:
            continue
        bi = int(valid_block_inds[sel[-1]].item())
        offset_in_block = at_idx - int(block_coord_offset[pi, bi].item())
        bt = pbt.active_block_types[int(block_type_ind[pi, bi].item())]
        atom_name = (
            bt.atoms[offset_in_block].name
            if 0 <= offset_in_block < len(bt.atoms)
            else f"#{offset_in_block}"
        )
        label = ""
        if pdb_info is not None and pdb_info.residue_labels is not None:
            chain = pdb_info.chain_labels[pi, bi]
            resid = pdb_info.residue_labels[pi, bi]
            label = f" chain={chain} resid={resid}"
        bad.append(
            f"pose={pi} block={bi} bt={bt.name}{label} atom={atom_name} "
            f"(global_atom_idx={at_idx})"
        )

    head = bad[:20]
    tail = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
    raise RuntimeError(
        "NaN coordinates produced by pose_stack_from_biotite:\n  "
        + "\n  ".join(head)
        + tail
    )


@validate_args
def biotite_from_pose_stack(
    pose_stack: PoseStack,
    co: CanonicalOrdering | None = None,
    merge_fragments: bool = True,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Convert PoseStack back to Biotite structure.

    Args:
        pose_stack: Pose stack to convert.
        co: Canonical ordering used for conversion. Provide the ordering that
            was used when ligands or custom residue types are present.
        merge_fragments: Restore fragmented ligands to their original residue
            identity. Set to False to keep fragment residues separate.

    Returns:
        Biotite AtomArray for single-pose or AtomArrayStack for multi-pose.
    """
    if co is None:
        co = canonical_ordering_for_biotite()
    cf = canonical_form_from_pose_stack(co, pose_stack)
    structure = biotite_from_canonical_form(cf, co=co)
    sbm = getattr(pose_stack, "split_block_mapping", None)
    if merge_fragments and sbm is not None and sbm.entries:
        from tmol.ligand import recombine_fragmented_ligands

        structure = recombine_fragmented_ligands(structure, pose_stack)
    return structure


def _map_atoms_to_canonical(co, atom_res_inds, res_names, atom_names):
    """Map Biotite atom names to canonical ordering indices.

    Returns (valid_atom_mask, valid_atom_inds, valid_res_inds).
    """

    atom_inds = []
    valid = []
    unmapped: dict[str, list[str]] = {}
    for i, (resname, atname) in enumerate(zip(res_names, atom_names)):
        mapping = co.restypes_atom_index_mapping.get(resname, {})
        idx = mapping.get(atname, -1)
        atom_inds.append(idx)
        valid.append(idx >= 0)
        if idx < 0:
            unmapped.setdefault(resname, []).append(atname)

    valid_atom_mask = numpy.array(valid)
    atom_inds_arr = numpy.array(atom_inds)
    return (
        valid_atom_mask,
        atom_inds_arr[valid_atom_mask],
        atom_res_inds[valid_atom_mask],
    )


def _res_names_for_structure(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
):
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        return biotite_structure[0].res_name
    return biotite_structure.res_name


def _filter_supported_atoms_and_connectivity(  # noqa: C901
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    co: CanonicalOrdering,
):
    biotite_residues = biotite.structure.get_residues(biotite_structure)[1]
    to_remove = {"HOH"}
    known_residue_names = set(co.restype_io_equiv_classes)
    for i_3lc in biotite_residues:
        if i_3lc in to_remove:
            continue
        if i_3lc not in known_residue_names:
            logger.warning("Unrecognized 3lc %s", i_3lc)
            to_remove.add(i_3lc)

    res_names = _res_names_for_structure(biotite_structure)
    biotite_residue_starts = biotite.structure.get_residue_starts(biotite_structure)
    valid_res = numpy.array([name not in to_remove for name in res_names])[
        biotite_residue_starts
    ]

    # Filter residues missing mainchain atoms required for rotamer building.
    # Only atoms present in every variant count as required, so an atom a terminus
    # patch removes (the DNA 5' phosphate) does not disqualify the residue.
    # Residues with no mainchain definition (non-polymer) are skipped.
    atom_names = biotite_structure.atom_name
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        coords = biotite_structure.coord  # (n_poses, n_atoms, 3)
    else:
        coords = biotite_structure.coord[numpy.newaxis, :]  # (1, n_atoms, 3)
    residue_ends = numpy.append(
        biotite_residue_starts[1:], biotite_structure.array_length()
    )
    for i in range(len(valid_res)):
        if not valid_res[i]:
            continue
        start, end = biotite_residue_starts[i], residue_ends[i]
        res_name3 = biotite_structure.res_name[start]
        required = co.restypes_required_mainchain_atoms.get(res_name3)
        if not required:
            continue
        res_atom_names = atom_names[start:end]
        missing = set()
        for req_atom in required:
            matches = numpy.where(res_atom_names == req_atom)[0]
            if len(matches) == 0 or numpy.isnan(coords[:, start + matches[0], :]).any():
                missing.add(req_atom)
        if missing:
            logger.warning(
                "Residue %s %s %d is missing mainchain atoms %s; skipping",
                biotite_structure.chain_id[start],
                res_name3,
                biotite_structure.res_id[start],
                sorted(missing),
            )
            valid_res[i] = False

    valid_atoms = valid_res[get_all_residue_positions(biotite_structure)]

    # A kept residue whose neighbor was dropped has an unknown connection on
    # that side; the ends of the kept set are termini, so they are marked after
    # filtering
    lower = numpy.roll(valid_res, 1)[valid_res]
    upper = numpy.roll(valid_res, -1)[valid_res]
    if lower.size:
        lower[0] = True
        upper[-1] = True
    not_connected = numpy.invert(numpy.column_stack((lower, upper)))

    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        biotite_structure = biotite_structure[:, valid_atoms]
    else:
        biotite_structure = biotite_structure[valid_atoms]

    return biotite_structure, not_connected


def _break_connections_for_missing_density(
    not_connected: numpy.ndarray,
    biotite_chain_id_for_res: numpy.ndarray,
    tmol_coords: torch.Tensor,
    threshold: float,
) -> None:
    """Break inter-residue connections where upper/lower atoms are too far apart.

    Modifies ``not_connected`` in-place. For each pair of adjacent residues
    (i, i+1) that are currently marked as connected and belong to the same
    chain, the minimum distance between any atom in residue i and any atom in
    residue i+1 is compared across all poses. If that minimum distance exceeds
    ``threshold`` (in Angstroms), the connection is broken by setting
    not_connected[i, 1] = True and not_connected[i+1, 0] = True.

    Args:
        not_connected: Shape (n_res, 2) boolean array. True = no connection
            (terminus or explicitly broken); False = connected.
        biotite_chain_id_for_res: Shape (n_res,) integer chain IDs.
        tmol_coords: Shape (n_poses, n_res, max_atoms, 3) coordinate tensor.
        threshold: Distance threshold in Angstroms. Connections where the
            closest inter-residue atom pair exceeds this distance are broken.
    """
    n_res = not_connected.shape[0]
    coords_np = tmol_coords.cpu().numpy()

    for i in range(n_res - 1):
        # Skip already-disconnected pairs
        if not_connected[i, 1] or not_connected[i + 1, 0]:
            continue
        # Skip cross-chain pairs (handled separately by chain-break logic)
        if biotite_chain_id_for_res[i] != biotite_chain_id_for_res[i + 1]:
            continue

        # Compute minimum inter-residue distance across all poses.
        # A connection is kept if *any* pose shows atoms within threshold.
        min_dist = numpy.inf
        for p in range(coords_np.shape[0]):
            c_i = coords_np[p, i]  # (max_atoms, 3)
            c_j = coords_np[p, i + 1]

            valid_i = ~numpy.isnan(c_i[:, 0])
            valid_j = ~numpy.isnan(c_j[:, 0])
            if not valid_i.any() or not valid_j.any():
                continue

            ci_v = c_i[valid_i]
            cj_v = c_j[valid_j]
            diffs = ci_v[:, numpy.newaxis, :] - cj_v[numpy.newaxis, :, :]
            pose_min = numpy.sqrt((diffs**2).sum(axis=-1)).min()
            if pose_min < min_dist:
                min_dist = pose_min
            if min_dist <= threshold:
                break  # already within range; no need to check more poses

        if min_dist > threshold:
            logger.debug(
                "Breaking connection between residues %d and %d "
                "(closest atom distance %.3f Å > threshold %.3f Å)",
                i,
                i + 1,
                min_dist,
                threshold,
            )
            not_connected[i, 1] = True
            not_connected[i + 1, 0] = True


def _extract_residue_metadata(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    not_connected,
    torch_device: torch.device,
):
    biotite_residue_starts = biotite.structure.get_residue_starts(biotite_structure)

    chain_starts = biotite.structure.get_chain_starts(biotite_structure)
    n_atoms = biotite_structure.array_length()
    per_atom_chain_idx = numpy.zeros(n_atoms, dtype=int)
    for i, start in enumerate(chain_starts):
        per_atom_chain_idx[start:] = i
    biotite_chain_id_for_res = per_atom_chain_idx[biotite_residue_starts]

    if len(biotite_chain_id_for_res) > 1:
        res_is_disconnected_from_neighbor = (
            biotite_chain_id_for_res[1:] != biotite_chain_id_for_res[:-1]
        )
        not_connected[1:, 0] &= ~res_is_disconnected_from_neighbor
        not_connected[:-1, 1] &= ~res_is_disconnected_from_neighbor

    res_not_connected_1 = torch.tensor(
        not_connected, dtype=torch.bool, device=torch_device
    ).unsqueeze(0)
    biotite_chain_labels = biotite_structure.chain_id[biotite_residue_starts]
    biotite_insertion_codes = biotite_structure.ins_code[biotite_residue_starts]
    biotite_residue_labels, biotite_residues = biotite.structure.get_residues(
        biotite_structure
    )
    return (
        biotite_chain_id_for_res,
        biotite_chain_labels,
        biotite_insertion_codes,
        biotite_residue_labels,
        biotite_residues,
        res_not_connected_1,
        not_connected,
    )


def _populate_canonical_coords(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering,
    biotite_residues,
    valid_atom_mask,
    valid_res_inds,
    valid_atom_inds,
):
    n_poses = 1
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        n_poses = biotite_structure.coord.shape[0]

    tmol_coords = torch.full(
        (n_poses, len(biotite_residues), co.max_n_canonical_atoms, 3),
        numpy.nan,
        dtype=torch.float32,
        device=torch_device,
    )
    biotite_coords = torch.tensor(biotite_structure.coord, device=torch_device)

    if n_poses == 1:
        tmol_coords[0, valid_res_inds, valid_atom_inds] = biotite_coords[
            valid_atom_mask
        ]
    else:
        for pose_ind in range(n_poses):
            tmol_coords[pose_ind, valid_res_inds, valid_atom_inds] = biotite_coords[
                pose_ind
            ][valid_atom_mask]
    return tmol_coords, n_poses


def _populate_optional_atom_metadata(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    n_poses: int,
    n_residues: int,
    max_n_canonical_atoms: int,
    valid_res_inds,
    valid_atom_inds,
    valid_atom_mask,
):
    biotite_b_factors = None
    biotite_occupancy = None

    if hasattr(biotite_structure, "b_factor"):
        b_factor = numpy.asarray(biotite_structure.b_factor)
        biotite_b_factors = numpy.full(
            (n_poses, n_residues, max_n_canonical_atoms),
            DEFAULT_ATOM_B_FACTOR,
            dtype=numpy.float32,
        )
        if n_poses == 1 or b_factor.ndim == 1:
            biotite_b_factors[:, valid_res_inds, valid_atom_inds] = b_factor[
                valid_atom_mask
            ]
        else:
            for pose_ind in range(n_poses):
                biotite_b_factors[pose_ind, valid_res_inds, valid_atom_inds] = b_factor[
                    pose_ind
                ][valid_atom_mask]

    if hasattr(biotite_structure, "occupancy"):
        occupancy = numpy.asarray(biotite_structure.occupancy)
        biotite_occupancy = numpy.full(
            (n_poses, n_residues, max_n_canonical_atoms),
            DEFAULT_ATOM_OCCUPANCY,
            dtype=numpy.float32,
        )
        if n_poses == 1 or occupancy.ndim == 1:
            biotite_occupancy[:, valid_res_inds, valid_atom_inds] = occupancy[
                valid_atom_mask
            ]
        else:
            for pose_ind in range(n_poses):
                biotite_occupancy[pose_ind, valid_res_inds, valid_atom_inds] = (
                    occupancy[pose_ind][valid_atom_mask]
                )
    return biotite_b_factors, biotite_occupancy


def _validate_atom37_coords(
    atom37_coords: torch.Tensor, torch_device: torch.device
) -> None:
    """Validate the tensor contract shared by direct and prepared adapters."""
    if atom37_coords.ndim != 4 or atom37_coords.shape[-2:] != (37, 3):
        raise ValueError(
            "atom37_coords must have shape [n_poses, n_tokens, 37, 3]; "
            f"got {tuple(atom37_coords.shape)}"
        )
    if atom37_coords.dtype != torch.float32:
        raise TypeError(
            "atom37_coords must have dtype torch.float32; " f"got {atom37_coords.dtype}"
        )
    if atom37_coords.device != torch_device:
        raise ValueError(
            f"atom37_coords is on '{atom37_coords.device}' but torch_device is "
            f"'{torch_device}'; they must match"
        )


def _atom37_mapping(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    valid_atom_mask: numpy.ndarray,
    valid_res_inds: numpy.ndarray,
    valid_atom_inds: numpy.ndarray,
    max_n_tokens: int | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Validate and return source/target indices for Atom37 routing."""
    categories = set(biotite_structure.get_annotation_categories())
    missing = {"token_id", "atom37_slot"} - categories
    if missing:
        raise Atom37MappingError(
            "atom37 coordinate routing requires integer Biotite annotation(s): "
            + ", ".join(sorted(missing))
        )

    token_id = numpy.asarray(biotite_structure.token_id)
    slot = numpy.asarray(biotite_structure.atom37_slot)
    if not numpy.issubdtype(token_id.dtype, numpy.integer):
        raise Atom37MappingError(
            "Biotite token_id annotation must have an integer dtype"
        )
    if not numpy.issubdtype(slot.dtype, numpy.integer):
        raise Atom37MappingError(
            "Biotite atom37_slot annotation must have an integer dtype"
        )

    token_id = token_id.astype(numpy.int64, copy=False)[valid_atom_mask]
    slot = slot.astype(numpy.int64, copy=False)[valid_atom_mask]
    mapped = (token_id >= 0) & (slot >= 0)
    if numpy.any(slot[mapped] >= 37):
        maximum = int(slot[mapped].max())
        raise Atom37MappingError(
            f"atom37_slot values must be less than 37; got {maximum}"
        )
    if not numpy.any(mapped):
        raise Atom37MappingError("No supported Biotite atoms map to atom37_coords")
    if max_n_tokens is not None and numpy.any(token_id[mapped] >= max_n_tokens):
        maximum = int(token_id[mapped].max())
        raise Atom37MappingError(
            f"token_id {maximum} exceeds atom37_coords token count {max_n_tokens}"
        )

    source_pairs = numpy.column_stack((token_id[mapped], slot[mapped]))
    if numpy.unique(source_pairs, axis=0).shape[0] != source_pairs.shape[0]:
        raise Atom37MappingError(
            "Each mapped Biotite atom must use a unique (token_id, atom37_slot) pair"
        )
    return (
        token_id[mapped],
        slot[mapped],
        numpy.asarray(valid_res_inds)[mapped],
        numpy.asarray(valid_atom_inds)[mapped],
    )


def _populate_canonical_coords_from_atom37(
    atom37_coords: torch.Tensor,
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering,
    biotite_residues,
    valid_atom_mask: numpy.ndarray,
    valid_res_inds: numpy.ndarray,
    valid_atom_inds: numpy.ndarray,
) -> tuple[torch.Tensor, int]:
    """Overlay mapped atom37 coordinates on the Biotite canonical coordinates.

    Finite tensor values replace their matching Biotite atoms through one
    differentiable indexed assignment. Unmapped and non-finite tensor entries
    retain the reference coordinates, which is important for hydrogens and for
    atoms that TMol may need to rebuild.

    Returns:
        The canonical coordinate tensor and its pose count.
    """
    _validate_atom37_coords(atom37_coords, torch_device)
    token_id, slot, mapped_res_inds, mapped_atom_inds = _atom37_mapping(
        biotite_structure,
        valid_atom_mask,
        valid_res_inds,
        valid_atom_inds,
        atom37_coords.shape[1],
    )

    reference_coords, reference_n_poses = _populate_canonical_coords(
        biotite_structure,
        torch_device,
        co,
        biotite_residues,
        valid_atom_mask,
        valid_res_inds,
        valid_atom_inds,
    )
    n_poses = atom37_coords.shape[0]
    if reference_n_poses not in (1, n_poses):
        raise ValueError(
            f"Biotite structure has {reference_n_poses} poses but atom37_coords "
            f"has {n_poses}"
        )
    if reference_n_poses == 1 and n_poses != 1:
        reference_coords = reference_coords.expand(n_poses, -1, -1, -1).clone()

    mapped_token_id = torch.as_tensor(token_id, device=torch_device)
    mapped_slot = torch.as_tensor(slot, device=torch_device)
    source_coords = atom37_coords[:, mapped_token_id, mapped_slot]
    finite = torch.isfinite(source_coords).all(dim=-1)
    mapped_res_inds = torch.as_tensor(mapped_res_inds, device=torch_device)
    mapped_atom_inds = torch.as_tensor(mapped_atom_inds, device=torch_device)
    reference_coords[:, mapped_res_inds, mapped_atom_inds] = torch.where(
        finite.unsqueeze(-1),
        source_coords,
        reference_coords[:, mapped_res_inds, mapped_atom_inds],
    )
    return reference_coords, n_poses


@validate_args
def canonical_form_from_biotite(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering | None = None,
    missing_density_distance_threshold: float = 2.4,
    atom37_coords: torch.Tensor | None = None,
) -> CanonicalForm:
    """Convert a Biotite AtomArray or AtomArrayStack to a CanonicalForm.

    This function bridges between Biotite's data structures and tmol's internal
    representation by converting atom and residue information from string-based
    identifiers to tmol's canonical integer-based indexing system.

    Args:
        biotite_structure: A Biotite AtomArray (single structure) or
            AtomArrayStack (multiple structures) containing the molecular data.
            Must contain atom coordinates, residue names, atom names, chain IDs,
            and optionally B-factors and occupancy values.
        torch_device: PyTorch device (e.g., torch.device('cuda') or torch.device('cpu'))
            where the resulting tensors should be allocated.
        co: A CanonicalForm in case you want to use a non-default database (and thus may need
            a different mapping)
        missing_density_distance_threshold: Maximum distance in angstroms for
            treating a polymer gap as missing density rather than a chain break.
        atom37_coords: Optional autograd-tracked coordinate tensor
            of shape [n_poses, n_tokens, 37, 3]. When provided, coordinates are
            sourced from this tensor (routed by the ``token_id`` and ``atom37_slot``
            annotations on ``biotite_structure``) instead of the static biotite
            coordinates, and the geometry-based missing-density check is skipped so
            the topology stays fixed and gradients flow. See
            :func:`~tmol.io.pose_stack_from_atom37_and_biotite`.

    Returns:
        CanonicalForm: A data structure containing:
            - chain_id: Tensor mapping residues to chain indices
            - res_types: Tensor mapping residues to tmol residue type indices
            - coords: 4D tensor of atomic coordinates (poses x residues x atoms x 3)
            - res_labels: Original residue sequence numbers from the structure
            - residue_insertion_codes: PDB insertion codes for residues
            - chain_labels: Original chain identifiers from the structure
            - atom_occupancy: Optional tensor of atom occupancy values
            - atom_b_factor: Optional tensor of atom B-factor values
            - disulfides: None (not handled in this conversion)
            - res_not_connected: Tensor describing whether two consecutive residues
              should be treated as chemically bonded.

    """
    torch_device = resolve_device(torch_device)
    if co is None:
        co = canonical_ordering_for_biotite()

    biotite_structure, not_connected = _filter_supported_atoms_and_connectivity(
        biotite_structure, co
    )
    (
        biotite_chain_id_for_res,
        biotite_chain_labels,
        biotite_insertion_codes,
        biotite_residue_labels,
        biotite_residues,
        res_not_connected_1,
        not_connected,
    ) = _extract_residue_metadata(biotite_structure, not_connected, torch_device)

    atom_res_inds = get_all_residue_positions(biotite_structure)
    biotite_name_for_atom = biotite_structure.atom_name
    biotite_res_name_for_atom = biotite_structure.res_name

    restype_to_index = {name: i for i, name in enumerate(co.restype_io_equiv_classes)}
    tmol_restypes = [restype_to_index[i_3lc] for i_3lc in biotite_residues]

    valid_atom_mask, valid_atom_inds, valid_res_inds = _map_atoms_to_canonical(
        co,
        atom_res_inds,
        biotite_res_name_for_atom,
        biotite_name_for_atom,
    )
    if atom37_coords is None:
        tmol_coords, n_poses = _populate_canonical_coords(
            biotite_structure,
            torch_device,
            co,
            biotite_residues,
            valid_atom_mask,
            valid_res_inds,
            valid_atom_inds,
        )
    else:
        tmol_coords, n_poses = _populate_canonical_coords_from_atom37(
            atom37_coords,
            biotite_structure,
            torch_device,
            co,
            biotite_residues,
            valid_atom_mask,
            valid_res_inds,
            valid_atom_inds,
        )
    biotite_b_factors, biotite_occupancy = _populate_optional_atom_metadata(
        biotite_structure,
        n_poses,
        len(biotite_residues),
        co.max_n_canonical_atoms,
        valid_res_inds,
        valid_atom_inds,
        valid_atom_mask,
    )

    # Format metadata for the CanonicalForm
    def copy_for_all_poses(dat):
        return numpy.repeat(dat[numpy.newaxis, ...], n_poses, axis=0)

    biotite_residue_labels = copy_for_all_poses(biotite_residue_labels)
    biotite_chain_labels = copy_for_all_poses(biotite_chain_labels)
    biotite_insertion_codes = copy_for_all_poses(biotite_insertion_codes)

    chain_id = (
        torch.tensor(biotite_chain_id_for_res, dtype=torch.int32, device=torch_device)
        .unsqueeze(0)
        .repeat(n_poses, 1)
    )
    res_types = (
        torch.tensor(tmol_restypes, dtype=torch.int32, device=torch_device)
        .unsqueeze(0)
        .repeat(n_poses, 1)
    )
    # Geometry-based missing density detection: break connections where the
    # upper atom of residue i and lower atom of residue i+1 are too far apart.
    # Skipped for the differentiable atom37 path: topology there is derived from
    # chemical identity alone so it stays fixed across (possibly noisy) coordinate
    # updates, and this check would both read a grad tensor and detach it.
    if (
        missing_density_distance_threshold > 0
        and len(biotite_residues) > 1
        and atom37_coords is None
    ):
        _break_connections_for_missing_density(
            not_connected,
            biotite_chain_id_for_res,
            tmol_coords,
            missing_density_distance_threshold,
        )
        res_not_connected_1 = torch.tensor(
            not_connected, dtype=torch.bool, device=torch_device
        ).unsqueeze(0)

    res_not_connected = res_not_connected_1.repeat(n_poses, 1, 1)

    # Return CanonicalForm with all converted data
    return CanonicalForm(
        chain_id=chain_id,
        res_types=res_types,
        coords=tmol_coords,
        chain_labels=biotite_chain_labels.astype(object),
        res_labels=biotite_residue_labels,
        residue_insertion_codes=biotite_insertion_codes.astype(object),
        atom_occupancy=biotite_occupancy,
        atom_b_factor=biotite_b_factors,
        disulfides=None,
        res_not_connected=res_not_connected,
    )


@toolz.functoolz.memoize
def _paramdb_for_biotite() -> ParameterDatabase:
    """For Biotite, let's just get the default param DB.
    We shouldn't need a subset since we're mapping from strings(?)"""

    return ParameterDatabase.get_default()


@toolz.functoolz.memoize
def _restype_set_for_biotite() -> ResidueTypeSet:
    paramdb = _paramdb_for_biotite()
    return ResidueTypeSet.from_database(paramdb.chemical)


@validate_args
@toolz.functoolz.memoize
def canonical_ordering_for_biotite() -> CanonicalOrdering:
    """Construct the CanonicalOrdering object to use for Biotite.
    This wont be used as a typical CanonicalOrdering object, since
    we aren't mapping from int-to-int, and instead are going from
    string-to-int.
    """

    paramdb = _paramdb_for_biotite()
    return CanonicalOrdering.from_chemdb(paramdb.chemical)


@validate_args
@toolz.functoolz.memoize
def packed_block_types_for_biotite(device: torch.device) -> PackedBlockTypes:
    """Construct the PackedBlockTypes (PBT) object that will used for Biotite.
    We'll use the defaults since anything might show up in a Biotite AtomArray.
    Some things may show up in the AtomArrays that are not handled by this
    PBT, but that is work for the future.
    """

    restype_set = _restype_set_for_biotite()

    return PackedBlockTypes.from_restype_list(
        restype_set.chem_db, restype_set, restype_set.residue_types, device
    )


@validate_args
@toolz.functoolz.memoize
def _default_pose_build_context(device: torch.device) -> PoseBuildContext:
    """Return the process-wide construction context for the default database."""
    return PoseBuildContext(
        canonical_ordering=canonical_ordering_for_biotite(),
        packed_block_types=packed_block_types_for_biotite(device),
        parameter_database=_paramdb_for_biotite(),
        restype_set=_restype_set_for_biotite(),
    )


def _derived_types_for_param_db(
    param_db: ParameterDatabase, device: torch.device
) -> tuple[CanonicalOrdering, ResidueTypeSet, PackedBlockTypes]:
    """Build canonical ordering and packed block types from a DB."""
    co = CanonicalOrdering.from_chemdb(param_db.chemical)
    rts = ResidueTypeSet.from_database(param_db.chemical)
    pbt = PackedBlockTypes.from_restype_list(
        rts.chem_db, rts, rts.residue_types, device
    )
    return co, rts, pbt


@validate_args
def biotite_from_canonical_form(  # noqa: C901
    cf: CanonicalForm,
    co: CanonicalOrdering | None = None,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Convert canonical TMol tensors to a Biotite atom array.

    Args:
        cf: Canonical coordinates, residue identities, and metadata.
        co: Canonical atom ordering. Defaults to the Biotite ordering.

    Returns:
        One atom array, or an atom-array stack for multiple coordinate sets.

    Raises:
        ValueError: If poses in a multi-pose input have different metadata.
    """
    import biotite.structure as struc

    if co is None:
        co = canonical_ordering_for_biotite()

    n_poses = cf.coords.size(0)
    n_residues = cf.coords.size(1)
    max_atoms = cf.coords.size(2)

    if n_poses > 1 and not _poses_have_identical_metadata(cf):
        raise ValueError(
            "Cannot convert CanonicalForm with multiple poses to biotite structure: "
            "poses have different metadata (chain_id, res_types, res_labels, "
            "residue_insertion_codes, or chain_labels). "
            "Only coordinate differences are allowed for multi-pose conversion."
        )

    # For multi-pose (NMR) structures, all poses must have the same atom
    # annotations. Use the union of non-NaN atoms across all poses to build
    # a consistent atom list; missing atoms in individual poses get NaN coords.
    atom_mask = torch.any(~torch.isnan(cf.coords[:, :, :, 0]), dim=0)

    template_atoms = []
    atom_indices = []
    for res_id in range(n_residues):
        chain_label = cf.chain_labels[0, res_id]
        res_label = cf.res_labels[0, res_id]
        res_type_id = cf.res_types[0, res_id].cpu()

        res_name = co.restype_io_equiv_classes[res_type_id]
        atom_name_list = co.restypes_ordered_atom_names[res_name]

        for atom_id in range(min(max_atoms, len(atom_name_list))):
            if not atom_mask[res_id, atom_id]:
                continue

            atom_name = atom_name_list[atom_id]
            template_atoms.append(
                struc.Atom(
                    [0.0, 0.0, 0.0],
                    chain_id=chain_label,
                    res_id=res_label,
                    res_name=res_name,
                    atom_name=atom_name,
                    element=get_element_from_atom_name(atom_name),
                    b_factor=(
                        cf.atom_b_factor[0, res_id, atom_id]
                        if cf.atom_b_factor is not None
                        else None
                    ),
                    occupancy=(
                        cf.atom_occupancy[0, res_id, atom_id]
                        if cf.atom_occupancy is not None
                        else None
                    ),
                )
            )
            atom_indices.append((res_id, atom_id))

    template = struc.array(template_atoms)

    if n_poses == 1:
        for i, (res_id, atom_id) in enumerate(atom_indices):
            template.coord[i] = cf.coords[0, res_id, atom_id].cpu().numpy()
        return template

    poses = []
    for pose_id in range(n_poses):
        arr = template.copy()
        for i, (res_id, atom_id) in enumerate(atom_indices):
            c = cf.coords[pose_id, res_id, atom_id].cpu()
            if torch.isnan(c).any():
                arr.coord[i] = [float("nan")] * 3
            else:
                arr.coord[i] = c.numpy()
        poses.append(arr)
    return struc.stack(poses)


@validate_args
def _poses_have_identical_metadata(cf: CanonicalForm) -> bool:
    """Check if all poses in the CanonicalForm have identical metadata.

    Returns True if all poses have the same:
    - chain_id
    - res_types
    - res_labels
    - residue_insertion_codes
    - chain_labels

    Only coordinates are allowed to differ between poses.
    """
    n_poses = cf.coords.size(0)

    if n_poses <= 1:
        return True

    if not torch.all(cf.chain_id[0] == cf.chain_id[1:]).item():
        return False

    if not torch.all(cf.res_types[0] == cf.res_types[1:]).item():
        return False

    for pose_id in range(1, n_poses):
        if not numpy.array_equal(cf.res_labels[0], cf.res_labels[pose_id]):
            return False

    for pose_id in range(1, n_poses):
        if not numpy.array_equal(
            cf.residue_insertion_codes[0], cf.residue_insertion_codes[pose_id]
        ):
            return False

    for pose_id in range(1, n_poses):
        if not numpy.array_equal(cf.chain_labels[0], cf.chain_labels[pose_id]):
            return False

    return True
