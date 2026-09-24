import copy
from collections import defaultdict
from typing import TYPE_CHECKING

import attr
import torch
import numpy
import toolz
import biotite
import biotite.structure
import logging
import warnings

from tmol.types import validate_args
from tmol.chemical import ResidueTypeSet
from tmol.chemical import BondType as ChemBondType
from tmol.database import ParameterDatabase
from tmol.database.chemical import metal_table
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

if TYPE_CHECKING:
    from tmol.ligand import FragmentedLigandPoseMapping

_MAX_PREPARED_BATCH_SIZES = 4


def _clone_pose_topology(pose_stack: PoseStack) -> PoseStack:
    """Clone caller-mutable pose data while retaining shared chemical types."""
    result = pose_stack.clone()
    result.pdb_info = copy.deepcopy(pose_stack.pdb_info)
    result.split_block_mapping = copy.deepcopy(pose_stack.split_block_mapping)
    return result


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
        block_leaf_atom_is_missing = torch.zeros(
            (
                pose_stack.n_poses,
                pose_stack.max_n_blocks,
                pose_stack.max_n_block_atoms,
            ),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        pose_atom_is_missing = torch.zeros(
            pose_stack.coords.shape[:2], dtype=torch.bool, device=pose_stack.device
        )
        pose_template = _clone_pose_topology(pose_stack)
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
        finite = torch.isfinite(source_coords).all(dim=-1)
        pose_ind = pose_mapping[:, 0]
        pose_atom = pose_mapping[:, 1]
        block_ind = mapping[:, 1]
        block_atom = (
            pose_atom - self.pose_stack.block_coord_offset64[pose_ind, block_ind]
        )
        block_leaf_atom_is_missing = self.block_leaf_atom_is_missing.clone()
        block_leaf_atom_is_missing[
            pose_ind[~finite], block_ind[~finite], block_atom[~finite]
        ] = True
        pose_atom_is_missing = self.pose_atom_is_missing.clone()
        pose_atom_is_missing[pose_ind[~finite], pose_atom[~finite]] = True
        coords = torch.zeros_like(self.pose_stack.coords)
        coords[pose_ind[finite], pose_atom[finite]] = source_coords[finite]

        pbt = self.pose_stack.packed_block_types
        from tmol.io.details._build_missing_leaf_atoms import (
            _apply_h_geometric_completion,
        )
        from tmol.io.details.compiled import gen_pose_leaf_atoms

        coords = gen_pose_leaf_atoms(
            coords,
            block_leaf_atom_is_missing,
            pose_atom_is_missing,
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
            block_leaf_atom_is_missing,
            self.pose_stack.block_coord_offset,
            self.pose_stack.block_type_ind,
            self.pose_stack.inter_residue_connections,
        )
        pose_stack = _clone_pose_topology(self.pose_stack)
        pose_stack.coords = coords
        return pose_stack

    def has_missing_nonleaf(self, canonical_coords: torch.Tensor) -> bool:
        """Return whether replay would require sidechain packing."""
        mapping = self.canonical_atom_mapping
        pose_mapping = self.pose_atom_mapping
        source_coords = canonical_coords[mapping[:, 0], mapping[:, 1], mapping[:, 2]]
        missing = ~torch.isfinite(source_coords).all(dim=-1)
        if not bool(torch.any(missing)):
            return False

        pose_ind = pose_mapping[:, 0]
        pose_atom = pose_mapping[:, 1]
        block_ind = mapping[:, 1]
        block_atom = (
            pose_atom - self.pose_stack.block_coord_offset64[pose_ind, block_ind]
        )
        block_type = self.pose_stack.block_type_ind64[pose_ind, block_ind]
        is_leaf = self.pose_stack.packed_block_types.is_leaf_atom[
            block_type, block_atom
        ]
        return bool(torch.any(missing & ~is_leaf))


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PreparedAtom37PoseBuilder:
    """Bind immutable Biotite topology for repeated Atom37 pose construction.

    Calls accept float32 coordinates shaped ``[n_poses, n_tokens, 37, 3]`` on
    the context's device. The first call for a batch size prepares its fixed
    pose topology; up to four recently used batch sizes are cached. Inputs with
    coordinate-dependent atom presence or ambiguous histidine hydrogens use the
    normal uncached construction path.

    The builder owns a mutable topology cache and is not safe for concurrent
    calls. Use one builder per calling thread when pose construction overlaps.
    """

    context: PoseBuildContext
    canonical_template: CanonicalForm
    mapped_token_id: torch.Tensor
    mapped_slot: torch.Tensor
    mapped_residue: torch.Tensor
    mapped_atom: torch.Tensor
    max_token_id: int
    required_mainchain_entries: tuple[tuple[int, int, str], ...]
    fragment_mapping: "FragmentedLigandPoseMapping | None" = None
    _topology_cache_safe: bool = True
    _pose_topologies: dict[int, _PreparedAtom37PoseTopology] = attr.ib(
        factory=dict, eq=False, repr=False
    )

    def __call__(
        self,
        atom37_coords: torch.Tensor,
        *,
        opt_h: bool = False,
    ) -> PoseStack:
        """Build a differentiable pose batch.

        Args:
            atom37_coords: Float32 coordinates shaped
                ``[n_poses, n_tokens, 37, 3]`` on the context's device.
            opt_h: Optimize hydrogen positions after construction. Disabled by
                default so finite input coordinates are preserved.

        Returns:
            A pose whose coordinates remain differentiable with respect to
            ``atom37_coords``.
        """
        canonical_coords = self._canonical_coords(atom37_coords)
        if not self._topology_cache_safe:
            return pose_stack_from_canonical_form_and_context(
                self._canonical_form(canonical_coords),
                self.context,
                no_optH=not opt_h,
                atom37_coords=atom37_coords,
                fragment_mapping=self.fragment_mapping,
            )
        n_poses = atom37_coords.shape[0]
        topology = self._pose_topologies.pop(n_poses, None)
        topology_was_cached = topology is not None
        if topology is None:
            cf = self._canonical_form(canonical_coords)
            pose_stack, details = pose_stack_from_canonical_form_and_context(
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
                return pose_stack_from_canonical_form_and_context(
                    cf,
                    self.context,
                    no_optH=False,
                    atom37_coords=atom37_coords,
                    fragment_mapping=self.fragment_mapping,
                )
            topology = _PreparedAtom37PoseTopology.from_pose(
                pose_stack,
                canonical_coords,
                details["can_atom_mapping"],
                details["ps_atom_mapping"],
                block_has_missing_atoms,
            )
        else:
            if topology.has_missing_nonleaf(canonical_coords):
                self._pose_topologies[n_poses] = topology
                return pose_stack_from_canonical_form_and_context(
                    self._canonical_form(canonical_coords),
                    self.context,
                    no_optH=not opt_h,
                    atom37_coords=atom37_coords,
                    fragment_mapping=self.fragment_mapping,
                )
            pose_stack = topology.pose_from_canonical(canonical_coords)
        if len(self._pose_topologies) >= _MAX_PREPARED_BATCH_SIZES:
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
                has_missing_atoms=False,
            )
            pose_stack = _restore_canonical_input_coords(
                pose_stack,
                canonical_coords,
                topology.canonical_atom_mapping,
                topology.pose_atom_mapping,
            )
        # Pose construction validates the initial topology. Check the initial
        # packed result too, but do not force a CUDA-to-host synchronization on
        # every replay of an already validated fixed topology.
        if opt_h and not topology_was_cached:
            _assert_no_nan_coords(pose_stack, topology.real_atoms)
        return pose_stack

    def _canonical_coords(self, atom37_coords: torch.Tensor) -> torch.Tensor:
        """Overlay one coordinate batch without expanding static topology data."""
        device = self.context.packed_block_types.device
        _validate_atom37_coords(atom37_coords, device)
        if self.max_token_id >= atom37_coords.shape[1]:
            raise Atom37MappingError(
                f"token_id {self.max_token_id} exceeds atom37_coords token count "
                f"{atom37_coords.shape[1]}"
            )

        n_poses = atom37_coords.shape[0]
        template = self.canonical_template
        template_n_poses = template.coords.shape[0]
        if template_n_poses not in (1, n_poses):
            raise ValueError(
                f"Biotite structure has {template_n_poses} poses but "
                f"atom37_coords has {n_poses}"
            )

        coords = template.coords
        coords = (
            coords.clone()
            if template_n_poses == n_poses
            else coords.expand(n_poses, *coords.shape[1:]).clone()
        )
        source_coords = atom37_coords[:, self.mapped_token_id, self.mapped_slot]
        _validate_mapped_atom37_triplets(
            source_coords, self.mapped_token_id, self.mapped_slot
        )
        coords[:, self.mapped_residue, self.mapped_atom] = source_coords
        _validate_effective_mainchain_coords(coords, self.required_mainchain_entries)
        return coords

    def _canonical_form(self, coords: torch.Tensor) -> CanonicalForm:
        """Expand the static canonical topology for an uncached batch size."""
        template = self.canonical_template
        n_poses = coords.shape[0]

        def tensor_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return value.expand(n_poses, *value.shape[1:]).clone()

        def array_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return numpy.repeat(value, n_poses, axis=0)

        def bonds_for_poses(value):
            if value is None or template.coords.shape[0] == n_poses:
                return value
            bonds = value.repeat(n_poses, 1)
            bonds[:, 0] = torch.arange(n_poses, device=coords.device).repeat_interleave(
                value.shape[0]
            )
            return bonds

        return CanonicalForm(
            chain_id=tensor_for_poses(template.chain_id),
            res_types=tensor_for_poses(template.res_types),
            coords=coords,
            res_labels=array_for_poses(template.res_labels),
            residue_insertion_codes=array_for_poses(template.residue_insertion_codes),
            chain_labels=array_for_poses(template.chain_labels),
            atom_occupancy=array_for_poses(template.atom_occupancy),
            atom_b_factor=array_for_poses(template.atom_b_factor),
            disulfides=bonds_for_poses(template.disulfides),
            res_not_connected=tensor_for_poses(template.res_not_connected),
            cyclic_bonds=bonds_for_poses(template.cyclic_bonds),
            covalent_bonds=bonds_for_poses(template.covalent_bonds),
            metal_coordination=bonds_for_poses(template.metal_coordination),
        )


@validate_args
def prepare_atom37_pose_builder(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    context: PoseBuildContext,
) -> PreparedAtom37PoseBuilder:
    """Prepare a callable for repeatedly binding Atom37 coordinates to topology.

    This is the campaign-oriented counterpart to
    :func:`pose_stack_from_atom37_and_topology`: immutable residue identity,
    connectivity, fragmentation, and Atom37 routing are resolved once. Calling
    the returned builder with a coordinate tensor constructs a differentiable
    pose while retaining TMol's usual missing-atom behavior. The returned
    builder preserves finite input hydrogens by default; pass ``opt_h=True`` to
    optimize them after construction.
    """
    device = context.packed_block_types.device
    fragment_mapping = None
    if context.fragment_definitions:
        from tmol.ligand import expand_fragmented_ligands

        biotite_structure, fragment_mapping = expand_fragmented_ligands(
            biotite_structure, context.fragment_definitions
        )

    biotite_structure = _normalize_input_identifiers(
        biotite_structure, context.canonical_ordering.name3_aliases
    )
    filtered, _ = _filter_supported_atoms_and_connectivity(
        biotite_structure,
        context.canonical_ordering,
        filter_missing_mainchain=False,
    )
    canonical_template = canonical_form_from_biotite(
        filtered,
        device,
        co=context.canonical_ordering,
        missing_density_distance_threshold=0.0,
        _filter_missing_mainchain=False,
    )
    atom_residue = get_all_residue_positions(filtered)
    valid_mask, valid_atom, valid_residue = _map_atoms_to_canonical(
        context.canonical_ordering,
        atom_residue,
        filtered.res_name,
        filtered.atom_name,
        filtered.element,
    )
    token_id, slot, mapped_residue, mapped_atom = _atom37_mapping(
        filtered, valid_mask, valid_residue, valid_atom
    )
    mapped_token_id = torch.as_tensor(token_id, device=device)
    mapped_slot = torch.as_tensor(slot, device=device)
    mapped_residue = torch.as_tensor(mapped_residue, device=device)
    mapped_atom = torch.as_tensor(mapped_atom, device=device)
    his_inds = context.canonical_ordering.his_inds
    ambiguous_his = False
    if his_inds.his_co_aa_inds:
        ambiguous_atom_inds = torch.tensor(
            [his_inds.his_HN_in_co, his_inds.his_NH_in_co, his_inds.his_NN_in_co],
            device=device,
        )
        is_his = torch.isin(
            canonical_template.res_types,
            torch.tensor(his_inds.his_co_aa_inds, device=device),
        )
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
        required_mainchain_entries=_required_mainchain_entries(
            filtered, context.canonical_ordering
        ),
        fragment_mapping=fragment_mapping,
        topology_cache_safe=not ambiguous_his,
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
    chem_comp_types: dict | None = None,
    use_ccd: bool = True,
    ligand_seed: int | None = None,
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
        chem_comp_types: ``{comp_id: type}`` from the input file's
            ``_chem_comp`` table (see
            ``tmol.ligand.chem_comp_types_from_cif``), which says whether a
            residue belongs to a polymer where the file does not number it
            along a sequence. Only used when prepare_ligands=True.
        use_ccd: Whether a residue the input declares no chemistry for may be
            completed from the component dictionary by its code. Pass False for
            a source that supplies whole molecules under codes of its own, such
            as a mol2. Only used when prepare_ligands=True.
        ligand_seed: Fixed RNG seed for the conformer each prepared residue
            is built from, making preparation reproducible. Only used when
            prepare_ligands=True.

    Returns:
        PoseBuildContext containing canonical ordering, packed block
        types, parameter database, and residue type set.
    """
    torch_device = resolve_device(torch_device)
    # aliased names are resolved before ligand detection, or the residue the
    #    alias points at would be prepared as a nonstandard one
    if ligand_params_files and not prepare_ligands:
        from tmol.ligand._params_file import inject_params_files

        param_db = inject_params_files(
            param_db or ParameterDatabase.get_default(),
            ligand_params_files,
            strict_atom_types=strict_atom_types,
        )
    chemdb = (param_db or ParameterDatabase.get_default()).chemical
    biotite_structure = _normalize_input_identifiers(
        biotite_structure,
        {alias.name3: alias.read_as for alias in chemdb.name3_aliases},
    )
    biotite_structure = _without_metal_coordination_bonds(biotite_structure)
    if prepare_ligands:
        from tmol.ligand import prepare_ligands as _prepare_ligands

        using_default_database = param_db is None
        if param_db is None:
            param_db = ParameterDatabase.get_default()

        # Take the names preparation cut a covalent bond to. A ligand skipped
        # under strict_ligands=False loses its bonds there, and building the
        # pose from the original graph would reject the dangling partner the
        # caller was told would simply be dropped.
        param_db, co, fragment_definitions, cut_partners = _prepare_ligands(
            biotite_structure,
            param_db=param_db,
            ph=ligand_ph,
            strict_atom_types=strict_atom_types,
            params_files=ligand_params_files,
            strict_ligands=strict_ligands,
            return_fragment_definitions=True,
            return_cut_partners=True,
            chem_comp_types=chem_comp_types,
            use_ccd=use_ccd,
            seed=ligand_seed,
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
            cut_covalent_partners=cut_partners,
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
    no_optH: bool = True,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    ligand_params_files: list[str] | None = None,
    chem_comp_types: dict | None = None,
    use_ccd: bool = True,
    ligand_seed: int | None = None,
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

    Missing non-polymer atoms use prepared conformer geometry and resolved
    coordinate anchors. If only an attachment's endpoints are resolved, its
    first declared torsion sample and the next resolved partner atom can orient
    the missing component. These are starting conformers for scoring/packing,
    not recovered experimental coordinates. Supplied heavy-atom coordinates
    stay unchanged; insufficient or degenerate references still raise.

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
        no_optH: When True (default), preserve finite input hydrogen coordinates
            and build only missing hydrogens and heavy-atom sidechains. When
            False, residues with complete heavy atoms are packed with OptHSampler
            to optimize hydrogen positions and NHQ flips, while residues with
            missing heavy atoms are rebuilt with DunbrackChiSampler. Generated
            ligand types may still rebuild hydrogens whose names changed during
            parameter generation; pass ``trust_hydrogen_names=True`` only when
            those names are known to match the prepared database.
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
        chem_comp_types: ``{comp_id: type}`` from the input file's
            ``_chem_comp`` table, which says whether a residue belongs to a
            polymer where the file does not number it along a sequence. Only
            used when prepare_ligands=True.
        use_ccd: Whether a residue the input declares no chemistry for may be
            completed from the component dictionary by its code. Pass False for
            a source that supplies whole molecules under codes of its own, such
            as a mol2. Only used when prepare_ligands=True.
        ligand_seed: Fixed RNG seed for the conformer each prepared residue
            is built from, making preparation reproducible. Only used when
            prepare_ligands=True.
        return_context: If True, return ``(pose_stack, PoseBuildContext)``.
        context: Reusable context from ``build_context_from_biotite``. It must
            be on ``torch_device`` and is mutually exclusive with ``param_db``
            and ``prepare_ligands=True``.
        atom37_coords: Optional coordinates shaped ``[pose, token, 37, xyz]``.
            When supplied, mapped coordinates are read from this tensor using
            the input structure's integer ``token_id`` and ``atom37_slot``
            annotations. Wholly finite mapped triplets are authoritative;
            all-NaN mapped triplets are missing and do not fall back to the
            Biotite coordinates. Unmapped finite Biotite atoms remain context,
            and absent leaf atoms are completed normally. Partial-NaN or
            infinite mapped triplets raise :class:`Atom37MappingError`. The
            resulting pose coordinates remain connected to this tensor for
            autograd. Geometry-based missing-density and
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
        if param_db is not None or ligand_params_files:
            raise ValueError(
                "Pass either context= or param_db=/ligand_params_files, not both; the context "
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
            chem_comp_types=chem_comp_types,
            use_ccd=use_ccd,
            ligand_seed=ligand_seed,
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
        _cut_covalent_partners=context.cut_covalent_partners,
    )
    disconnected = kwargs.pop("res_not_connected", None)
    if disconnected is not None:
        if (
            disconnected.shape != (*cf.chain_id.shape, 2)
            or disconnected.dtype != torch.bool
            or disconnected.device != torch_device
        ):
            raise ValueError(
                "res_not_connected must be a boolean [pose, residue, 2] tensor on the pose device"
            )
        bonds = cf.covalent_bonds
        if bonds is not None:
            ports = context.canonical_ordering.polymer_conn_inds
            remove = torch.zeros(len(bonds), dtype=torch.bool, device=torch_device)
            for direction, names in enumerate(
                (ports.down_atom_for_co_restype, ports.up_atom_for_co_restype)
            ):
                lookup = torch.tensor(names, device=torch_device)
                for column in (1, 3):
                    pose, residue = bonds[:, 0], bonds[:, column]
                    remove |= disconnected[pose, residue, direction] & (
                        bonds[:, column + 1] == lookup[cf.res_types[pose, residue]]
                    )
            bonds = bonds[~remove]
        cf = attr.evolve(cf, res_not_connected=disconnected, covalent_bonds=bonds)

    return pose_stack_from_canonical_form_and_context(
        cf,
        context,
        no_optH=no_optH,
        atom37_coords=atom37_coords,
        fragment_mapping=fragment_mapping,
        return_context=return_context,
        **kwargs,
    )


def pose_stack_from_canonical_form_and_context(
    cf: CanonicalForm,
    context: PoseBuildContext,
    *,
    no_optH: bool,
    atom37_coords: torch.Tensor | None,
    fragment_mapping=None,
    return_context: bool = False,
    **kwargs: object,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Build a pose from a canonical form and a reusable build context.

    This is the single construction path. Every entry point -- canonical
    amino-acid Atom37 tensors, an Atom37 batch bound to a fixed topology, a
    parsed CIF/PDB structure -- reduces its input to a
    :py:class:`~tmol.io.CanonicalForm` plus a
    :py:class:`~tmol.io.PoseBuildContext` and finishes here. The entry points
    differ only in how they *derive* those two objects, not in how a pose is
    built from them.

    The context carries the canonical ordering and packed block types, so
    noncanonical residues, ligands, and covalent links are handled by the same
    code as standard amino acids: nothing here inspects an ``AtomArray``.
    Callers that already hold a canonical form and a context -- notably
    repeated guidance or search steps over one fixed topology -- should call
    this directly rather than re-deriving topology per batch.

    Args:
      cf: Canonical-form tensors describing residue identity and coordinates.
      context: Structure-independent chemistry resolved once.
      no_optH: Preserve finite input hydrogens instead of optimizing them.
      atom37_coords: When supplied, the autograd-tracked source of ``cf``'s
        coordinates; retained so gradients survive hydrogen rebuilding.
      fragment_mapping: Mapping produced when fragmented ligands were expanded.
      return_context: Also return the context used.

    Returns:
      The constructed pose, optionally with atom mappings or the context.
    """
    from tmol.io import pose_stack_from_canonical_form
    from tmol.pack import build_missing_sidechains

    if atom37_coords is not None:
        # Both searches are geometric, so leaving either on makes the chemistry
        # depend on how close the coordinates happen to be -- and, because the
        # topology is cached, on which frame of a trajectory arrived first.
        kwargs.setdefault("find_additional_disulfides", False)
        kwargs.setdefault("find_additional_cyclic_closures", False)

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
            has_missing_atoms=has_missing_atoms,
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
    """Reject ligand gaps left unresolved by the coordinate builder.

    Available construction frames have already been used. Polymer rotamer
    sampling cannot place an unanchored ligand or resolve the remaining gaps.
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
            "the available construction frames cannot place them. "
            "Provide enough resolved anchors or a complete ligand structure "
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
    include_virtual_atoms: bool = False,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Convert PoseStack back to Biotite structure.

    Args:
        pose_stack: Pose stack to convert.
        co: Canonical ordering used for conversion. Provide the ordering that
            was used when ligands or custom residue types are present.
        merge_fragments: Restore fragmented ligands to their original residue
            identity. Set to False to keep fragment residues separate.
        include_virtual_atoms: Also write virtual atoms, such as a metal's
            site virtuals.

    Returns:
        Biotite AtomArray for single-pose or AtomArrayStack for multi-pose,
        with every bond the poses' residue types and connections declare:
        bond orders within residues, and chemical bonds between them, metal
        coordination typed COORDINATION.
    """
    if co is None:
        co = canonical_ordering_for_biotite()
    cf = canonical_form_from_pose_stack(co, pose_stack)
    structure, block_for_atom = _biotite_from_canonical_form(
        cf, co, include_virtual_atoms
    )
    structure.bonds = _bonds_from_pose_stack(pose_stack, structure, block_for_atom)
    sbm = getattr(pose_stack, "split_block_mapping", None)
    if merge_fragments and sbm is not None and sbm.entries:
        from tmol.ligand import recombine_fragmented_ligands

        structure = recombine_fragmented_ligands(structure, pose_stack)
    return _renumbered_for_cif(structure)


def _order_name(order):
    return order.upper() if isinstance(order, str) else ChemBondType(int(order)).name


def _chemical_bond_orders(bt, element_of):
    """One Lewis structure for a block type's bonds, by atom-name pair.

    io_bond_orders restore the orders a prepared ligand's typing promoted. What
    remains of tmol's delocalized convention, AROMATIC outside a ring, is
    resolved by rule: at an atom whose delocalized bonds reach terminal
    heteroatoms (carboxylate, guanidinium, phosphate), the first by name is
    double unless the atom already has one, and every other such bond is single.
    """
    order_of, in_ring = {}, {}
    for a, b, order, *rest in bt.bonds:
        key = frozenset((a, b))
        order_of[key] = _order_name(order)
        in_ring[key] = bool(rest[0]) if rest else False
    for a, b, order in bt.io_bond_orders:
        order_of[frozenset((a, b))] = order
    neighbors = defaultdict(set)
    for key in order_of:
        a, b = tuple(key)
        neighbors[a].add(b)
        neighbors[b].add(a)

    def terminal_heteroatom(name):
        heavy = [n for n in neighbors[name] if element_of[n] != "H"]
        return element_of[name] in ("O", "N", "S") and len(heavy) == 1

    delocalized = [
        k for k, order in order_of.items() if order == "AROMATIC" and not in_ring[k]
    ]
    ends_of = defaultdict(list)
    for key in delocalized:
        for center, end in (tuple(key), tuple(key)[::-1]):
            if terminal_heteroatom(end):
                ends_of[center].append(end)
    for center in sorted(ends_of):
        has_double = any(
            order_of[frozenset((center, n))] == "DOUBLE"
            and element_of[n] in ("O", "N", "S")
            for n in neighbors[center]
        )
        for end in sorted(ends_of[center]):
            key = frozenset((center, end))
            if order_of[key] == "AROMATIC":
                order_of[key] = "SINGLE" if has_double else "DOUBLE"
                has_double = True
    for key in delocalized:
        if order_of[key] == "AROMATIC":
            order_of[key] = "SINGLE"
    return order_of


def _bonds_from_pose_stack(pose_stack, structure, block_for_atom):
    """The bond table of an exported structure, from the pose's residue types.

    Bond orders are one Lewis structure: see _chemical_bond_orders. Every pose of
    a stack shares one bond table, so their residue types and connections must
    agree. Bonds to atoms the export left out are dropped.
    """
    block_types = pose_stack.block_type_ind64
    connections = pose_stack.inter_residue_connections64
    if not (
        torch.equal(block_types, block_types[:1].expand_as(block_types))
        and torch.equal(connections, connections[:1].expand_as(connections))
    ):
        raise ValueError(
            "poses with different residue types or connections cannot share "
            "one bond table"
        )
    pbt = pose_stack.packed_block_types
    element_for_type = {at.name: at.element for at in pbt.chem_db.atom_types}
    bt_for_block = block_types[0].tolist()
    irc = connections[0].tolist()
    index = {
        (int(block), str(name)): i
        for i, (block, name) in enumerate(zip(block_for_atom, structure.atom_name))
    }

    def metal_bond(bt, conn):
        sites = bt.metal_sites[0].site_connections if bt.metal_sites else ()
        return bt.connections[conn].name in sites

    orders_for_type = {}
    bonds = []
    for block, bt_ind in enumerate(bt_for_block):
        if bt_ind < 0:
            continue
        bt = pbt.active_block_types[bt_ind]
        element_of = {a.name: element_for_type[a.atom_type] for a in bt.atoms}
        if bt_ind not in orders_for_type:
            orders_for_type[bt_ind] = _chemical_bond_orders(bt, element_of)
        for key, order in orders_for_type[bt_ind].items():
            a, b = tuple(key)
            i, j = index.get((block, a)), index.get((block, b))
            if i is not None and j is not None:
                bonds.append((i, j, order))
        for conn, (partner, partner_conn) in enumerate(irc[block]):
            if partner < 0 or (partner, partner_conn) <= (block, conn):
                continue
            other = pbt.active_block_types[bt_for_block[partner]]
            i = index.get((block, bt.connections[conn].atom))
            j = index.get((partner, other.connections[partner_conn].atom))
            if i is None or j is None:
                continue
            if metal_bond(bt, conn) or metal_bond(other, partner_conn):
                bonds.append((i, j, "COORDINATION"))
            else:
                order = ChemBondType(bt.connection_bond_types[conn]).name
                bonds.append((i, j, "SINGLE" if order == "AROMATIC" else order))

    bond_array = numpy.array(
        [
            (i, j, int(getattr(biotite.structure.BondType, order)))
            for i, j, order in bonds
        ],
        dtype=numpy.int64,
    ).reshape(-1, 3)
    return biotite.structure.BondList(structure.array_length(), bond_array)


def _map_atoms_to_canonical(co, atom_res_inds, res_names, atom_names, elements):
    """Map Biotite atom names to canonical ordering indices.

    Returns (valid_atom_mask, valid_atom_inds, valid_res_inds).
    """

    atom_inds = []
    valid = []
    unmapped = set()
    destinations = set()
    for i, (resname, atname) in enumerate(zip(res_names, atom_names)):
        mapping = co.restypes_atom_index_mapping.get(resname, {})
        idx = mapping.get(atname, -1)
        atom_inds.append(idx)
        valid.append(idx >= 0)
        if idx >= 0:
            destination = (int(atom_res_inds[i]), idx)
            if destination in destinations:
                raise ValueError(
                    f"Multiple input atoms map to canonical atom {idx} of "
                    f"{resname} at residue index {destination[0]} ({atname}). "
                    "Resolve alternate locations and preserve residue identifiers "
                    "and insertion codes before constructing a pose."
                )
            destinations.add(destination)
        if idx < 0 and str(elements[i]).strip().upper() not in ("H", "D"):
            unmapped.add((int(atom_res_inds[i]), str(resname), str(atname)))

    if unmapped:
        details = ", ".join(
            f"{resname} at residue index {res}: {name}"
            for res, resname, name in sorted(unmapped)
        )
        raise ValueError(
            "Heavy atoms are absent from the selected chemical definitions: "
            f"{details}. Supply a matching chemical definition or correct the "
            "input atom names; these atoms cannot be silently discarded."
        )

    valid_atom_mask = numpy.array(valid, dtype=bool)
    atom_inds_arr = numpy.array(atom_inds, dtype=numpy.int64)
    return (
        valid_atom_mask,
        atom_inds_arr[valid_atom_mask],
        atom_res_inds[valid_atom_mask],
    )


def _renumbered_for_cif(structure):
    """Renumber the chains a biotite CIF round trip cannot carry, with a warning.

    biotite writes res_id as both label_seq_id and auth_seq_id, and reads a
    label_seq_id of -1 as missing; label-based readers such as atomworks also
    reject numbering that decreases within a chain. A chain containing -1 is
    shifted to start at 1; a chain whose numbering decreases is renumbered 1..N.
    """
    template = _template_array(structure)
    res_id = template.res_id.copy()
    ins_code = template.ins_code.copy()
    starts = biotite.structure.get_residue_starts(template, add_exclusive_stop=True)
    changed = False
    for chain in dict.fromkeys(template.chain_id.tolist()):
        in_chain = template.chain_id == chain
        ids = res_id[in_chain]
        if (numpy.diff(ids) < 0).any():
            reason = "numbering that decreases within a chain"
            number = 0
            for begin, end in zip(starts[:-1], starts[1:]):
                if template.chain_id[begin] == chain:
                    number += 1
                    res_id[begin:end] = number
            ins_code[in_chain] = ""
        elif (ids == -1).any():
            reason = "a residue id of -1"
            res_id[in_chain] += 1 - ids.min()
        else:
            continue
        warnings.warn(
            f"Renumbering chain {chain} of the output AtomArray: biotite's CIF "
            f"writer does not support {reason}"
        )
        changed = True
    if not changed:
        return structure
    structure = structure.copy()
    structure.res_id = res_id
    structure.ins_code = ins_code
    return structure


def _metal_coordination_bond_mask(structure):
    """Which rows of the bond table are metal coordination.

    These are bonds typed COORDINATION (a CIF's metalc) and bonds from a metal
    to another residue, as a PDB's CONECT lists them.
    """
    bonds = structure.bonds.as_array()
    if not len(bonds):
        return bonds, numpy.zeros(0, dtype=bool)
    template = _template_array(structure)
    metals = [ion["element"].upper() for ion in metal_table()["ions"]]
    is_metal = numpy.isin(numpy.char.upper(template.element.astype(str)), metals)
    residue = biotite.structure.get_residue_positions(
        template, numpy.arange(template.array_length())
    )
    crosses = residue[bonds[:, 0]] != residue[bonds[:, 1]]
    coordination = (bonds[:, 2] == biotite.structure.BondType.COORDINATION) | (
        crosses & (is_metal[bonds[:, 0]] | is_metal[bonds[:, 1]])
    )
    return bonds, coordination


def _without_metal_coordination_bonds(structure):
    """Drop metal coordination bonds, which ligand preparation does not read."""
    if structure.bonds is None:
        return structure
    bonds, coordination = _metal_coordination_bond_mask(structure)
    if not coordination.any():
        return structure
    structure = structure.copy()
    structure.bonds = biotite.structure.BondList(
        _template_array(structure).array_length(), bonds[~coordination]
    )
    return structure


def _with_metal_coordination_typed(structure):
    """Type every metal coordination bond COORDINATION, however it was read."""
    if structure.bonds is None:
        return structure
    bonds, coordination = _metal_coordination_bond_mask(structure)
    untyped = coordination & (bonds[:, 2] != biotite.structure.BondType.COORDINATION)
    if not untyped.any():
        return structure
    structure = structure.copy()
    bonds = bonds.copy()
    bonds[untyped, 2] = biotite.structure.BondType.COORDINATION
    structure.bonds = biotite.structure.BondList(
        _template_array(structure).array_length(), bonds
    )
    return structure


def _metal_coordination_from_biotite(
    array, atom_res_inds, restype_for_res, valid_atom_mask, valid_atom_inds, co
):
    """(metal, -1, donor, donor atom) for each declared metal bond.

    The site is left for detection to choose.
    """
    if array.bonds is None:
        return numpy.zeros((0, 4), dtype=numpy.int64)
    atom_canonical_ind = numpy.full(array.array_length(), -1, dtype=numpy.int64)
    atom_canonical_ind[valid_atom_mask] = valid_atom_inds
    metals = [ion["element"].upper() for ion in metal_table()["ions"]]
    is_metal = numpy.isin(numpy.char.upper(array.element.astype(str)), metals)
    rows = []
    for atom1, atom2, order in array.bonds.as_array():
        if order != biotite.structure.BondType.COORDINATION:
            continue
        if is_metal[atom2] and not is_metal[atom1]:
            atom1, atom2 = atom2, atom1
        if not is_metal[atom1] or is_metal[atom2]:
            continue
        metal, donor = int(atom_res_inds[atom1]), int(atom_res_inds[atom2])
        donor_atom = int(atom_canonical_ind[atom2])
        if metal == donor or donor_atom < 0:
            continue
        rows.append((metal, -1, donor, donor_atom))
    return numpy.array(sorted(set(rows)), dtype=numpy.int64).reshape(-1, 4)


def _template_array(structure):
    """The single AtomArray whose bond table describes every model."""
    if isinstance(structure, biotite.structure.AtomArrayStack):
        return structure[0]
    return structure


def _bonds_for_poses(bonds, n_poses, torch_device):
    """Prefix each bond row with its pose index."""
    if bonds is None:
        return None
    if bonds.shape[0] == 0:
        return torch.zeros(
            (0, bonds.shape[1] + 1), dtype=torch.int64, device=torch_device
        )
    repeated = numpy.tile(bonds, (n_poses, 1))
    pose_column = numpy.repeat(numpy.arange(n_poses), bonds.shape[0])
    return torch.tensor(
        numpy.column_stack((pose_column, repeated)),
        dtype=torch.int64,
        device=torch_device,
    )


def _covalent_bonds_from_biotite(
    array, co, atom_res_inds, restype_for_res, valid_atom_mask, valid_atom_inds
):
    """Declared cross-residue bonds, including nonsequential polymer links.

    Disulfides use the dedicated variant-selection channel, preserving declared
    bonds even when the sulfur coordinates are unresolved or far apart.
    """
    if array.bonds is None:
        return numpy.zeros((0, 4), dtype=numpy.int64), None

    atom_canonical_ind = numpy.full(array.array_length(), -1, dtype=numpy.int64)
    atom_canonical_ind[valid_atom_mask] = valid_atom_inds

    cys_classes = frozenset(co.cys_inds.cys_co_aa_inds)
    sg_atom = co.cys_inds.sg_atom_for_co_cys

    found = []
    disulfides = []
    for atom1, atom2, order in array.bonds.as_array():
        # A metal's bonds are retyped COORDINATION upstream. They are not
        # covalent attachments, and declaring one asks for a residue type that
        # describes an ion bonded through a named connection. The sibling scan
        # below skips them for the same reason.
        if order == biotite.structure.BondType.COORDINATION:
            continue
        res1, res2 = int(atom_res_inds[atom1]), int(atom_res_inds[atom2])
        if res1 == res2:
            continue
        canonical1 = int(atom_canonical_ind[atom1])
        canonical2 = int(atom_canonical_ind[atom2])
        if canonical1 < 0 or canonical2 < 0:
            continue
        restype1, restype2 = restype_for_res[res1], restype_for_res[res2]
        if (
            canonical1 == sg_atom
            and canonical2 == sg_atom
            and restype1 in cys_classes
            and restype2 in cys_classes
        ):
            disulfides.append(tuple(sorted((res1, res2))))
            continue
        if res1 > res2:
            res1, canonical1, res2, canonical2 = res2, canonical2, res1, canonical1
        found.append((res1, canonical1, res2, canonical2))

    return (
        numpy.array(sorted(set(found)), dtype=numpy.int64).reshape(-1, 4),
        numpy.array(sorted(set(disulfides)), dtype=numpy.int64).reshape(-1, 2),
    )


def _res_names_for_structure(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
):
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        return biotite_structure[0].res_name
    return biotite_structure.res_name


def _validate_filtered_covalent_partners(
    array, co, atom_res, valid_res, res_names, cut_partners=frozenset()
):
    """Allow sequential backbone gaps, but not a dangling chemical partner.

    This runs only when a non-water residue is removed. Complete structures
    and ordinary water filtering need no extra bond-table scan.
    """
    if array.bonds is None:
        return
    # A residue whose covalent bond ligand preparation already cut, because the
    # caller passed strict_ligands=False, is a partner it was told to drop.
    removed = (
        ~valid_res & (res_names != "HOH") & ~numpy.isin(res_names, list(cut_partners))
    )
    if not numpy.any(removed):
        return
    bonds = array.bonds.as_array()
    if not len(bonds):
        return
    ends = atom_res[bonds[:, :2]]
    crosses = (removed[ends[:, 0]] & valid_res[ends[:, 1]]) | (
        removed[ends[:, 1]] & valid_res[ends[:, 0]]
    )
    type_indices = {name: i for i, name in enumerate(co.restype_io_equiv_classes)}
    connections = co.polymer_conn_inds
    for first, second, order in bonds[crosses]:
        if order == biotite.structure.BondType.COORDINATION:
            continue
        first_res, second_res = atom_res[[first, second]]
        # Orient the candidate in input residue order, preserving insertion codes.
        if first_res > second_res:
            first, second = second, first
            first_res, second_res = second_res, first_res
        first_name, second_name = array.res_name[[first, second]]
        first_type = type_indices.get(first_name)
        second_type = type_indices.get(second_name)
        if (
            second_res == first_res + 1
            and array.chain_id[first] == array.chain_id[second]
            and first_type is not None
            and second_type is not None
        ):
            first_atom = co.restypes_atom_index_mapping[first_name].get(
                array.atom_name[first], -1
            )
            second_atom = co.restypes_atom_index_mapping[second_name].get(
                array.atom_name[second], -1
            )
            if (
                first_atom >= 0
                and second_atom >= 0
                and (
                    (
                        first_atom == connections.up_atom_for_co_restype[first_type]
                        and second_atom
                        == connections.down_atom_for_co_restype[second_type]
                    )
                    or (
                        # Reordering an explicitly numbered chain does not turn
                        # an ordinary gap into a cyclic/crosslinked attachment.
                        # Require the source sequence direction; a backward link
                        # in forward residue order still closes a cycle.
                        first_atom == connections.down_atom_for_co_restype[first_type]
                        and second_atom
                        == connections.up_atom_for_co_restype[second_type]
                        and (array.res_id[first], array.ins_code[first])
                        > (array.res_id[second], array.ins_code[second])
                    )
                )
            ):
                continue

        def label(index):
            return (
                f"{array.res_name[index]} {array.chain_id[index]}:"
                f"{array.res_id[index]}{array.ins_code[index]}"
                f"/{array.atom_name[index]}"
            )

        raise ValueError(
            "Cannot discard an incomplete or unsupported residue while retaining "
            "its covalent partner: declared bond "
            f"{label(first)} -- {label(second)} would be lost. "
            "Supply the required backbone coordinates/chemical definition, or "
            "explicitly select a complete covalent component before construction."
        )


def _filter_supported_atoms_and_connectivity(  # noqa: C901
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    co: CanonicalOrdering,
    *,
    filter_missing_mainchain: bool = True,
    cut_partners: frozenset[str] = frozenset(),
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
    valid_res = numpy.array([name not in to_remove for name in res_names], dtype=bool)[
        biotite_residue_starts
    ]

    # Filter residues missing mainchain atoms required for rotamer building.
    # Only atoms present in every variant count as required, so an atom a terminus
    # patch removes (the DNA 5' phosphate) does not disqualify the residue.
    # Residues with no mainchain definition (non-polymer) are skipped.
    if filter_missing_mainchain:
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
            mapping = co.restypes_atom_index_mapping[res_name3]
            resolved = numpy.isfinite(coords[:, start:end, :]).all(axis=(0, 2))
            present = {
                mapping[name]
                for name in atom_names[start:end][resolved]
                if name in mapping
            }
            missing = {name for name in required if mapping[name] not in present}
            if missing:
                logger.warning(
                    "Residue %s %s %d is missing mainchain atoms %s; skipping",
                    biotite_structure.chain_id[start],
                    res_name3,
                    biotite_structure.res_id[start],
                    sorted(missing),
                )
                valid_res[i] = False

    atom_res = get_all_residue_positions(biotite_structure)
    _validate_filtered_covalent_partners(
        _template_array(biotite_structure),
        co,
        atom_res,
        valid_res,
        biotite_residues,
        cut_partners,
    )
    valid_atoms = valid_res[atom_res]

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
    is_polymeric: numpy.ndarray | None = None,
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
        is_polymeric: Shape (n_res,) boolean array; pairs where either residue
            is not a chain member are left alone.
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
        # A ligand numbered in the chain it sits in is not the next link of
        #    that chain, so its distance says nothing about a break. Marking
        #    one would take the C-terminus off the residue before it.
        if is_polymeric is not None and not (is_polymeric[i] and is_polymeric[i + 1]):
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


def _orient_polymer_gap_flags(not_connected, chain_id, restypes, bonds, co):
    """Translate input-neighbor gap flags to chemical down/up on reversed chains.

    Declared adjacent polymer bonds establish direction. Non-polymer links and
    cyclic closures do not vote; mixed directions have no single chain ordering.
    """
    if not len(bonds) or not numpy.any(not_connected):
        return
    first, a, second, b = bonds.T
    conn = co.polymer_conn_inds
    up = numpy.asarray(conn.up_atom_for_co_restype)[restypes]
    down = numpy.asarray(conn.down_atom_for_co_restype)[restypes]
    adjacent = (second == first + 1) & (chain_id[first] == chain_id[second])
    forward = adjacent & (a == up[first]) & (b == down[second])
    reverse = adjacent & (a == down[first]) & (b == up[second])
    for chain in numpy.unique(chain_id[first[reverse]]):
        if not numpy.any(forward & (chain_id[first] == chain)):
            members = chain_id == chain
            not_connected[members] = not_connected[members, ::-1]


def _extract_residue_metadata(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    not_connected,
):
    biotite_residue_starts = biotite.structure.get_residue_starts(biotite_structure)

    # Residue labels are not chain identities. Biotite's get_chain_starts also
    # splits whenever res_id decreases, turning a reversed chain into one chain
    # per residue. Work at residue granularity and retain explicit symmetry IDs.
    # Author chain labels can be shared by a polymer and separate ligand
    # entities. That must not hide the polymer's terminal boundary.
    keys = ["chain_id"]
    if "label_entity_id" in biotite_structure.get_annotation_categories():
        keys.append("label_entity_id")
    if "sym_id" in biotite_structure.get_annotation_categories():
        keys.append("sym_id")
    boundaries = numpy.zeros(max(0, len(biotite_residue_starts) - 1), dtype=bool)
    for key in keys:
        values = biotite_structure.get_annotation(key)[biotite_residue_starts]
        boundaries |= values[1:] != values[:-1]
    biotite_chain_id_for_res = numpy.cumsum(numpy.r_[0, boundaries])[
        : len(biotite_residue_starts)
    ]

    if len(biotite_chain_id_for_res) > 1:
        res_is_disconnected_from_neighbor = (
            biotite_chain_id_for_res[1:] != biotite_chain_id_for_res[:-1]
        )
        not_connected[1:, 0] &= ~res_is_disconnected_from_neighbor
        not_connected[:-1, 1] &= ~res_is_disconnected_from_neighbor

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
    biotite_coords = torch.as_tensor(biotite_structure.coord, device=torch_device)
    if biotite_coords.ndim == 2:
        biotite_coords = biotite_coords.unsqueeze(0)
    tmol_coords[:, valid_res_inds, valid_atom_inds] = biotite_coords[:, valid_atom_mask]
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


def _validate_mapped_atom37_triplets(
    source_coords: torch.Tensor,
    mapped_token_id: torch.Tensor,
    mapped_slot: torch.Tensor,
) -> None:
    """Reject mapped triplets that are neither wholly finite nor all-NaN."""
    has_infinity = torch.isinf(source_coords).any(dim=-1)
    nan_count = torch.isnan(source_coords).sum(dim=-1)
    partial_nan = (nan_count > 0) & (nan_count < 3)
    malformed = has_infinity | partial_nan
    if not bool(torch.any(malformed)):
        return

    details = []
    for pose, mapped_atom in (
        torch.nonzero(malformed, as_tuple=False).cpu().tolist()[:20]
    ):
        reason = (
            "contains infinity" if has_infinity[pose, mapped_atom] else "partial NaN"
        )
        details.append(
            f"pose={pose} token_id={int(mapped_token_id[mapped_atom])} "
            f"atom37_slot={int(mapped_slot[mapped_atom])} ({reason})"
        )
    count = int(torch.count_nonzero(malformed))
    tail = f"; ... and {count - 20} more" if count > 20 else ""
    raise Atom37MappingError(
        "Malformed mapped Atom37 coordinate triplet(s): "
        + "; ".join(details)
        + tail
        + ". Each mapped triplet must be wholly finite or exactly "
        "[NaN, NaN, NaN] for a missing atom."
    )


def _required_mainchain_entries(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    co: CanonicalOrdering,
) -> tuple[tuple[int, int, str], ...]:
    """Describe canonical mainchain atoms required for pose construction."""
    starts = biotite.structure.get_residue_starts(biotite_structure)
    residue_names = biotite.structure.get_residues(biotite_structure)[1]
    entries = []
    for residue, (start, residue_name) in enumerate(zip(starts, residue_names)):
        required = co.restypes_required_mainchain_atoms.get(residue_name) or ()
        mapping = co.restypes_atom_index_mapping[residue_name]
        label = (
            f"{residue_name} {biotite_structure.chain_id[start]}:"
            f"{biotite_structure.res_id[start]}"
            f"{biotite_structure.ins_code[start]}"
        )
        entries.extend(
            (residue, mapping[atom_name], f"{label}/{atom_name}")
            for atom_name in required
        )
    return tuple(entries)


def _validate_effective_mainchain_coords(
    canonical_coords: torch.Tensor,
    required_mainchain_entries: tuple[tuple[int, int, str], ...],
) -> None:
    """Require every pose's effective mainchain coordinates to be finite."""
    if not required_mainchain_entries:
        return

    # Gather every (pose, entry) at once. Testing one entry at a time costs a
    # device synchronization each, which otherwise dominates repeated replay.
    device = canonical_coords.device
    residues = torch.tensor(
        [entry[0] for entry in required_mainchain_entries],
        dtype=torch.int64,
        device=device,
    )
    atoms = torch.tensor(
        [entry[1] for entry in required_mainchain_entries],
        dtype=torch.int64,
        device=device,
    )
    absent = ~torch.isfinite(canonical_coords[:, residues, atoms]).all(dim=-1)
    if not bool(absent.any()):
        return

    # Entry-major, pose-ascending: the order the per-entry loop reported.
    missing = [
        f"pose={pose} residue={required_mainchain_entries[entry][2]}"
        for entry, pose in torch.nonzero(absent.t(), as_tuple=False).cpu().tolist()
    ]

    head = missing[:20]
    tail = f"; ... and {len(missing) - 20} more" if len(missing) > 20 else ""
    raise Atom37MappingError(
        "Required mainchain coordinates are missing from both usable "
        "Atom37/Biotite coordinate sources: "
        + "; ".join(head)
        + tail
        + ". Supply a wholly finite mapped Atom37 triplet or, for an unmapped "
        "atom, a finite Biotite reference coordinate. Mapped all-NaN triplets "
        "are missing and never fall back to the reference."
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

    Mapped tensor values replace their matching Biotite atoms through one
    differentiable indexed assignment. Wholly finite triplets are authoritative
    and all-NaN triplets are missing; only unmapped atoms retain reference
    coordinates. Partial-NaN and infinite mapped triplets are rejected.

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
    _validate_mapped_atom37_triplets(source_coords, mapped_token_id, mapped_slot)
    mapped_res_inds = torch.as_tensor(mapped_res_inds, device=torch_device)
    mapped_atom_inds = torch.as_tensor(mapped_atom_inds, device=torch_device)
    reference_coords[:, mapped_res_inds, mapped_atom_inds] = source_coords
    return reference_coords, n_poses


@validate_args
def _normalize_input_identifiers(biotite_structure, name3_aliases):
    """Resolve residue aliases and AtomWorks assembly instances without mutation.

    An aliased residue is read as the one it names, so nothing downstream --
    atom mapping, restype lookup, nonstandard-residue detection -- ever sees
    the input name. Atom names follow through the target's atom aliases.
    """
    from atomworks.io.utils.atom_array import chain_identifier

    chains = chain_identifier(biotite_structure)
    names = biotite_structure.res_name
    rename_residues = bool(name3_aliases) and any(
        name in name3_aliases for name in numpy.unique(names)
    )
    if not rename_residues and (
        chains is biotite_structure.chain_id
        or numpy.array_equal(chains, biotite_structure.chain_id)
    ):
        return biotite_structure
    renamed = biotite_structure.copy()
    renamed.chain_id = chains.copy()
    if rename_residues:
        renamed.res_name = numpy.array([name3_aliases.get(n, n) for n in names])
    return renamed


def canonical_form_from_biotite(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering | None = None,
    missing_density_distance_threshold: float = 2.4,
    atom37_coords: torch.Tensor | None = None,
    _filter_missing_mainchain: bool = True,
    _cut_covalent_partners: frozenset[str] = frozenset(),
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
            annotations on ``biotite_structure``); all-NaN mapped triplets are
            treated as missing while unmapped finite reference atoms remain
            context. The geometry-based missing-density check is skipped so the
            topology stays fixed and gradients flow. See
            :func:`~tmol.io.pose_stack_from_atom37_and_topology`.

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
            - disulfides: Explicit cysteine sulfur bonds, including unresolved SG
            - res_not_connected: Tensor describing whether two consecutive residues
              should be treated as chemically bonded.

    """
    torch_device = resolve_device(torch_device)
    if co is None:
        co = canonical_ordering_for_biotite()

    biotite_structure = _normalize_input_identifiers(
        biotite_structure, co.name3_aliases
    )
    biotite_structure = _with_metal_coordination_typed(biotite_structure)
    biotite_structure, not_connected = _filter_supported_atoms_and_connectivity(
        biotite_structure,
        co,
        filter_missing_mainchain=(_filter_missing_mainchain and atom37_coords is None),
        cut_partners=_cut_covalent_partners,
    )
    (
        biotite_chain_id_for_res,
        biotite_chain_labels,
        biotite_insertion_codes,
        biotite_residue_labels,
        biotite_residues,
    ) = _extract_residue_metadata(biotite_structure, not_connected)

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
        biotite_structure.element,
    )
    covalent_bonds_np, disulfides_np = _covalent_bonds_from_biotite(
        _template_array(biotite_structure),
        co,
        atom_res_inds,
        tmol_restypes,
        valid_atom_mask,
        valid_atom_inds,
    )
    metal_coordination_np = _metal_coordination_from_biotite(
        _template_array(biotite_structure),
        atom_res_inds,
        tmol_restypes,
        valid_atom_mask,
        valid_atom_inds,
        co,
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
        _validate_effective_mainchain_coords(
            tmol_coords, _required_mainchain_entries(biotite_structure, co)
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
        conn_inds = co.polymer_conn_inds
        polymeric = numpy.array(
            [
                conn_inds.down_atom_for_co_restype[restype] >= 0
                or conn_inds.up_atom_for_co_restype[restype] >= 0
                for restype in tmol_restypes
            ]
        )
        _break_connections_for_missing_density(
            not_connected,
            biotite_chain_id_for_res,
            tmol_coords,
            missing_density_distance_threshold,
            polymeric,
        )
    _orient_polymer_gap_flags(
        not_connected, biotite_chain_id_for_res, tmol_restypes, covalent_bonds_np, co
    )
    res_not_connected = (
        torch.tensor(not_connected, dtype=torch.bool, device=torch_device)
        .unsqueeze(0)
        .repeat(n_poses, 1, 1)
    )

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
        disulfides=_bonds_for_poses(disulfides_np, n_poses, torch_device),
        res_not_connected=res_not_connected,
        covalent_bonds=_bonds_for_poses(covalent_bonds_np, n_poses, torch_device),
        metal_coordination=_bonds_for_poses(
            metal_coordination_np, n_poses, torch_device
        ),
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
def biotite_from_canonical_form(
    cf: CanonicalForm,
    co: CanonicalOrdering | None = None,
    include_virtual_atoms: bool = False,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Export coordinates and chemical/author labels using a shared atom layout.

    Multi-model arrays require identical residue and atom annotations. Their atom
    layout is the union of resolved atoms; absent coordinates remain NaN. Missing
    author labels default to internal chain IDs and one-based residue positions.
    Virtual atoms are left out unless ``include_virtual_atoms``. This host-array
    export detaches coordinates from autograd. A canonical form carries no
    residue types, so the result has no bonds; biotite_from_pose_stack adds them.
    """
    return _biotite_from_canonical_form(cf, co, include_virtual_atoms)[0]


def _biotite_from_canonical_form(cf, co, include_virtual_atoms):
    """The exported structure, and the canonical residue of each of its atoms."""
    import biotite.structure as struc

    if co is None:
        co = canonical_ordering_for_biotite()
    n_poses, n_residues, max_atoms = cf.coords.shape[:3]
    if n_poses > 1 and not _poses_have_identical_metadata(cf):
        raise ValueError(
            "Cannot convert CanonicalForm with multiple poses to biotite structure: "
            "poses have different metadata. Only coordinate differences are allowed "
            "for multi-pose conversion."
        )

    coords = cf.coords.detach().cpu().numpy()
    res_types = cf.res_types[0].cpu().numpy()
    present = ~numpy.isnan(coords).any(axis=-1)
    atom_mask = present.any(axis=0)
    names, elements, rows, columns = [], [], [], []
    for res_id in numpy.flatnonzero(res_types >= 0):
        res_name = co.restype_io_equiv_classes[res_types[res_id]]
        atom_names = co.restypes_ordered_atom_names[res_name][:max_atoms]
        virtual = (
            frozenset()
            if include_virtual_atoms
            else co.restypes_virtual_atoms.get(res_name, frozenset())
        )
        indices = [
            i
            for i in numpy.flatnonzero(atom_mask[res_id, : len(atom_names)])
            if atom_names[i] not in virtual
        ]
        names.extend(atom_names[i] for i in indices)
        elements.extend(
            co.restypes_atom_elements[res_name][atom_names[i]] for i in indices
        )
        rows.extend([res_id] * len(indices))
        columns.extend(indices)
    rows, columns = numpy.asarray(rows, dtype=int), numpy.asarray(columns, dtype=int)
    result = (
        struc.AtomArray(len(rows))
        if n_poses == 1
        else struc.AtomArrayStack(n_poses, len(rows))
    )
    selected = coords[:, rows, columns]
    selected[~present[:, rows, columns]] = numpy.nan
    result.coord = selected[0] if n_poses == 1 else selected
    result.set_annotation("atom_name", numpy.asarray(names, dtype=str))
    result.set_annotation("element", numpy.asarray(elements, dtype=str))
    result.set_annotation(
        "res_name", numpy.asarray(co.restype_io_equiv_classes)[res_types[rows]]
    )
    chain_labels = (
        cf.chain_labels
        if cf.chain_labels is not None
        else cf.chain_id.cpu().numpy().astype(str)
    )
    res_labels = (
        cf.res_labels
        if cf.res_labels is not None
        else numpy.arange(1, n_residues + 1)[None, :]
    )
    result.set_annotation("chain_id", numpy.asarray(chain_labels[0, rows], dtype=str))
    result.set_annotation("res_id", res_labels[0, rows])
    if cf.residue_insertion_codes is not None:
        result.set_annotation(
            "ins_code", numpy.asarray(cf.residue_insertion_codes[0, rows], dtype=str)
        )
    for name, values in (
        ("b_factor", cf.atom_b_factor),
        ("occupancy", cf.atom_occupancy),
    ):
        if values is not None:
            result.set_annotation(name, values[0, rows, columns].copy())
    return result, rows


@validate_args
def _poses_have_identical_metadata(cf: CanonicalForm) -> bool:
    """Check if all poses in the CanonicalForm have identical metadata.

    Returns True if all poses have the same:
    - chain_id
    - res_types
    - res_labels
    - residue_insertion_codes
    - chain_labels
    - atom_occupancy and atom_b_factor

    Only coordinates are allowed to differ between poses.
    """
    n_poses = cf.coords.size(0)

    if n_poses <= 1:
        return True

    if not torch.all(cf.chain_id[0] == cf.chain_id[1:]).item():
        return False

    if not torch.all(cf.res_types[0] == cf.res_types[1:]).item():
        return False

    for values in (
        cf.res_labels,
        cf.residue_insertion_codes,
        cf.chain_labels,
        cf.atom_b_factor,
        cf.atom_occupancy,
    ):
        if values is not None and not all(
            numpy.array_equal(values[0], row, equal_nan=values.dtype.kind in "fc")
            for row in values[1:]
        ):
            return False
    return True
