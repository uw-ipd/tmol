import math

import numpy
import torch
from dataclasses import dataclass

from .._atom_type_dependent_term import AtomTypeDependentTerm
from .._bond_dependent_term import BondDependentTerm
from .._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    latest_annotation,
    store_annotation,
)

from tmol.database import ParameterDatabase
from tmol.score.elec import ElecParamResolver, ElecGlobalParams
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)


@dataclass(frozen=True)
class _BlockGeometry:
    representatives: numpy.ndarray
    inter: numpy.ndarray
    intra: numpy.ndarray


@dataclass(frozen=True)
class _BlockParameters:
    charges: numpy.ndarray
    geometry: _BlockGeometry


@dataclass(frozen=True)
class _PackedParameters:
    rosetta_typed: frozenset
    charges: torch.Tensor
    inter: torch.Tensor
    intra: torch.Tensor
    all_ligand_typed: torch.Tensor
    block_geometry: tuple


class ElecEnergyTerm(AtomTypeDependentTerm, BondDependentTerm):
    param_resolver: ElecParamResolver
    global_params: ElecGlobalParams

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        param_resolver = ElecParamResolver.from_database(
            param_db.scoring.elec, device=device
        )
        super(ElecEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.param_resolver = param_resolver
        self.global_params = self.param_resolver.global_params
        self.rosetta_typed = frozenset(param_db.scoring.genbonded.rosetta_typed)
        # One most-recent annotation per BT/PBT. A weak owner reference cannot
        # alias a reused id or keep old parameter databases alive.
        self.elec_database = param_db.scoring.elec
        # genbonded belongs here as well as in the packed key below: the intra
        # count-pair rule now reads rosetta_typed, so two terms differing only in
        # their genbonded database must not share a block annotation. Evolving
        # genbonded leaves scoring.elec the same object, so an elec-only key
        # cannot tell them apart and the second term silently inherits the
        # first's 1-4 encoding.
        self._block_annotation_key = AnnotationKey.from_sources(
            self.elec_database, param_db.scoring.genbonded
        )
        self._packed_annotation_key = AnnotationKey.from_sources(
            self.elec_database,
            param_db.scoring.genbonded,
            settings=(self.device,),
        )
        self._scoring_global_params = None
        self._max_dis = float(self.elec_database.global_parameters.elec_max_dis)

    @classmethod
    def class_name(cls):
        return "Elec"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._elec_creator

        return tmol.score.terms._elec_creator.ElecTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ElecEnergyTerm, self).setup_block_type(block_type)
        cached = cached_annotation(
            block_type, "_elec_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached
        previous = latest_annotation(block_type, "_elec_annotation")
        partial_charge = self.param_resolver.get_partial_charges_for_block(block_type)
        representative_mapping = (
            self.param_resolver.get_bonded_path_length_mapping_for_block(block_type)
        )
        if previous is not None and numpy.array_equal(
            representative_mapping, previous.geometry.representatives
        ):
            geometry = previous.geometry
        else:
            # inter[a,b] = path_dist[a, rep(b)]; intra[a,b] =
            # path_dist[rep(a), rep(b)]. Reuse these quadratic arrays when
            # only charges change. Advanced indexing already owns the arrays.
            inter = block_type.path_distance[:, representative_mapping]
            intra = inter[representative_mapping, :]
            # Full weight for 3/4-bond paths between ligand-typed atoms, keyed
            # on the atoms rather than on the residue -- the same convention
            # ljlk uses, and elec's own inter rule. A noncanonical with a
            # ligand-typed sidechain is still a polymer, so a residue-level
            # test left ljlk weighting such a 1-4 pair at 1.0 while elec
            # weighted it at 0.2.
            ligand = numpy.array(
                [a.atom_type not in self.rosetta_typed for a in block_type.atoms],
                dtype=bool,
            )
            both = ligand[:, None] & ligand[None, :]
            intra[both & ((intra == 3) | (intra == 4))] = 5
            geometry = _BlockGeometry(representative_mapping, inter, intra)
        setattr(block_type, "elec_partial_charge", partial_charge)
        setattr(block_type, "elec_inter_repr_path_distance", geometry.inter)
        setattr(block_type, "elec_intra_repr_path_distance", geometry.intra)
        annotation = _BlockParameters(partial_charge, geometry)
        return store_annotation(
            block_type,
            "_elec_annotation",
            self._block_annotation_key,
            annotation,
            fields=(
                "elec_partial_charge",
                "elec_inter_repr_path_distance",
                "elec_intra_repr_path_distance",
            ),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ElecEnergyTerm, self).setup_packed_block_types(packed_block_types)
        pbt = packed_block_types
        cached = cached_annotation(pbt, "_elec_annotation", self._packed_annotation_key)
        if cached is not None:
            return cached
        previous = latest_annotation(pbt, "_elec_annotation")
        if previous is not None and previous.all_ligand_typed.device != self.device:
            # latest_annotation returns the newest entry without comparing the
            # key, so it may have been staged for another device.
            previous = None
        block_parameters = [self.setup_block_type(bt) for bt in pbt.active_block_types]
        geometry = tuple(p.geometry for p in block_parameters)
        reuse_geometry = (
            previous is not None
            and len(geometry) == len(previous.block_geometry)
            and all(a is b for a, b in zip(geometry, previous.block_geometry))
        )

        # Stage once on the host, retaining geometry tensors when only charges change.
        charges = numpy.zeros((pbt.n_types, pbt.max_n_atoms), dtype=numpy.float32)
        if not reuse_geometry:
            shape = (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms)
            inter = numpy.zeros(shape, dtype=numpy.int32)
            intra = numpy.zeros(shape, dtype=numpy.int32)
        for i, (bt, params) in enumerate(zip(pbt.active_block_types, block_parameters)):
            charges[i, : bt.n_atoms] = params.charges
            if not reuse_geometry:
                inter[i, : bt.n_atoms, : bt.n_atoms] = params.geometry.inter
                intra[i, : bt.n_atoms, : bt.n_atoms] = params.geometry.intra
        elec_partial_charge = torch.as_tensor(charges, device=self.device)
        elec_inter_repr_path_distance = (
            previous.inter
            if reuse_geometry
            else torch.as_tensor(inter, device=self.device)
        )
        elec_intra_repr_path_distance = (
            previous.intra
            if reuse_geometry
            else torch.as_tensor(intra, device=self.device)
        )
        pbt.elec_partial_charge = elec_partial_charge
        all_ligand_typed = (
            previous.all_ligand_typed
            if previous is not None and previous.rosetta_typed == self.rosetta_typed
            else torch.tensor(
                [
                    all(a.atom_type not in self.rosetta_typed for a in bt.atoms)
                    for bt in packed_block_types.active_block_types
                ],
                dtype=torch.int32,
                device=self.device,
            )
        )
        setattr(pbt, "elec_all_atoms_ligand_typed", all_ligand_typed)
        setattr(
            packed_block_types,
            "elec_inter_repr_path_distance",
            torch.as_tensor(elec_inter_repr_path_distance, device=self.device),
        )
        setattr(
            packed_block_types,
            "elec_intra_repr_path_distance",
            torch.as_tensor(elec_intra_repr_path_distance, device=self.device),
        )
        annotation = _PackedParameters(
            self.rosetta_typed,
            elec_partial_charge,
            elec_inter_repr_path_distance,
            elec_intra_repr_path_distance,
            all_ligand_typed,
            geometry,
        )
        return store_annotation(
            pbt,
            "_elec_annotation",
            self._packed_annotation_key,
            annotation,
            fields=(
                "elec_partial_charge",
                "elec_inter_repr_path_distance",
                "elec_intra_repr_path_distance",
                "elec_all_atoms_ligand_typed",
            ),
            bindings=tuple(
                (block_type, field)
                for block_type in pbt.active_block_types
                for field in (
                    "elec_partial_charge",
                    "elec_inter_repr_path_distance",
                    "elec_intra_repr_path_distance",
                )
            ),
        )

    def setup_poses(self, poses: PoseStack):
        super(ElecEnergyTerm, self).setup_poses(poses)

    score_only_in_no_grad = True

    def get_pose_score_term_function(self):
        from tmol.score.elec.potentials import elec_pose_scores

        return elec_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.elec.potentials import elec_rotamer_scores_shared

        return elec_rotamer_scores_shared

    def get_block_neighbor_cutoff(self):
        return self._max_dis

    def rotamer_dispatch_key(self):
        return "sphere_overlap"

    def accepts_shared_rotamer_dispatch(self):
        return True

    def get_score_term_attributes(self, pose_stack):
        if self._scoring_global_params is None:
            D = float(self.elec_database.global_parameters.elec_sigmoidal_die_D)
            D0 = float(self.elec_database.global_parameters.elec_sigmoidal_die_D0)
            S = float(self.elec_database.global_parameters.elec_sigmoidal_die_S)
            min_dis = float(self.elec_database.global_parameters.elec_min_dis)
            max_dis = float(self.elec_database.global_parameters.elec_max_dis)

            def eps(dist):
                return D - 0.5 * (D - D0) * (
                    2 + 2 * dist * S + dist * dist * S * S
                ) * math.exp(-dist * S)

            def deps(dist):
                return 0.5 * (D - D0) * dist * dist * S * S * S * math.exp(-dist * S)

            eps_at_cutoff = eps(max_dis)
            cutoff_offset = 322.0637 / (max_dis * eps_at_cutoff)
            min_score = 322.0637 / (min_dis * eps(min_dis)) - cutoff_offset

            low_end = min_dis + 0.25
            low_eps = eps(low_end)
            low_score = 322.0637 / (low_end * low_eps) - cutoff_offset
            low_deriv = (
                -322.0637
                * (low_eps + low_end * deps(low_end))
                / (low_end * low_end * low_eps * low_eps)
            )

            high_start = max_dis - 1.0
            high_eps = eps(high_start)
            high_score = 322.0637 / (high_start * high_eps) - cutoff_offset
            high_deriv = (
                -322.0637
                * (high_eps + high_start * deps(high_start))
                / (high_start * high_start * high_eps * high_eps)
            )

            self._scoring_global_params = torch.tensor(
                [
                    D,
                    D0,
                    S,
                    min_dis,
                    max_dis,
                    cutoff_offset,
                    min_score,
                    low_score,
                    low_deriv,
                    high_score,
                    high_deriv,
                ],
                dtype=torch.float32,
                device=pose_stack.device,
            )[None, :]
            self._max_dis = float(max_dis)
        # Capture this term's tables even after another database used the PBT.
        # Previously rendered modules retain their original tensor references.
        parameters = self.setup_packed_block_types(pose_stack.packed_block_types)

        return [
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            parameters.charges,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            parameters.inter,
            parameters.intra,
            parameters.all_ligand_typed,
            self._scoring_global_params,
            # elec_max_dis as host scalar for detect-neighbors call
            self._max_dis,
        ]
