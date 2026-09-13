import torch
import numpy
import weakref
from dataclasses import dataclass

from .._atom_type_dependent_term import AtomTypeDependentTerm
from .._bond_dependent_term import BondDependentTerm

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
    database: weakref.ReferenceType
    charges: numpy.ndarray
    geometry: _BlockGeometry


@dataclass(frozen=True)
class _PackedParameters:
    database: weakref.ReferenceType
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
        self._database_ref = weakref.ref(self.elec_database)
        self._global_parameters = torch.stack(
            [
                self.global_params.elec_sigmoidal_die_D,
                self.global_params.elec_sigmoidal_die_D0,
                self.global_params.elec_sigmoidal_die_S,
                self.global_params.elec_min_dis,
                self.global_params.elec_max_dis,
            ]
        )[None, :]
        # Keep the host cutoff without converting a CUDA scalar during render.
        self._max_distance = float(self.elec_database.global_parameters.elec_max_dis)

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
        previous = getattr(block_type, "_elec_parameters", None)
        if previous is not None and previous.database() is self.elec_database:
            return previous
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
            if not block_type.properties.polymer.is_polymer:
                # Ligands use full weight for paths of 3/4 bonds.
                intra[(intra == 3) | (intra == 4)] = 5
            geometry = _BlockGeometry(representative_mapping, inter, intra)
        setattr(block_type, "elec_partial_charge", partial_charge)
        setattr(block_type, "elec_inter_repr_path_distance", geometry.inter)
        setattr(block_type, "elec_intra_repr_path_distance", geometry.intra)
        annotation = _BlockParameters(self._database_ref, partial_charge, geometry)
        block_type._elec_parameters = annotation
        return annotation

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ElecEnergyTerm, self).setup_packed_block_types(packed_block_types)
        pbt = packed_block_types
        previous = getattr(pbt, "_elec_parameters", None)
        if (
            previous is not None
            and previous.database() is self.elec_database
            and previous.rosetta_typed == self.rosetta_typed
        ):
            return previous
        block_parameters = [self.setup_block_type(bt) for bt in pbt.active_block_types]
        geometry = tuple(p.geometry for p in block_parameters)
        reuse_geometry = previous is not None and all(
            a is b for a, b in zip(geometry, previous.block_geometry)
        )

        def _ti(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        def _tf(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        elec_partial_charge = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms), dtype=torch.float32, device=self.device
        )
        elec_inter_repr_path_distance = (
            previous.inter
            if reuse_geometry
            else torch.zeros(
                (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
                dtype=torch.int32,
                device=self.device,
            )
        )
        elec_intra_repr_path_distance = (
            previous.intra
            if reuse_geometry
            else torch.zeros(
                (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
                dtype=torch.int32,
                device=self.device,
            )
        )

        for i, (bt, params) in enumerate(zip(pbt.active_block_types, block_parameters)):
            elec_partial_charge[i, : bt.n_atoms] = _tf(params.charges)
            if not reuse_geometry:
                elec_inter_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                    params.geometry.inter
                )
                elec_intra_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                    params.geometry.intra
                )

        setattr(packed_block_types, "elec_partial_charge", elec_partial_charge)
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
            elec_inter_repr_path_distance,
        )
        setattr(
            packed_block_types,
            "elec_intra_repr_path_distance",
            elec_intra_repr_path_distance,
        )
        annotation = _PackedParameters(
            self._database_ref,
            self.rosetta_typed,
            elec_partial_charge,
            elec_inter_repr_path_distance,
            elec_intra_repr_path_distance,
            all_ligand_typed,
            geometry,
        )
        pbt._elec_parameters = annotation
        return annotation

    def setup_poses(self, poses: PoseStack):
        super(ElecEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.elec.potentials import elec_pose_scores

        return elec_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.elec.potentials import elec_rotamer_scores

        return elec_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
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
            self._global_parameters,
            # elec_max_dis as host scalar for detect-neighbors call
            self._max_distance,
        ]
