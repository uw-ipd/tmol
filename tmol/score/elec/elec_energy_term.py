import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm

from tmol.database import ParameterDatabase
from tmol.score.elec.params import ElecParamResolver, ElecGlobalParams
from tmol.score.elec.elec_whole_pose_module import ElecWholePoseScoringModule
from tmol.score.elec.potentials.compiled import elec_pose_scores, elec_rotamer_scores

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


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

    @classmethod
    def class_name(cls):
        return "Elec"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.elec_creator

        return tmol.score.terms.elec_creator.ElecTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ElecEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "elec_inter_repr_path_distance"):
            assert hasattr(block_type, "elec_intra_repr_path_distance")
            assert hasattr(block_type, "elec_partial_charge")
            return

        # fd let's try not to grab data members from the param resolver ...

        partial_charge = self.param_resolver.get_partial_charges_for_block(block_type)

        # count pair representative logic:
        # decide whether or not two atoms i and j should have their interaction counted
        # on the basis of the number of chemical bonds that separate their respective
        # representative atoms r(i) and r(j).
        # We will create a tensor to measure the distance of all representative atoms
        # to all connection atoms:
        # Let rpd be the acronym for "representative_path_distance"
        # inter_rpd[a,b] will be the number of chemical bonds between atom a and
        # atom r(b). Then we can answer how far apart the representative atoms are for
        # atoms i and j on residues k and l by computing:
        # min(ci cj, inter_rpd_k[ck, i] + inter_rpd_l[cl, j] + sep(ck, cl))
        # over all pairs of connection atoms ck and cl on residues k and l.
        # The second tensor intra_rpd[a,b] will hold path_dist[rep(a), rep(b)] so that
        # it can be looked up directly in the intra-block energy evaluation step
        representative_mapping = (
            self.param_resolver.get_bonded_path_length_mapping_for_block(block_type)
        )

        inter_rep_path_dist = block_type.path_distance[:, representative_mapping]
        intra_rep_path_dist = inter_rep_path_dist[representative_mapping, :]

        setattr(block_type, "elec_partial_charge", partial_charge)
        setattr(block_type, "elec_inter_repr_path_distance", inter_rep_path_dist)
        setattr(block_type, "elec_intra_repr_path_distance", intra_rep_path_dist)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ElecEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "elec_inter_repr_path_distance"):
            assert hasattr(packed_block_types, "elec_intra_repr_path_distance")
            assert hasattr(packed_block_types, "elec_partial_charge")
            return

        def _ti(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        def _tf(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        pbt = packed_block_types
        elec_partial_charge = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms), dtype=torch.float32, device=self.device
        )
        elec_inter_repr_path_distance = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
            dtype=torch.int32,
            device=self.device,
        )
        elec_intra_repr_path_distance = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
            dtype=torch.int32,
            device=self.device,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            elec_partial_charge[i, : bt.n_atoms] = _tf(bt.elec_partial_charge)
            elec_inter_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                bt.elec_inter_repr_path_distance
            )
            elec_intra_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                bt.elec_intra_repr_path_distance
            )

        setattr(packed_block_types, "elec_partial_charge", elec_partial_charge)
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

    def setup_poses(self, poses: PoseStack):
        super(ElecEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return elec_pose_scores

    def get_rotamer_score_term_function(self):
        return elec_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        global_params = torch.tensor(
            [
                self.global_params.elec_sigmoidal_die_D,
                self.global_params.elec_sigmoidal_die_D0,
                self.global_params.elec_sigmoidal_die_S,
                self.global_params.elec_min_dis,
                self.global_params.elec_max_dis,
            ],
            dtype=torch.float32,
            device=pose_stack.device,
        )[None, :]

        return [
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.elec_partial_charge,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.elec_inter_repr_path_distance,
            pose_stack.packed_block_types.elec_intra_repr_path_distance,
            global_params,
        ]
