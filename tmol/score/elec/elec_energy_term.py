import torch
import numpy

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm

from tmol.database import ParameterDatabase
from tmol.score.common.stack_condense import tile_subset_indices
from tmol.score.elec.params import ElecParamResolver, ElecGlobalParams
from tmol.score.elec.elec_whole_pose_module import ElecWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor


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
        partial_charge = numpy.zeros((block_type.n_atoms,), dtype=numpy.float32)

        for i, atname in enumerate(block_type.atoms):
            if (block_type.name, atname.name) in self.param_resolver.partial_charges:
                partial_charge[i] = self.param_resolver.partial_charges[
                    (block_type.name, atname.name)
                ]

        # count pair representative logic:
        # decide whether or not two atoms i and j should have their interaction counted
        # on the basis of the number of chemical bonds that separate their respective
        # representative atoms r(i) and r(j).
        # We will create a tensor to measure the distance of all representative atoms to:
        # all connection atoms.
        # let rpd be the acronym for "representative_path_distance"
        # rpd[a,b] will be the number of chemical bonds between atom a and atom r(b). Then
        # we can answer how far apart the representative atoms are for atoms i and j on
        # residues k and l by computinging:
        # min(ci cj, rpd_i[ci, i] + rpd_j[cj, j] + sep(ci, cj))
        # over all pairs of connection atoms ci and cj on residues i and j
        # NOTE: the connection atoms must be their own representatives for this to work

        representative_mapping = numpy.arange(block_type.n_atoms, dtype=numpy.int32)
        if block_type.name in self.param_resolver.cp_reps_by_res:
            for outer, inner in self.param_resolver.cp_reps_by_res[
                block_type.name
            ].items():
                if (
                    outer not in block_type.atom_to_idx
                    or inner not in block_type.atom_to_idx
                ):
                    continue
                # note that the "inner" and "outer" atoms are flipped relative to
                # the natural interpretation in the file. That if one inner:outer
                # pair is "N": "1H" and another inner:outer pair is "N": "2H", one
                # would naturally conclude that 1H's representative is N and that
                # 2H's representative is also N. However, in actuallity, N's
                # representative will be 2H; N's representative starts out 1H, but
                # then it is overwritten when the 2H entry is parsed.
                #
                # In general, the approach is to use the further atom for the closer
                # atom so that more interactions are counted (because the closer atom
                # will interact with fewer other atoms; the further out something is,
                # the more other atoms will be at least 4 chemical bonds from it.
                # As long as all the atoms j that are listed as representatives for
                # a particular atom i are chemically bound to i, then one atom
                # overriding another as the representative will have no effect.
                representative_mapping[
                    block_type.atom_to_idx[inner]
                ] = block_type.atom_to_idx[outer]

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

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        return ElecWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_min_block_bondsep=pose_stack.min_block_bondsep,
            pose_stack_inter_block_bondsep=pose_stack.inter_block_bondsep,
            bt_n_atoms=pbt.n_atoms,
            bt_partial_charge=pbt.elec_partial_charge,
            bt_n_interblock_bonds=pbt.n_interblock_bonds,
            bt_atoms_forming_chemical_bonds=pbt.atoms_for_interblock_bonds,
            bt_inter_repr_path_distance=pbt.elec_inter_repr_path_distance,
            bt_intra_repr_path_distance=pbt.elec_intra_repr_path_distance,
            global_params=self.global_params,
        )
