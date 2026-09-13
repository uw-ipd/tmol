import torch

from ._hbond_dependent_term import HBondDependentTerm
from ._params import CompactedHBondDatabase
from .._atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)


class HBondEnergyTerm(AtomTypeDependentTerm, HBondDependentTerm):
    tile_size: int = 32
    hb_param_db: CompactedHBondDatabase

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(HBondEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.tile_size = HBondEnergyTerm.tile_size
        self.hb_param_db = CompactedHBondDatabase.from_database(
            param_db.chemical, param_db.scoring.hbond, device
        )
        # Cache of parameter tables converted to whichever coord dtype is used
        # in scoring. Avoids a per-forward .to(coords_dtype) on three tensors.
        # Keyed by dtype; stored value is a (pair_param, pair_poly, global_param) tuple.
        # pair_poly_table is always kept at float64 — the C++ kernel templates
        # HBondPolynomials on double independently of the Real coord dtype.
        self._param_tables_by_dtype: dict = {}

    def _param_tables(self, coords_dtype):
        tables = self._param_tables_by_dtype.get(coords_dtype)
        if tables is None:
            tables = (
                self.hb_param_db.pair_param_table.to(coords_dtype),
                self.hb_param_db.pair_poly_table.to(torch.float64),
                self.hb_param_db.global_param_table.to(coords_dtype),
            )
            self._param_tables_by_dtype[coords_dtype] = tables
        return tables

    @classmethod
    def class_name(cls):
        return "HBond"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._hbond_creator

        return tmol.score.terms._hbond_creator.HBondTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondEnergyTerm, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: PoseStack):
        super(HBondEnergyTerm, self).setup_poses(poses)
        # Count actual residues once, before scoring or CUDA Graph capture.
        # Padded stack size alone can overestimate work in jagged batches.
        poses._hbond_allow_split_pairs = (
            poses.device.type == "cuda"
            and poses.block_type_ind.numel() >= 4096
            and int((poses.block_type_ind >= 0).sum()) >= 2048
        )

    def pose_score_hbond(self, *args):
        from tmol.score.hbond.potentials import (
            hbond_pose_scores,
            gen_hbond_bases,
        )

        common_args = args[:-3]
        pose_stack = args[-3]
        block_pair_scoring = args[-2]
        shared_block_neighbors = args[-1]
        allow_split_pairs = getattr(pose_stack, "_hbond_allow_split_pairs", False)
        coords_dtype = common_args[0].dtype
        pair_param_table, pair_poly_table, global_param_table = self._param_tables(
            coords_dtype
        )

        # Derived atom coords do not need gradients - gradients for hbond
        # energies flow through derived_atom_inds back to the source atoms
        # inside the pairwise kernel directly.
        with torch.no_grad():
            derived_coords, derived_atom_inds = gen_hbond_bases(
                common_args[0],
                common_args[1],
                common_args[3],
                common_args[4],
                common_args[5],
                common_args[6],
                common_args[7],
                pose_stack.inter_residue_connections,
                pose_stack.packed_block_types.n_atoms,
                pose_stack.packed_block_types.n_conn,
                pose_stack.packed_block_types.conn_atom,
                pose_stack.packed_block_types.n_all_bonds,
                pose_stack.packed_block_types.all_bonds,
                pose_stack.packed_block_types.atom_all_bond_ranges,
                pose_stack.packed_block_types.hbpbt_params.tile_n_donH,
                pose_stack.packed_block_types.hbpbt_params.tile_n_acc,
                pose_stack.packed_block_types.hbpbt_params.tile_donH_inds,
                pose_stack.packed_block_types.hbpbt_params.tile_acc_inds,
                pose_stack.packed_block_types.hbpbt_params.tile_acceptor_hybridization,
                pose_stack.packed_block_types.hbpbt_params.is_hydrogen,
            )

        return hbond_pose_scores(
            *common_args,
            pose_stack.inter_residue_connections,
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.n_all_bonds,
            pose_stack.packed_block_types.all_bonds,
            pose_stack.packed_block_types.atom_all_bond_ranges,
            pose_stack.packed_block_types.bond_separation,
            pose_stack.packed_block_types.hbpbt_params.tile_n_donH,
            pose_stack.packed_block_types.hbpbt_params.tile_n_acc,
            pose_stack.packed_block_types.hbpbt_params.tile_donH_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_acc_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_donorH_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_hybridization,
            pose_stack.packed_block_types.hbpbt_params.is_hydrogen,
            pair_param_table,
            pair_poly_table,
            global_param_table,
            derived_coords,
            derived_atom_inds,
            block_pair_scoring,
            shared_block_neighbors,
            allow_split_pairs,
        )

    def rotamer_score_hbond(self, *args):
        from tmol.score.hbond.potentials import (
            hbond_rotamer_scores,
            hbond_rotamer_scores_shared,
            gen_hbond_bases,
        )

        common_args = args[:-3]
        pose_stack = args[-3]
        block_pair_scoring = args[-2]
        shared_dispatch_indices = args[-1]
        coords_dtype = common_args[0].dtype
        pair_param_table, pair_poly_table, global_param_table = self._param_tables(
            coords_dtype
        )

        with torch.no_grad():
            derived_coords, derived_atom_inds = gen_hbond_bases(
                common_args[0],
                common_args[1],
                common_args[3],
                common_args[4],
                common_args[5],
                common_args[6],
                common_args[7],
                pose_stack.inter_residue_connections,
                pose_stack.packed_block_types.n_atoms,
                pose_stack.packed_block_types.n_conn,
                pose_stack.packed_block_types.conn_atom,
                pose_stack.packed_block_types.n_all_bonds,
                pose_stack.packed_block_types.all_bonds,
                pose_stack.packed_block_types.atom_all_bond_ranges,
                pose_stack.packed_block_types.hbpbt_params.tile_n_donH,
                pose_stack.packed_block_types.hbpbt_params.tile_n_acc,
                pose_stack.packed_block_types.hbpbt_params.tile_donH_inds,
                pose_stack.packed_block_types.hbpbt_params.tile_acc_inds,
                pose_stack.packed_block_types.hbpbt_params.tile_acceptor_hybridization,
                pose_stack.packed_block_types.hbpbt_params.is_hydrogen,
            )

        score_op = (
            hbond_rotamer_scores_shared
            if shared_dispatch_indices.numel() != 0
            else hbond_rotamer_scores
        )
        score_args = (
            *common_args,
            pose_stack.inter_residue_connections,
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.n_all_bonds,
            pose_stack.packed_block_types.all_bonds,
            pose_stack.packed_block_types.atom_all_bond_ranges,
            pose_stack.packed_block_types.bond_separation,
            pose_stack.packed_block_types.hbpbt_params.tile_n_donH,
            pose_stack.packed_block_types.hbpbt_params.tile_n_acc,
            pose_stack.packed_block_types.hbpbt_params.tile_donH_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_acc_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_donorH_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_hybridization,
            pose_stack.packed_block_types.hbpbt_params.is_hydrogen,
            pair_param_table,
            pair_poly_table,
            global_param_table,
            derived_coords,
            derived_atom_inds,
            block_pair_scoring,
        )
        if shared_dispatch_indices.numel() != 0:
            score_args += (shared_dispatch_indices,)
        return score_op(*score_args)

    @property
    def score_only_in_no_grad(self):
        # CPU compilers can round score-only and derivative paths differently
        # (including float64 on ARM). Preserve tracked-input values there.
        return self.device.type == "cuda"

    def get_pose_score_term_function(self):
        return self.pose_score_hbond

    def get_rotamer_score_term_function(self):
        return self.rotamer_score_hbond

    def get_block_neighbor_cutoff(self):
        return 5.5

    def accepts_shared_rotamer_dispatch(self):
        return True

    def rotamer_dispatch_key(self):
        return "sphere_overlap"

    def get_score_term_attributes(self, pose_stack: PoseStack):
        return [pose_stack]
