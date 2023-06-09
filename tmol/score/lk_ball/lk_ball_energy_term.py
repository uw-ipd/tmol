import numpy
import torch

from .params import LKBallBlockTypeParams, LKBallPackedBlockTypesParams
from .lk_ball_whole_pose_module import LKBallWholePoseScoringModule
from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..hbond.hbond_dependent_term import HBondDependentTerm
from ..ljlk.params import LJLKGlobalParams, LJLKParamResolver
from tmol.database import ParameterDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.score.common.stack_condense import arg_tile_subset_indices


class LKBallEnergyTerm(AtomTypeDependentTerm, HBondDependentTerm):
    tile_size: int = HBondDependentTerm.tile_size
    ljlk_global_params: LJLKGlobalParams
    ljlk_param_resolver: LJLKParamResolver

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(LKBallEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ljlk_param_resolver = LJLKParamResolver.from_database(
            param_db.chemical, param_db.scoring.ljlk, device=device
        )
        self.tile_size = LKBallEnergyTerm.tile_size

    @classmethod
    def score_types(cls):
        import tmol.score.terms.lk_ball_creator

        return tmol.score.terms.lk_ball_creator.LKBallTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(LKBallEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "lk_ball_params"):
            return

        # we are going to order the data needed for score evaluation around the
        # idea that, first, we will bin all the atoms into groups of "tile_size"
        # and examine tile_size atoms from one residue against tile_size atoms
        # of another residue.
        # within each tile, the data is further structured as follows:
        # polar atoms -- i.e. those with attached waters -- are going to be listed
        # first and all remaining heavy atoms will be listed second.
        # Each tile will list the number of polar atoms (heavy atoms with waters
        # attached) and the number of "occluder" atoms (heavy atoms with waters
        # attached and heavy atoms without waters attached) and then
        # the tile indices of those atoms (in a single array with the first
        # n_polar atoms representing the heavy atoms with attached waters).
        # The lk-ball properties needed to evaluate the energy will be stored
        # also in polars-before-non-polars order.

        hbbt_params = block_type.hbbt_params
        n_tiles = hbbt_params.tile_donH_inds.shape[0]

        atom_is_polar = numpy.full((block_type.n_atoms,), False, dtype=bool)
        atom_is_polar[hbbt_params.don_hvy_inds] = True
        atom_is_polar[hbbt_params.acc_inds] = True
        polar_inds = numpy.nonzero(atom_is_polar)[0].astype(numpy.int32)

        tile_size = LKBallEnergyTerm.tile_size
        tiled_polar_orig_inds, tile_n_polar = arg_tile_subset_indices(
            polar_inds, tile_size, block_type.n_atoms
        )
        tiled_polar_orig_inds = tiled_polar_orig_inds.reshape(n_tiles, tile_size)

        is_tiled_polar = tiled_polar_orig_inds != -1
        tiled_polars = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
        tiled_polars[is_tiled_polar] = polar_inds

        # ASSUMPTION! either h or hvy; change when VRTS are added!
        # Grace: lk parameters for VRTs should not affect score even if included,
        # it would just be slightly inefficient
        atom_is_heavy = numpy.invert(hbbt_params.is_hydrogen == 1)

        # "apolar" here means "does not build waters"; e.g. proline's N would be
        # considered apolar.
        atom_is_heavy_apolar = numpy.logical_and(
            atom_is_heavy, numpy.invert(atom_is_polar)
        )
        heavy_apolar_inds = numpy.nonzero(atom_is_heavy_apolar)[0].astype(numpy.int32)

        tiled_heavy_apolar_orig_inds, tile_n_apolar = arg_tile_subset_indices(
            heavy_apolar_inds, tile_size, block_type.n_atoms
        )
        tiled_heavy_apolar_orig_inds = tiled_heavy_apolar_orig_inds.reshape(
            n_tiles, tile_size
        )
        is_tiled_heavy_apolar = tiled_heavy_apolar_orig_inds != -1
        tiled_heavy_apolar = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
        tiled_heavy_apolar[is_tiled_heavy_apolar] = heavy_apolar_inds

        # now lets combine the heavy polar indices and the heavy non-polar indices
        tiled_pols_and_occs = numpy.copy(tiled_polars)
        tile_n_occ = numpy.copy(tile_n_polar)
        for i in range(n_tiles):
            tile_n_occ[i] = tile_n_polar[i] + tile_n_apolar[i]
            r = slice(tile_n_polar[i], tile_n_occ[i])
            tiled_pols_and_occs[i, r] = tiled_heavy_apolar[i, : tile_n_apolar[i]]
        is_pol_or_occ = tiled_pols_and_occs != -1

        # ok, now let's collect the properties of the atoms in this block
        # needed for LKBallTypeParams (see properties/params.hh)
        assert hasattr(block_type, "atom_types")
        at = block_type.atom_types
        type_params = self.ljlk_param_resolver.type_params.to(torch.device("cpu"))
        bt_lj_radius = type_params.lj_radius[at].numpy()
        bt_lk_dgfree = type_params.lk_dgfree[at].numpy()
        bt_lk_lambda = type_params.lk_lambda[at].numpy()
        bt_lk_volume = type_params.lk_volume[at].numpy()
        bt_is_donor = type_params.is_donor[at].numpy()
        bt_is_hydroxyl = type_params.is_hydroxyl[at].numpy()
        bt_is_polarh = type_params.is_polarh[at].numpy()
        bt_is_acceptor = type_params.is_acceptor[at].numpy()

        bt_lk_ball_at_params = numpy.stack(
            (
                bt_lj_radius,
                bt_lk_dgfree,
                bt_lk_lambda,
                bt_lk_volume,
                bt_is_donor,
                bt_is_hydroxyl,
                bt_is_polarh,
                bt_is_acceptor,
            ),
            axis=1,
        )
        tiled_bt_lk_ball_at_params = numpy.zeros(
            (n_tiles, tile_size, 8), dtype=numpy.float32
        )
        tiled_bt_lk_ball_at_params[is_pol_or_occ] = bt_lk_ball_at_params[
            tiled_pols_and_occs[is_pol_or_occ]
        ]

        bt_lk_ball_params = LKBallBlockTypeParams(
            tile_n_polar_atoms=tile_n_polar,
            tile_n_occluder_atoms=tile_n_occ,
            tile_pol_occ_inds=tiled_pols_and_occs,
            tile_lk_ball_params=tiled_bt_lk_ball_at_params,
        )
        setattr(block_type, "lk_ball_params", bt_lk_ball_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LKBallEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "lk_ball_params"):
            return
        n_types = packed_block_types.n_types
        n_tiles = packed_block_types.hbpbt_params.tile_donH_inds.shape[1]
        tile_size = LKBallEnergyTerm.tile_size

        tile_n_polar_atoms = numpy.full((n_types, n_tiles), 0, dtype=numpy.int32)
        tile_n_occluder_atoms = numpy.full((n_types, n_tiles), 0, dtype=numpy.int32)
        tile_pol_occ_inds = numpy.full(
            (n_types, n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_lk_ball_params = numpy.full(
            (n_types, n_tiles, tile_size, 8), 0, dtype=numpy.float32
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            i_lkbp = bt.lk_ball_params
            i_n_tiles = i_lkbp.tile_n_polar_atoms.shape[0]

            tile_n_polar_atoms[i, :i_n_tiles] = i_lkbp.tile_n_polar_atoms
            tile_n_occluder_atoms[i, :i_n_tiles] = i_lkbp.tile_n_occluder_atoms
            tile_pol_occ_inds[i, :i_n_tiles] = i_lkbp.tile_pol_occ_inds
            tile_lk_ball_params[i, :i_n_tiles] = i_lkbp.tile_lk_ball_params

        def _t(t):
            return torch.tensor(t, device=packed_block_types.device)

        lk_ball_params = LKBallPackedBlockTypesParams(
            tile_n_polar_atoms=_t(tile_n_polar_atoms),
            tile_n_occluder_atoms=_t(tile_n_occluder_atoms),
            tile_pol_occ_inds=_t(tile_pol_occ_inds),
            tile_lk_ball_params=_t(tile_lk_ball_params),
        )
        setattr(packed_block_types, "lk_ball_params", lk_ball_params)

    def setup_poses(self, pose_stack: PoseStack):
        super(LKBallEnergyTerm, self).setup_poses(pose_stack)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        ljlk_global_params = self.ljlk_param_resolver.global_params

        return LKBallWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_type=pose_stack.block_type_ind,
            pose_stack_inter_residue_connections=pose_stack.inter_residue_connections,
            pose_stack_min_bond_separation=pose_stack.min_block_bondsep,
            pose_stack_inter_block_bondsep=pose_stack.inter_block_bondsep,
            bt_n_atoms=pbt.n_atoms,
            bt_n_interblock_bonds=pbt.n_conn,
            bt_atoms_forming_chemical_bonds=pbt.conn_atom,
            bt_n_all_bonds=pbt.n_all_bonds,
            bt_all_bonds=pbt.all_bonds,
            bt_atom_all_bond_ranges=pbt.atom_all_bond_ranges,
            bt_tile_n_donH=pbt.hbpbt_params.tile_n_donH,
            bt_tile_n_acc=pbt.hbpbt_params.tile_n_acc,
            bt_tile_donH_inds=pbt.hbpbt_params.tile_donH_inds,
            bt_tile_don_hvy_inds=pbt.hbpbt_params.tile_donH_hvy_inds,
            bt_tile_which_donH_for_hvy=pbt.hbpbt_params.tile_which_donH_of_donH_hvy,
            bt_tile_acc_inds=pbt.hbpbt_params.tile_acc_inds,
            bt_tile_hybridization=pbt.hbpbt_params.tile_acceptor_hybridization,
            bt_tile_acc_n_attached_H=pbt.hbpbt_params.tile_acceptor_n_attached_H,
            bt_atom_is_hydrogen=pbt.hbpbt_params.is_hydrogen,
            bt_tile_n_polar_atoms=pbt.lk_ball_params.tile_n_polar_atoms,
            bt_tile_n_occluder_atoms=pbt.lk_ball_params.tile_n_occluder_atoms,
            bt_tile_pol_occ_inds=pbt.lk_ball_params.tile_pol_occ_inds,
            bt_tile_lk_ball_params=pbt.lk_ball_params.tile_lk_ball_params,
            bt_path_distance=pbt.bond_separation,
            lk_ball_global_params=self.stack_lk_ball_global_params(),
            water_gen_global_params=self.stack_lk_ball_water_gen_global_params(),
            sp2_water_tors=ljlk_global_params.lkb_water_tors_sp2,
            sp3_water_tors=ljlk_global_params.lkb_water_tors_sp3,
            ring_water_tors=ljlk_global_params.lkb_water_tors_ring,
        )

    def _tfloat(self, ts):
        return tuple(map(lambda t: t.to(torch.float), ts))

    def stack_lk_ball_global_params(self):
        return torch.stack(
            self._tfloat(
                [
                    self.ljlk_param_resolver.global_params.lj_hbond_dis,
                    self.ljlk_param_resolver.global_params.lj_hbond_OH_donor_dis,
                    self.ljlk_param_resolver.global_params.lj_hbond_hdis,
                    self.ljlk_param_resolver.global_params.lkb_water_dist,
                    self.ljlk_param_resolver.global_params.max_dis,
                ]
            ),
            dim=1,
        )

    def stack_lk_ball_water_gen_global_params(self):
        return torch.stack(
            self._tfloat(
                [
                    self.ljlk_param_resolver.global_params.lkb_water_dist,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_sp2,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_sp3,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_ring,
                ]
            ),
            dim=1,
        )
