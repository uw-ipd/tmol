from enum import Enum

import torch
import numpy

from .._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, cached_annotation, store_annotation

from tmol.database import ParameterDatabase
from tmol.score.disulfide import DisulfideGlobalParams
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)


class _DisulfideParameter(Enum):
    """Describe the C++ parameter struct layout and D-residue transformation."""

    D_LOCATION = ("d_location", False)
    D_SCALE = ("d_scale", False)
    D_SHAPE = ("d_shape", False)
    A_LOG_A = ("a_logA", False)
    A_KAPPA = ("a_kappa", False)
    A_MU = ("a_mu", False)
    DSS_LOG_A1 = ("dss_logA1", False)
    DSS_KAPPA1 = ("dss_kappa1", False)
    DSS_MU1 = ("dss_mu1", True)
    DSS_LOG_A2 = ("dss_logA2", False)
    DSS_KAPPA2 = ("dss_kappa2", False)
    DSS_MU2 = ("dss_mu2", True)
    DCS_LOG_A1 = ("dcs_logA1", False)
    DCS_MU1 = ("dcs_mu1", True)
    DCS_KAPPA1 = ("dcs_kappa1", False)
    DCS_LOG_A2 = ("dcs_logA2", False)
    DCS_MU2 = ("dcs_mu2", True)
    DCS_KAPPA2 = ("dcs_kappa2", False)
    DCS_LOG_A3 = ("dcs_logA3", False)
    DCS_MU3 = ("dcs_mu3", True)
    DCS_KAPPA3 = ("dcs_kappa3", False)
    WT_DIH_SS = ("wt_dih_ss", False)
    WT_DIH_CS = ("wt_dih_cs", False)
    WT_ANG = ("wt_ang", False)
    WT_LEN = ("wt_len", False)
    SHIFT = ("shift", False)
    DSS_MIXED_LOG_A1 = ("dss_mixed_logA1", False)
    DSS_MIXED_KAPPA1 = ("dss_mixed_kappa1", False)
    DSS_MIXED_MU1 = ("dss_mixed_mu1", False)
    DSS_MIXED_LOG_A2 = ("dss_mixed_logA2", False)
    DSS_MIXED_KAPPA2 = ("dss_mixed_kappa2", False)
    DSS_MIXED_MU2 = ("dss_mixed_mu2", False)
    CHIRALITY = (None, True)

    def __init__(self, attribute: str | None, negated_for_d: bool):
        self.attribute = attribute
        self.negated_for_d = negated_for_d


_MIRRORED_PARAMETER_SIGNS = torch.tensor(
    [-1.0 if parameter.negated_for_d else 1.0 for parameter in _DisulfideParameter]
)


class DisulfideEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DisulfideEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = DisulfideGlobalParams.from_database(
            param_db.scoring.disulfide, device
        )
        self.device = device
        self._block_annotation_key = AnnotationKey.from_sources()
        self._packed_annotation_key = AnnotationKey.from_sources(
            settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "Disulfide"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._disulfide_creator

        return tmol.score.terms._disulfide_creator.DisulfideTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DisulfideEnergyTerm, self).setup_block_type(block_type)

        cached = cached_annotation(
            block_type, "_disulfide_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached

        disulfide_connections = numpy.array([], dtype=numpy.int32)
        if "dslf" in block_type.connection_to_cidx.keys():
            disulfide_connections = numpy.append(
                disulfide_connections, [block_type.connection_to_cidx["dslf"]]
            )

        setattr(block_type, "disulfide_connections", disulfide_connections)
        return store_annotation(
            block_type,
            "_disulfide_annotation",
            self._block_annotation_key,
            disulfide_connections,
            fields=("disulfide_connections",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DisulfideEnergyTerm, self).setup_packed_block_types(packed_block_types)

        cached = cached_annotation(
            packed_block_types,
            "_disulfide_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached

        disulfide_conns = torch.full(
            (packed_block_types.n_types, packed_block_types.max_n_conn),
            False,
            dtype=torch.bool,
            device=self.device,
        )

        block_connections = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]
        for i, connections in enumerate(block_connections):
            for conn in connections:
                disulfide_conns[i, conn] = True

        setattr(packed_block_types, "disulfide_conns", disulfide_conns)

        # a reflected residue contributes negated dihedrals; cos is even, so
        #    the same energy comes from negating the dihedral means instead,
        #    which needs no correction to the derivative
        is_mirrored = torch.tensor(
            [
                bt.properties.polymer.sidechain_chirality == "d"
                for bt in packed_block_types.active_block_types
            ],
            dtype=torch.bool,
            device=self.device,
        )
        setattr(packed_block_types, "disulfide_bt_is_mirrored", is_mirrored)
        return store_annotation(
            packed_block_types,
            "_disulfide_annotation",
            self._packed_annotation_key,
            (disulfide_conns, is_mirrored),
            fields=("disulfide_conns", "disulfide_bt_is_mirrored"),
            bindings=tuple(
                (block_type, "disulfide_connections")
                for block_type in packed_block_types.active_block_types
            ),
        )

    def setup_poses(self, poses: PoseStack):
        super(DisulfideEnergyTerm, self).setup_poses(poses)

    @property
    def score_only_in_no_grad(self):
        # Tiny CPU disulfide calls do not amortize the per-call mode check.
        return self.device.type == "cuda"

    def get_pose_score_term_function(self):
        from tmol.score.disulfide.potentials import disulfide_pose_scores

        return disulfide_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.disulfide.potentials import disulfide_rotamer_scores

        return disulfide_rotamer_scores

    def pose_score_term_is_invariant_zero(self, pose_stack):
        """Disulfide scoring is zero unless the pose has a connected dslf edge."""
        pbt = pose_stack.packed_block_types
        block_types = pose_stack.block_type_ind64
        real_blocks = block_types >= 0
        dslf_connections = pbt.disulfide_conns[block_types.clamp_min(0)]
        connected = pose_stack.inter_residue_connections[..., 0] >= 0
        return not bool(
            (dslf_connections & connected & real_blocks.unsqueeze(-1)).any()
        )

    def get_score_term_attributes(self, pose_stack):
        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        def parameter_value(parameter):
            if parameter is _DisulfideParameter.CHIRALITY:
                return torch.ones_like(self.global_params.shift)
            return getattr(self.global_params, parameter.attribute)

        global_params = torch.stack(
            _t(parameter_value(parameter) for parameter in _DisulfideParameter),
            dim=1,
        )
        pbt = pose_stack.packed_block_types

        # the kernel reads one parameter row per block type so that each half
        #    of a disulfide is scored with its own residue's parameters
        per_block_type = global_params.expand(pbt.n_types, -1).clone()
        per_block_type[pbt.disulfide_bt_is_mirrored, :] *= _MIRRORED_PARAMETER_SIGNS.to(
            per_block_type.device
        )
        return [
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            pbt.disulfide_conns,
            pbt.atom_downstream_of_conn,
            per_block_type,
        ]
