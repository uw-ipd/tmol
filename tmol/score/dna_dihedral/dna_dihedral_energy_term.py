import numpy
import torch

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

from .params import (
    DnaDihedralParams,
    BASE_FOR_NAME3,
    TORSION_NAMES,
    TORSION_IND,
    SYN_MEAN,
)
from .potentials import (
    bi_bii_weight,
    blended_devsq,
    dihedral,
    pucker_weights,
    syn_weight,
    triple_bin_weights,
    wrap_degrees,
)

N_TORSION = len(TORSION_NAMES)
ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA = 0, 1, 2, 3, 4, 5
CHI = TORSION_IND["chi1"]
SUGAR_SLOTS = (DELTA, TORSION_IND["nu4"], TORSION_IND["nu0"], TORSION_IND["nu1"])


class DnaDihedralEnergyTerm(EnergyTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DnaDihedralEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.params = DnaDihedralParams.from_database(
            param_db.scoring.dna_dihedral, device
        )
        self.element_for_atom_type = {
            at.name: at.element for at in param_db.chemical.atom_types
        }
        self.device = device

    def _ring_atoms(self, block_type):
        """Ordered sugar ring, derived from the nu torsions rather than named.

        nu0 and nu1 each span four consecutive ring atoms offset by one, so
        together they give the whole cycle in order. The pucker slot arithmetic
        is defined relative to a cycle ending on the ring heteroatom, so rotate
        to put it last.
        """
        nu0 = block_type.torsion_to_uaids.get("nu0")
        nu1 = block_type.torsion_to_uaids.get("nu1")
        if nu0 is None or nu1 is None:
            return None
        cycle = [uaid[0] for uaid in nu0] + [nu1[-1][0]]
        if len(set(cycle)) != 5 or any(a < 0 for a in cycle):
            return None

        def element(atom_index):
            return self.element_for_atom_type[block_type.atoms[atom_index].atom_type]

        hetero = [i for i, a in enumerate(cycle) if element(a) != "C"]
        if len(hetero) != 1:
            return None
        k = hetero[0]
        return cycle[k + 1 :] + cycle[: k + 1]

    @classmethod
    def class_name(cls):
        return "DnaDihedral"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.dna_dihedral_creator

        return (
            tmol.score.terms.dna_dihedral_creator.DnaDihedralTermCreator.score_types()
        )

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DnaDihedralEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "dna_dihedral_params"):
            return

        base = BASE_FOR_NAME3.get(block_type.name3, -1)
        uaids = numpy.full((N_TORSION, 4, 3), -1, dtype=numpy.int32)
        ring = numpy.full((5,), -1, dtype=numpy.int32)
        if base >= 0:
            for i, name in enumerate(TORSION_NAMES):
                tor = block_type.torsion_to_uaids.get(name)
                if tor is not None:
                    uaids[i] = numpy.array(tor, dtype=numpy.int32)
            ring_atoms = self._ring_atoms(block_type)
            if ring_atoms is not None:
                ring[:] = ring_atoms
            # backbone torsions may be absent at a terminus and are masked at
            # scoring time; the sugar and glycosidic ones must all be present
            required = [DELTA, CHI] + [TORSION_IND[n] for n in ("nu0", "nu1", "nu4")]
            if (ring < 0).any() or (uaids[required, :, 0] < 0).any():
                base = -1

        down = block_type.connection_to_cidx.get("down", -1)
        setattr(
            block_type,
            "dna_dihedral_params",
            dict(base=base, uaids=uaids, ring=ring, down=down),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DnaDihedralEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "dna_dihedral_base"):
            return

        bts = packed_block_types.active_block_types
        stack = lambda key: numpy.stack(  # noqa: E731
            [bt.dna_dihedral_params[key] for bt in bts]
        )

        def to_t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        setattr(
            packed_block_types,
            "dna_dihedral_base",
            to_t(numpy.array([bt.dna_dihedral_params["base"] for bt in bts])),
        )
        setattr(packed_block_types, "dna_dihedral_uaids", to_t(stack("uaids")))
        setattr(packed_block_types, "dna_dihedral_ring", to_t(stack("ring")))
        setattr(
            packed_block_types,
            "dna_dihedral_down",
            to_t(numpy.array([bt.dna_dihedral_params["down"] for bt in bts])),
        )

    def setup_poses(self, poses: PoseStack):
        super(DnaDihedralEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return eval_dna_dihedral_for_pose

    def get_rotamer_score_term_function(self):
        raise NotImplementedError("dna_torsion does not support rotamer scoring")

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types
        p = self.params
        bt = pose_stack.block_type_ind64
        has_dna = bool(
            (pbt.dna_dihedral_base.to(torch.int64)[bt.clamp_min(0)] >= 0)[bt >= 0].any()
        )
        return [
            has_dna,
            pbt.dna_dihedral_base,
            pbt.dna_dihedral_uaids,
            pbt.dna_dihedral_ring,
            pbt.dna_dihedral_down,
            pbt.atom_downstream_of_conn,
            pose_stack.block_coord_offset,
            pose_stack.inter_residue_connections,
            p.backbone_means,
            p.backbone_sdev,
            p.sugar_means,
            p.chi_means,
            p.sdev_sugar,
            p.sdev_chi,
            p.weight_bb,
            p.weight_chi,
            p.weight_sugar,
            p.pucker_temperature,
            p.bin_blend_sdev,
        ]


def _resolve_uaids(
    uaids, block_type, block_coord_offset, inter_res_conn, atom_downstream_of_conn
):
    """Global (flattened) atom indices for unresolved atom ids; -1 if absent.

    A uaid is (atom, connection, bonds-from-connection): connection -1 means the
    atom is local, otherwise it is reached through a residue connection.
    """
    n_poses, max_n_blocks = block_type.shape
    trailing = uaids.shape[2:-1]  # (n_torsion, 4)
    shape = (n_poses, max_n_blocks) + tuple(trailing)

    atom = uaids[..., 0].to(torch.int64)
    conn = uaids[..., 1].to(torch.int64)
    sep = uaids[..., 2].to(torch.int64)
    is_inter = conn >= 0

    self_block = (
        torch.arange(max_n_blocks, device=uaids.device)
        .view(1, max_n_blocks, *([1] * len(trailing)))
        .expand_as(atom)
    )

    # partner block/connection for every inter-residue uaid
    conn_idx = conn.clamp_min(0).reshape(n_poses, max_n_blocks, -1)
    other_block = torch.gather(inter_res_conn[..., 0].to(torch.int64), 2, conn_idx)
    other_conn = torch.gather(inter_res_conn[..., 1].to(torch.int64), 2, conn_idx)
    other_block = other_block.reshape(atom.shape)
    other_conn = other_conn.reshape(atom.shape)

    target_block = torch.where(is_inter, other_block, self_block)
    ok = (atom >= 0) & (~is_inter | (other_block >= 0))
    target_block_safe = target_block.clamp_min(0)

    target_bt = torch.gather(
        block_type.to(torch.int64), 1, target_block_safe.reshape(n_poses, -1)
    ).reshape(*shape)
    ok = ok & (target_bt >= 0)

    local_inter = atom_downstream_of_conn.to(torch.int64)[
        target_bt.clamp_min(0), other_conn.clamp_min(0), sep.clamp_min(0)
    ]
    local = torch.where(is_inter, local_inter, atom)
    ok = ok & (local >= 0)

    offset = torch.gather(
        block_coord_offset.to(torch.int64), 1, target_block_safe.reshape(n_poses, -1)
    ).reshape(*shape)
    pose = (
        torch.arange(n_poses, device=uaids.device)
        .view(n_poses, *([1] * (atom.dim() - 1)))
        .expand_as(atom)
    )
    return pose, offset + local, ok


def eval_dna_dihedral_for_pose(
    # common args
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    # term args
    has_dna,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    weight_bb,
    weight_chi,
    weight_sugar,
    pucker_temperature,
    bin_blend_sdev,
    output_block_pair_energies: bool,
):
    if not has_dna:
        n_poses, max_n_blocks = block_coord_offset.shape
        zero = torch.zeros(
            (n_poses, max_n_blocks), dtype=rot_coords.dtype, device=rot_coords.device
        )
        return _finish(zero, output_block_pair_energies)

    e_bb, e_chi, e_sugar, is_dna = dna_dihedral_subterms(
        rot_coords,
        block_type_ind_for_rot,
        bt_base,
        bt_uaids,
        bt_ring,
        bt_down,
        atom_downstream_of_conn,
        block_coord_offset,
        inter_residue_connections,
        backbone_means,
        backbone_sdev,
        sugar_means,
        chi_means,
        sdev_sugar,
        sdev_chi,
        pucker_temperature,
        bin_blend_sdev,
    )
    score = weight_bb * e_bb + weight_chi * e_chi + weight_sugar * e_sugar
    return _finish(
        torch.where(is_dna, score, torch.zeros_like(score)), output_block_pair_energies
    )


def dna_dihedral_subterms(
    rot_coords,
    block_type_ind_for_rot,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    pucker_temperature,
    bin_blend_sdev,
):
    """Per-block (bb, chi, sugar) energies and the DNA mask, unweighted."""
    dtype = rot_coords.dtype
    n_poses, max_n_blocks = block_coord_offset.shape
    max_n_atoms = rot_coords.shape[0] // n_poses

    block_type = block_type_ind_for_rot.view(n_poses, max_n_blocks).to(torch.int64)
    real = block_type >= 0
    bt_safe = block_type.clamp_min(0)
    base = torch.where(real, bt_base.to(torch.int64)[bt_safe], -1)
    is_dna = base >= 0

    score = torch.zeros((n_poses, max_n_blocks), dtype=dtype, device=rot_coords.device)

    def gather_coords(pose, index, ok):
        flat = (pose * max_n_atoms + index).clamp_min(0)
        xyz = rot_coords[flat.reshape(-1)].reshape(*index.shape, 3)
        return torch.where(ok.unsqueeze(-1), xyz, torch.zeros_like(xyz))

    # --- torsions -----------------------------------------------------------
    pose, index, ok = _resolve_uaids(
        bt_uaids[bt_safe],
        block_type,
        block_coord_offset,
        inter_residue_connections,
        atom_downstream_of_conn,
    )
    xyz = gather_coords(pose, index, ok)
    tor_ok = ok.all(-1) & is_dna.unsqueeze(-1)
    tors = dihedral(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :], xyz[..., 3, :])
    tors = torch.where(tor_ok, tors, torch.zeros_like(tors)) % 360.0

    # --- pucker -------------------------------------------------------------
    ring_local = bt_ring.to(torch.int64)[bt_safe]
    ring_pose = torch.arange(n_poses, device=rot_coords.device).view(n_poses, 1, 1)
    ring_index = block_coord_offset.to(torch.int64).unsqueeze(-1) + ring_local
    ring_ok = (ring_local >= 0) & is_dna.unsqueeze(-1)
    ring_xyz = gather_coords(ring_pose.expand_as(ring_local), ring_index, ring_ok)
    pucker = pucker_weights(ring_xyz, pucker_temperature)
    pucker = torch.where(is_dna.unsqueeze(-1), pucker, torch.zeros_like(pucker)).to(
        dtype
    )

    # --- backbone -----------------------------------------------------------
    means = backbone_means.to(dtype)
    e_bb = torch.zeros_like(score)

    for tor in (ALPHA, GAMMA):
        ang = tors[..., tor]
        w = triple_bin_weights(ang, means[tor, :3], bin_blend_sdev)
        e_bb = e_bb + torch.where(
            tor_ok[..., tor],
            blended_devsq(ang, means[tor, :3], w) / backbone_sdev[tor] ** 2,
            torch.zeros_like(ang),
        )

    w_bi = bi_bii_weight(tors[..., EPSILON], tors[..., ZETA])
    both = tor_ok[..., EPSILON] & tor_ok[..., ZETA]
    for tor in (EPSILON, ZETA):
        ang = tors[..., tor]
        w = torch.stack([w_bi, 1.0 - w_bi], dim=-1)
        e_bb = e_bb + torch.where(
            both,
            blended_devsq(ang, means[tor, :2], w) / backbone_sdev[tor] ** 2,
            torch.zeros_like(ang),
        )

    # beta is binned on the previous residue's BI/BII state;
    # fall back to BI mean when there is no preceding DNA residue
    down = bt_down.to(torch.int64)[bt_safe]
    prev_block = torch.gather(
        inter_residue_connections[..., 0].to(torch.int64),
        2,
        down.clamp_min(0)[..., None],
    ).squeeze(-1)
    has_prev = (down >= 0) & (prev_block >= 0)
    prev_safe = prev_block.clamp_min(0)
    prev_bi = torch.gather(w_bi, 1, prev_safe)
    prev_ok = has_prev & torch.gather(both, 1, prev_safe)
    w_beta = torch.where(prev_ok, prev_bi, torch.ones_like(prev_bi))
    ang = tors[..., BETA]
    e_bb = e_bb + torch.where(
        tor_ok[..., BETA],
        blended_devsq(ang, means[BETA, :2], torch.stack([w_beta, 1.0 - w_beta], dim=-1))
        / backbone_sdev[BETA] ** 2,
        torch.zeros_like(ang),
    )

    # --- chi ----------------------------------------------------------------
    chi = tors[..., CHI]
    w_syn = syn_weight(chi).unsqueeze(-1)
    dev_pucker = wrap_degrees(
        chi.unsqueeze(-1) - chi_means.to(dtype)[base.clamp_min(0)]
    )
    dev_syn = wrap_degrees(chi - SYN_MEAN).unsqueeze(-1)
    e_chi = (pucker * ((1.0 - w_syn) * dev_pucker**2 + w_syn * dev_syn**2)).sum(
        -1
    ) / sdev_chi**2
    e_chi = torch.where(tor_ok[..., CHI], e_chi, torch.zeros_like(e_chi))

    # --- sugar --------------------------------------------------------------
    e_sugar = torch.zeros_like(score)
    for slot, tor in enumerate(SUGAR_SLOTS):
        dev = wrap_degrees(
            tors[..., tor].unsqueeze(-1) - sugar_means.to(dtype)[:, slot]
        )
        e_sugar = e_sugar + torch.where(
            tor_ok[..., tor],
            (pucker * dev**2).sum(-1),
            torch.zeros_like(e_sugar),
        )
    e_sugar = e_sugar / sdev_sugar**2

    return e_bb, e_chi, e_sugar, is_dna


def _finish(score, output_block_pair_energies):
    if output_block_pair_energies:
        return torch.diag_embed(score).unsqueeze(0), None
    return torch.sum(score, dim=1).unsqueeze(0), None
