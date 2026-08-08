import numpy
import torch

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

from .params import (
    NaTorsionParams,
    block_type_params,
    polymer_index,
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


class NaTorsionEnergyTerm(EnergyTerm):
    """DNA/RNA torsion, coupling, sugar, and rotamer-well preference term."""

    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(NaTorsionEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.params = NaTorsionParams.from_database(param_db.scoring.na_torsion, device)
        self.element_for_atom_type = {
            at.name: at.element for at in param_db.chemical.atom_types
        }
        self.device = device

    @classmethod
    def class_name(cls):
        return "NaTorsion"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.na_torsion_creator

        return tmol.score.terms.na_torsion_creator.NaTorsionTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(NaTorsionEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "na_torsion_params"):
            return
        setattr(
            block_type,
            "na_torsion_params",
            block_type_params(block_type, self.element_for_atom_type),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(NaTorsionEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "na_torsion_base"):
            return

        bts = packed_block_types.active_block_types
        stack = lambda key: numpy.stack(  # noqa: E731
            [bt.na_torsion_params[key] for bt in bts]
        )

        def to_t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        setattr(
            packed_block_types,
            "na_torsion_base",
            to_t(numpy.array([bt.na_torsion_params["base"] for bt in bts])),
        )
        setattr(packed_block_types, "na_torsion_uaids", to_t(stack("uaids")))
        setattr(packed_block_types, "na_torsion_ring", to_t(stack("ring")))
        setattr(
            packed_block_types,
            "na_torsion_down",
            to_t(numpy.array([bt.na_torsion_params["down"] for bt in bts])),
        )

    def setup_poses(self, poses: PoseStack):
        super(NaTorsionEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return eval_na_torsion_for_pose

    def get_rotamer_score_term_function(self):
        return eval_na_torsion_for_rotamers

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types
        p = self.params
        bt = pose_stack.block_type_ind64
        has_na = bool(
            (pbt.na_torsion_base.to(torch.int64)[bt.clamp_min(0)] >= 0)[bt >= 0].any()
        )
        return [
            has_na,
            *self.subterm_attributes(pose_stack),
            p.weight_bb,
            p.weight_chi,
            p.weight_sugar,
        ]

    def subterm_attributes(self, pose_stack):
        """Arguments of na_torsion_subterms, which the eval functions take
        first and then follow with the harmonic subterm weights."""
        pbt = pose_stack.packed_block_types
        p = self.params
        return [
            pbt.na_torsion_base,
            pbt.na_torsion_uaids,
            pbt.na_torsion_ring,
            pbt.na_torsion_down,
            pbt.atom_downstream_of_conn,
            pose_stack.block_coord_offset,
            pose_stack.inter_residue_connections,
            p.backbone_means,
            p.backbone_sdev,
            p.sugar_means,
            p.chi_means,
            p.sdev_sugar,
            p.sdev_chi,
            p.well_pucker,
            p.well_alpha_gamma,
            p.well_bibii_pucker,
            p.well_alphanext_bibii,
            p.well_chi_syn,
            p.is_north,
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
    # an inter-residue uaid carries atom == -1; its atom index comes from the
    # partner block below
    ok = torch.where(is_inter, other_block >= 0, atom >= 0)
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


def eval_na_torsion_for_pose(
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
    has_na,
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
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
    weight_bb,
    weight_chi,
    weight_sugar,
    output_block_pair_energies: bool,
):
    if not has_na:
        n_poses, max_n_blocks = block_coord_offset.shape
        zero = torch.zeros(
            (n_poses, max_n_blocks), dtype=rot_coords.dtype, device=rot_coords.device
        )
        return _finish((zero, zero), output_block_pair_energies)

    e_bb, e_chi, e_sugar, e_well, is_na, base = na_torsion_subterms(
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
        well_pucker,
        well_alpha_gamma,
        well_bibii_pucker,
        well_alphanext_bibii,
        well_chi_syn,
        is_north,
        pucker_temperature,
        bin_blend_sdev,
    )
    poly = polymer_index(base)
    harmonic = (
        weight_bb[poly] * e_bb + weight_chi[poly] * e_chi + weight_sugar[poly] * e_sugar
    )
    well = e_well
    zero = torch.zeros_like(harmonic)
    return _finish(
        (torch.where(is_na, harmonic, zero), torch.where(is_na, well, zero)),
        output_block_pair_energies,
    )


def na_torsion_subterms(
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
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
):
    """Per-block (bb, chi, sugar, well) energies, the mask, and the base index.

    The base index carries the polymer, which the caller needs in order to pick
    the per-polymer subterm weights.
    """
    n_poses, max_n_blocks = block_coord_offset.shape
    max_n_atoms = rot_coords.shape[0] // n_poses

    block_type = block_type_ind_for_rot.view(n_poses, max_n_blocks).to(torch.int64)
    real = block_type >= 0
    bt_safe = block_type.clamp_min(0)
    base = torch.where(real, bt_base.to(torch.int64)[bt_safe], -1)
    is_na = base >= 0

    def gather_coords(pose, index, ok):
        flat = (pose * max_n_atoms + index).clamp_min(0)
        xyz = rot_coords[flat.reshape(-1)].reshape(*index.shape, 3)
        return torch.where(ok.unsqueeze(-1), xyz, torch.zeros_like(xyz))

    pose, index, ok = _resolve_uaids(
        bt_uaids[bt_safe],
        block_type,
        block_coord_offset,
        inter_residue_connections,
        atom_downstream_of_conn,
    )
    xyz = gather_coords(pose, index, ok)
    tor_ok = ok.all(-1) & is_na.unsqueeze(-1)

    ring_local = bt_ring.to(torch.int64)[bt_safe]
    ring_pose = torch.arange(n_poses, device=rot_coords.device).view(n_poses, 1, 1)
    ring_index = block_coord_offset.to(torch.int64).unsqueeze(-1) + ring_local
    ring_ok = (ring_local >= 0) & is_na.unsqueeze(-1)
    ring_xyz = gather_coords(ring_pose.expand_as(ring_local), ring_index, ring_ok)

    # index of the preceding block in the flattened pose x block axis
    down = bt_down.to(torch.int64)[bt_safe]
    prev_block = torch.gather(
        inter_residue_connections[..., 0].to(torch.int64),
        2,
        down.clamp_min(0)[..., None],
    ).squeeze(-1)
    has_prev = (down >= 0) & (prev_block >= 0)
    pose_ind = torch.arange(n_poses, device=rot_coords.device).view(n_poses, 1)
    prev = torch.where(
        has_prev,
        pose_ind * max_n_blocks + prev_block,
        torch.full_like(prev_block, -1),
    )

    def flat(t):
        return t.reshape(n_poses * max_n_blocks, *t.shape[2:])

    e_bb, e_chi, e_sugar, e_well = _subterm_energies(
        flat(xyz),
        flat(tor_ok),
        flat(ring_xyz),
        flat(base),
        flat(prev),
        backbone_means,
        backbone_sdev,
        sugar_means,
        chi_means,
        sdev_sugar,
        sdev_chi,
        well_pucker,
        well_alpha_gamma,
        well_bibii_pucker,
        well_alphanext_bibii,
        well_chi_syn,
        is_north,
        pucker_temperature,
        bin_blend_sdev,
    )
    shape = (n_poses, max_n_blocks)
    return (
        e_bb.reshape(shape),
        e_chi.reshape(shape),
        e_sugar.reshape(shape),
        e_well.reshape(shape),
        is_na,
        base,
    )


def _subterm_energies(
    xyz,
    tor_ok,
    ring_xyz,
    base,
    prev,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
):
    """(bb, chi, sugar, well) energies for a flat list of nucleotides.

    xyz: (N, n_torsion, 4, 3) torsion atoms, tor_ok: (N, n_torsion),
    ring_xyz: (N, 5, 3), base: (N), prev: (N) index into N of the preceding
    nucleotide, -1 if there is none.
    """
    dtype = xyz.dtype
    is_na = base >= 0
    # DNA and RNA share the functional form and nothing else; every table below
    # that is not per-base is gathered on this index
    poly = polymer_index(base)
    tors = dihedral(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :], xyz[..., 3, :])
    tors = torch.where(tor_ok, tors, torch.zeros_like(tors)) % 360.0
    zero = torch.zeros(tors.shape[:-1], dtype=dtype, device=tors.device)

    pucker = pucker_weights(ring_xyz, pucker_temperature).to(dtype)
    pucker = torch.where(is_na.unsqueeze(-1), pucker, torch.zeros_like(pucker))

    # --- backbone -----------------------------------------------------------
    means = backbone_means.to(dtype)[poly]
    sdev_bb = backbone_sdev.to(dtype)[poly]
    e_bb = zero

    bin_w = {}
    for tor in (ALPHA, GAMMA):
        ang = tors[..., tor]
        w = triple_bin_weights(ang, means[:, tor, :3], bin_blend_sdev)
        bin_w[tor] = w
        e_bb = e_bb + torch.where(
            tor_ok[..., tor],
            blended_devsq(ang, means[:, tor, :3], w) / sdev_bb[:, tor] ** 2,
            zero,
        )

    w_bi = bi_bii_weight(tors[..., EPSILON], tors[..., ZETA])
    both = tor_ok[..., EPSILON] & tor_ok[..., ZETA]
    for tor in (EPSILON, ZETA):
        ang = tors[..., tor]
        w = torch.stack([w_bi, 1.0 - w_bi], dim=-1)
        e_bb = e_bb + torch.where(
            both,
            blended_devsq(ang, means[:, tor, :2], w) / sdev_bb[:, tor] ** 2,
            zero,
        )

    # beta is binned on the previous residue's BI/BII state;
    # fall back to BI mean when there is no preceding nucleotide
    prev_safe = prev.clamp_min(0)
    prev_ok = (prev >= 0) & both[prev_safe]
    w_beta = torch.where(prev_ok, w_bi[prev_safe], torch.ones_like(w_bi))
    ang = tors[..., BETA]
    e_bb = e_bb + torch.where(
        tor_ok[..., BETA],
        blended_devsq(
            ang, means[:, BETA, :2], torch.stack([w_beta, 1.0 - w_beta], dim=-1)
        )
        / sdev_bb[:, BETA] ** 2,
        zero,
    )

    # --- chi ----------------------------------------------------------------
    chi = tors[..., CHI]
    w_syn = syn_weight(chi)
    dev_pucker = wrap_degrees(
        chi.unsqueeze(-1) - chi_means.to(dtype)[base.clamp_min(0)]
    )
    dev_syn = wrap_degrees(chi - SYN_MEAN).unsqueeze(-1)
    w_syn1 = w_syn.unsqueeze(-1)
    e_chi = (pucker * ((1.0 - w_syn1) * dev_pucker**2 + w_syn1 * dev_syn**2)).sum(
        -1
    ) / sdev_chi.to(dtype)[poly] ** 2
    e_chi = torch.where(tor_ok[..., CHI], e_chi, zero)

    # --- sugar --------------------------------------------------------------
    e_sugar = zero
    for slot, tor in enumerate(SUGAR_SLOTS):
        dev = wrap_degrees(
            tors[..., tor].unsqueeze(-1) - sugar_means.to(dtype)[poly][:, :, slot]
        )
        e_sugar = e_sugar + torch.where(
            tor_ok[..., tor], (pucker * dev**2).sum(-1), zero
        )
    e_sugar = e_sugar / sdev_sugar.to(dtype)[poly] ** 2

    # --- wells --------------------------------------------------------------
    # -ln P of each bin assignment
    e_well = torch.where(is_na, (pucker * well_pucker.to(dtype)[poly]).sum(-1), zero)

    w_a, w_g = bin_w[ALPHA], bin_w[GAMMA]
    e_well = e_well + torch.where(
        tor_ok[..., ALPHA] & tor_ok[..., GAMMA],
        torch.einsum("ni,nij,nj->n", w_a, well_alpha_gamma.to(dtype)[poly], w_g),
        zero,
    )

    w_bibii = torch.stack([w_bi, 1.0 - w_bi], dim=-1)
    w_north = (pucker * is_north.to(dtype)).sum(-1)
    w_ns = torch.stack([w_north, 1.0 - w_north], dim=-1)
    e_well = e_well + torch.where(
        both,
        torch.einsum("nb,nbs,ns->n", w_bibii, well_bibii_pucker.to(dtype)[poly], w_ns),
        zero,
    )

    w_bprev = torch.stack([w_beta, 1.0 - w_beta], dim=-1)
    e_well = e_well + torch.where(
        prev_ok & tor_ok[..., ALPHA],
        torch.einsum(
            "ni,nib,nb->n", w_a, well_alphanext_bibii.to(dtype)[poly], w_bprev
        ),
        zero,
    )

    syn_table = well_chi_syn.to(dtype)[..., base.clamp_min(0)].permute(2, 0, 1)
    e_well = e_well + torch.where(
        tor_ok[..., CHI],
        torch.einsum(
            "ns,nsp,np->n",
            torch.stack([1.0 - w_syn, w_syn], dim=-1),
            syn_table,
            pucker,
        ),
        zero,
    )

    return e_bb, e_chi, e_sugar, e_well


def _resolve_rotamer_uaids(
    uaids,
    pose_ind_for_rot,
    block_ind_for_rot,
    rot_coord_offset,
    first_rot_for_block,
    first_rot_block_type,
    inter_res_conn,
    atom_downstream_of_conn,
):
    """Indices into the rotamer coordinate array for unresolved atom ids.

    Atoms reached through a residue connection come from the neighbor's
    background rotamer: only one block at a time is being rebuilt.
    """
    n_rots = uaids.shape[0]
    trailing = uaids.shape[1:-1]  # (n_torsion, 4)
    view = (n_rots,) + (1,) * len(trailing)

    atom = uaids[..., 0].to(torch.int64)
    conn = uaids[..., 1].to(torch.int64)
    sep = uaids[..., 2].to(torch.int64)
    is_inter = conn >= 0

    pose = pose_ind_for_rot.to(torch.int64).view(view).expand_as(atom)
    block = block_ind_for_rot.to(torch.int64).view(view).expand_as(atom)

    other = inter_res_conn.to(torch.int64)[pose, block, conn.clamp_min(0)]
    other_block = other[..., 0]
    other_conn = other[..., 1]
    ok = torch.where(is_inter, other_block >= 0, atom >= 0)

    other_safe = other_block.clamp_min(0)
    other_rot = first_rot_for_block.to(torch.int64)[pose, other_safe]
    other_bt = first_rot_block_type.to(torch.int64)[pose, other_safe]
    ok = ok & (~is_inter | ((other_rot >= 0) & (other_bt >= 0)))

    local_inter = atom_downstream_of_conn.to(torch.int64)[
        other_bt.clamp_min(0), other_conn.clamp_min(0), sep.clamp_min(0)
    ]
    local = torch.where(is_inter, local_inter, atom)
    ok = ok & (local >= 0)

    self_rot = torch.arange(n_rots, device=uaids.device).view(view).expand_as(atom)
    rot = torch.where(is_inter, other_rot.clamp_min(0), self_rot)
    return rot_coord_offset.to(torch.int64)[rot] + local, ok


def eval_na_torsion_for_rotamers(
    # common args
    rot_coords,
    rot_coord_offset,
    _pose_ind_for_atom,
    first_rot_for_block,
    first_rot_block_type,
    block_ind_for_rot,
    pose_ind_for_rot,
    block_type_ind_for_rot,
    n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    # term args
    has_na,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    _block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
    weight_bb,
    weight_chi,
    weight_sugar,
    output_block_pair_energies: bool,
):
    device = rot_coords.device
    dtype = rot_coords.dtype
    n_rots = block_type_ind_for_rot.shape[0]
    n_poses = n_rots_for_pose.shape[0]

    if not has_na:
        score = torch.zeros((2, n_rots), dtype=dtype, device=device)
    else:
        bt = block_type_ind_for_rot.to(torch.int64)
        base = torch.where(bt >= 0, bt_base.to(torch.int64)[bt.clamp_min(0)], -1)
        pose = pose_ind_for_rot.to(torch.int64)
        block = block_ind_for_rot.to(torch.int64)
        bt_safe = bt.clamp_min(0)

        index, ok = _resolve_rotamer_uaids(
            bt_uaids[bt_safe],
            pose,
            block,
            rot_coord_offset,
            first_rot_for_block,
            first_rot_block_type,
            inter_residue_connections,
            atom_downstream_of_conn,
        )
        is_na = base >= 0
        tor_ok = ok.all(-1) & is_na.unsqueeze(-1)

        def gather_coords(index, ok):
            xyz = rot_coords[index.clamp_min(0).reshape(-1)].reshape(*index.shape, 3)
            return torch.where(ok.unsqueeze(-1), xyz, torch.zeros_like(xyz))

        xyz = gather_coords(index, ok)

        ring_local = bt_ring.to(torch.int64)[bt_safe]
        ring_ok = (ring_local >= 0) & is_na.unsqueeze(-1)
        ring_xyz = gather_coords(
            rot_coord_offset.to(torch.int64).unsqueeze(-1) + ring_local, ring_ok
        )

        # the preceding nucleotide is never itself a rotamer being built, so
        # beta reads the BI/BII state off that block's background rotamer
        down = bt_down.to(torch.int64)[bt_safe]
        prev_block = inter_residue_connections.to(torch.int64)[
            pose, block, down.clamp_min(0), 0
        ]
        has_prev = (down >= 0) & (prev_block >= 0)
        prev = torch.where(
            has_prev,
            first_rot_for_block.to(torch.int64)[pose, prev_block.clamp_min(0)],
            torch.full_like(prev_block, -1),
        )

        e_bb, e_chi, e_sugar, e_well = _subterm_energies(
            xyz,
            tor_ok,
            ring_xyz,
            base,
            prev,
            backbone_means,
            backbone_sdev,
            sugar_means,
            chi_means,
            sdev_sugar,
            sdev_chi,
            well_pucker,
            well_alpha_gamma,
            well_bibii_pucker,
            well_alphanext_bibii,
            well_chi_syn,
            is_north,
            pucker_temperature,
            bin_blend_sdev,
        )
        poly = polymer_index(base)
        harmonic = (
            weight_bb[poly] * e_bb
            + weight_chi[poly] * e_chi
            + weight_sugar[poly] * e_sugar
        )
        score = torch.stack([harmonic, e_well])
        score = torch.where(is_na.unsqueeze(0), score, torch.zeros_like(score))

    if output_block_pair_energies:
        # a one-body term: each energy sits on the diagonal of the rotamer pair
        rot_ind = torch.arange(n_rots, dtype=torch.int32, device=device)
        indices = torch.stack([pose_ind_for_rot.to(torch.int32), rot_ind, rot_ind])
        output_scores = score
    else:
        output_scores = torch.zeros((2, n_poses), dtype=dtype, device=device)
        output_scores.index_add_(1, pose_ind_for_rot.to(torch.int64), score)
        indices = torch.zeros((0,), dtype=torch.int32, device=device)

    return output_scores, indices


def _finish(scores, output_block_pair_energies):
    stacked = torch.stack(scores)  # (score type, pose, block)
    if output_block_pair_energies:
        return torch.diag_embed(stacked), None
    return torch.sum(stacked, dim=2), None
