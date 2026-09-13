import math

import numpy
import torch
import attr

from typing import Tuple

from tmol.types import (
    Tensor,
    validate_args,
)
from tmol.utility.tensor import exclusive_cumsum1d

from tmol.database import ParameterDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.pack import SetPackerTask
from tmol.pack._packer_task import (
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)
from tmol.pack.rotamer._chi_budget import (
    apply_chi_sample_budget,
    chi_depths,
    sampler_for_task,
)
from tmol.pack.rotamer._single_residue_kinforest import (
    construct_single_residue_kinforest,
)
from tmol.pack.rotamer import ChiSampler, sc_roots_for_chis
from tmol.score.na_torsion import (
    NaTorsionParams,
    N_PUCKER,
    POLYMER_IND,
    SYN_MEAN,
    block_type_params,
    polymer_index,
    pucker_weights,
)

# k in chi = mean + k * sdev_chi, by extra-chi sample level, mirroring Rosetta's
# expansion of its fitted chi gaussians
CHI_STEPS = {
    0: (0.0,),
    1: (0.0, 1.0, -1.0),
    2: (0.0, 0.5, -0.5),
    3: (0.0, 1.0, 2.0, -1.0, -2.0),
    4: (0.0, 0.5, 1.0, -0.5, -1.0),
}

# a syn well deeper than this is too rare to be worth a rotamer
MAX_SYN_WELL = 5.0


@attr.s(auto_attribs=True, frozen=True)
class NaChiRotamerSampler(ChiSampler):
    """Glycosidic chi rotamers for DNA and RNA, taken from the na_torsion tables.

    Every chi generated sits at a minimum of the term that will score it: the
    anti rotamer is the per-base, per-pucker mean and the syn rotamer is the
    fixed syn mean, each expanded in units of that term's own sdev_chi. The
    pucker comes from the input sugar, which is not itself a packing degree of
    freedom.

    The 2'-OH proton chi is expanded here rather than left to OptHSampler,
    which is applied to disjoint blocks; see DunbrackChiSampler, which folds
    proton chis in the same way.
    """

    params: NaTorsionParams
    element_for_atom_type: dict
    chi_sample_level: int = 0
    sample_syn: bool = True
    device: torch.device = torch.device("cpu")

    chi_sample_expanded_limit: int = DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT
    chi_sample_limit: int = DEFAULT_CHI_SAMPLE_LIMIT

    @classmethod
    def from_database(
        cls,
        param_db: ParameterDatabase,
        device: torch.device,
        chi_sample_level: int = 0,
        sample_syn: bool = True,
    ):
        return cls(
            params=NaTorsionParams.from_database(param_db.scoring.na_torsion, device),
            element_for_atom_type={
                at.name: at.element for at in param_db.chemical.atom_types
            },
            chi_sample_level=chi_sample_level,
            sample_syn=sample_syn,
            device=device,
        )

    @classmethod
    def sampler_name(cls):
        return "NaChiRotamerSampler"

    def _annotation_key(self):
        # Keep only the most recent tables on each type/PBT, but validate all
        # inputs to those tables. A different sampler must not inherit the
        # first sampler's budget or chemical element assignments.
        return (
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
            tuple(sorted(self.element_for_atom_type.items())),
        )

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        key = self._annotation_key()
        if getattr(rt, "_na_chi_sampler_key", None) == key and hasattr(
            rt, "na_chi_sampler_params"
        ):
            return rt.na_chi_sampler_params
        p = block_type_params(rt, self.element_for_atom_type)
        chi1 = rt.torsion_to_uaids.get("chi1")
        # the third atom of a torsion is the one whose dof carries it
        p["chi1_atom"] = -1 if chi1 is None else chi1[2][0]
        # one residue at a time, so the budget is per residue: no grouping, but
        #    the same limits the rest of the packer applies
        # Protein/ligand chi must not enlarge this sampler's Cartesian product
        # or padded tensors. Other samplers own those residues.
        sampled = list(rt.chi_samples) if p["base"] >= 0 else []
        if sampled:
            construct_single_residue_kinforest(rt)
            depths = chi_depths(
                rt.residue_kinforest_data,
                [rt.torsion_to_uaids[cs.chi_dihedral][2][0] for cs in sampled],
            )
            sampled = list(
                apply_chi_sample_budget(
                    sampled,
                    depths,
                    self.chi_sample_expanded_limit,
                    self.chi_sample_limit,
                )
            )
        p["proton_chi"] = [
            (
                int(samp.chi_dihedral[3:]) - 1,
                rt.torsion_to_uaids[samp.chi_dihedral][2][0],
                _expanded_samples(samp),
            )
            for samp in sampled
        ]
        setattr(rt, "na_chi_sampler_params", p)
        setattr(rt, "_na_chi_sampler_key", key)
        return p

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        key = self._annotation_key()
        if getattr(packed_block_types, "_na_chi_sampler_key", None) == key:
            return packed_block_types.na_chi_sampler_cache
        bts = packed_block_types.active_block_types
        params = [self.annotate_residue_type(bt) for bt in bts]
        n_chi = max(1 + len(p["proton_chi"]) for p in params)

        def to_t(arr, dtype=torch.int32):
            return torch.tensor(arr, dtype=dtype, device=packed_block_types.device)

        ring = numpy.stack([p["ring"] for p in params])
        base = numpy.array([p["base"] for p in params])
        chi_atom = numpy.full((len(bts), n_chi), -1, dtype=numpy.int32)
        combos = [_proton_combinations(p["proton_chi"]) for p in params]
        n_combos = numpy.array([len(c) for c in combos], dtype=numpy.int64)
        proton = numpy.zeros((len(bts), n_combos.max(), n_chi - 1), dtype=numpy.float32)
        for i, p in enumerate(params):
            if p["base"] < 0:
                continue
            chi_atom[i, 0] = p["chi1_atom"]
            for slot, (_, atom, _) in enumerate(p["proton_chi"]):
                chi_atom[i, slot + 1] = atom
            for j, combo in enumerate(combos[i]):
                proton[i, j, : len(combo)] = combo

        cache = dict(
            base=to_t(base, torch.int64),
            ring=to_t(ring, torch.int64),
            chi_atom=to_t(chi_atom),
            builds_bt=to_t(base >= 0, torch.bool),
            n_combos=to_t(n_combos, torch.int64),
            proton=to_t(proton, torch.float32),
            n_chi=n_chi,
        )
        setattr(packed_block_types, "na_chi_sampler_cache", cache)
        setattr(packed_block_types, "_na_chi_sampler_key", key)
        return cache

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return self.annotate_residue_type(rt)["base"] >= 0

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return self.annotate_packed_block_types(pbt)["builds_bt"][bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return sc_roots_for_chis(
            rt, ("chi1",) + tuple(cs.chi_dihedral for cs in rt.chi_samples)
        )

    def _pucker_for_blocks(self, poses: PoseStack, pbt: PackedBlockTypes, cache=None):
        """argmax pucker state of every block's input sugar, -1 where absent."""
        if cache is None:
            cache = self.annotate_packed_block_types(pbt)
        bt = poses.block_type_ind64
        real = bt >= 0
        ring_local = cache["ring"][bt.clamp_min(0)]
        offset = poses.block_coord_offset64.unsqueeze(-1)
        ok = (ring_local >= 0) & real.unsqueeze(-1)
        index = (offset + ring_local).clamp_min(0)
        n_poses = bt.shape[0]
        max_n_atoms = poses.coords.shape[1]
        flat = (
            torch.arange(n_poses, device=poses.device).view(-1, 1, 1) * max_n_atoms
            + index
        )
        xyz = poses.coords.reshape(-1, 3)[flat.reshape(-1)].reshape(*index.shape, 3)
        xyz = torch.where(ok.unsqueeze(-1), xyz, torch.zeros_like(xyz))
        w = pucker_weights(
            xyz.reshape(-1, 5, 3), self.params.pucker_temperature
        ).reshape(*bt.shape, N_PUCKER)
        pucker = w.argmax(dim=-1)
        return torch.where(ok.all(-1), pucker, torch.full_like(pucker, -1))

    @validate_args
    def sample_chi_for_poses(
        self, poses: PoseStack, task: SetPackerTask
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        configured, task = sampler_for_task(self, task)
        if configured is not self:
            return configured.sample_chi_for_poses(poses, task)
        pbt = poses.packed_block_types
        cache = self.annotate_packed_block_types(pbt)

        self_ind = task.conformer_sampler_index[id(self)]
        allowed = task.per_block_conformer_sampler_allowed[:, :, self_ind]
        allowed_for_cons = allowed[task.cons_bt_pose, task.cons_bt_block]
        builds = cache["builds_bt"][task.cons_bt_block_type]
        bt_allowed = task.per_block_is_block_type_allowed[
            task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
        ]
        active = allowed_for_cons & builds & bt_allowed

        pucker_for_block = self._pucker_for_blocks(poses, pbt, cache)
        pucker = pucker_for_block[task.cons_bt_pose, task.cons_bt_block]
        base = cache["base"][task.cons_bt_block_type]
        active = active & (pucker >= 0)

        n_chi = cache["n_chi"]
        n_gbt = active.shape[0]
        bt_for_gbt = task.cons_bt_block_type
        p = self.params

        # how many chi1 values each block gets: the anti mode always, the syn
        # mode when the well is shallow enough and the polymer is RNA
        steps = torch.tensor(
            CHI_STEPS[self.chi_sample_level], dtype=torch.float32, device=poses.device
        )
        n_steps = steps.shape[0]
        safe_base = base.clamp_min(0)
        poly = polymer_index(base)
        syn_ok = (
            active
            & (poly == POLYMER_IND["rna"])
            & (p.well_chi_syn[1, pucker.clamp_min(0), safe_base] <= MAX_SYN_WELL)
        )
        if not self.sample_syn:
            syn_ok = torch.zeros_like(syn_ok)
        n_modes = torch.where(syn_ok, 2, 1)
        n_combos = cache["n_combos"][bt_for_gbt]
        # Preserve an out-of-range sentinel while bounding the int64 product.
        # Chi levels have at most five steps and two modes. Narrow only after
        # validating both the individual counts and their total.
        bounded_combos = n_combos.clamp(-1, torch.iinfo(torch.int32).max + 1)
        counts = torch.where(
            active, n_modes * n_steps * bounded_combos, torch.zeros_like(n_combos)
        )

        from tmol.pack.rotamer._chi_budget import checked_sample_count

        n_rots = checked_sample_count(
            counts, self.chi_sample_expanded_limit, self.chi_sample_limit
        )
        n_rots_for_gbt = counts.to(torch.int32)

        if n_rots == 0:
            return (
                n_rots_for_gbt,
                torch.zeros((0,), dtype=torch.int32, device=poses.device),
                torch.zeros((0, n_chi), dtype=torch.int32, device=poses.device),
                torch.zeros((0, n_chi), dtype=torch.float32, device=poses.device),
            )

        gbt_for_rotamer = torch.repeat_interleave(
            torch.arange(n_gbt, dtype=torch.int64, device=poses.device), counts
        )
        # index of each rotamer within its own block
        local = torch.arange(
            n_rots, dtype=torch.int64, device=poses.device
        ) - torch.repeat_interleave(exclusive_cumsum1d(counts), counts)

        combos_per_rot = n_combos[gbt_for_rotamer]
        chi1_slot = torch.div(local, combos_per_rot, rounding_mode="floor")
        combo_slot = local % combos_per_rot
        mode = torch.div(chi1_slot, n_steps, rounding_mode="floor")
        step_ind = chi1_slot % n_steps

        rot_base = safe_base[gbt_for_rotamer]
        rot_pucker = pucker.clamp_min(0)[gbt_for_rotamer]
        sdev = p.sdev_chi[poly[gbt_for_rotamer]]
        center = torch.where(
            mode == 0,
            p.chi_means[rot_base, rot_pucker],
            torch.full_like(sdev, SYN_MEAN),
        )
        chi1 = center + steps[step_ind] * sdev

        rot_bt = bt_for_gbt[gbt_for_rotamer]
        chi = torch.cat(
            [
                torch.deg2rad(chi1).unsqueeze(-1),
                cache["proton"][rot_bt, combo_slot],
            ],
            dim=-1,
        )
        return (
            n_rots_for_gbt,
            gbt_for_rotamer.to(torch.int32),
            cache["chi_atom"][rot_bt],
            chi,
        )


def _expanded_samples(samp):
    """A proton chi's samples in radians, each with its expansions either side."""
    values = []
    for sample in samp.samples:
        values.append(sample)
        for expansion in samp.expansions:
            values.extend((sample + expansion, sample - expansion))
    return tuple(math.radians(float(v)) for v in values)


def _proton_combinations(proton_chi):
    """Cross product of the proton-chi samples, or a single empty set."""
    if not proton_chi:
        return [()]
    combos = [()]
    for _, _, samples in proton_chi:
        combos = [c + (s,) for c in combos for s in samples]
    return combos
