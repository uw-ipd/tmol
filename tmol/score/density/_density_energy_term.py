import torch

from .._energy_term import EnergyTerm
from ._fast_density import FastDensityScore
from .density import ElectronDensityMap, block_type_atomic_numbers

from tmol.database import ParameterDatabase
from tmol.pose import PoseStack

# ``scale_sc_dens_byres`` of Rosetta's cryoem_glycan_refinement.xml: side-chain
# density is down-weighted by residue type.
CRYOEM_SIDECHAIN_SCALE: dict[str, float] = {
    **dict.fromkeys(("ARG", "LYS", "GLU", "ASP", "MET"), 0.66),
    **dict.fromkeys(("CYS", "GLN", "HIS", "ASN", "THR", "SER"), 0.71),
    **dict.fromkeys(("TYR", "TRP", "ALA", "PHE", "PRO", "ILE", "LEU", "VAL"), 0.78),
}
_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


class DensityEnergyTerm(EnergyTerm):
    """Rosetta-style ``elec_dens_fast`` fit-to-density score of every heavy atom.

    ``E = - sum_atoms (a_elt / 6) * sidechain_scale * S(x_atom)`` over heavy
    atoms, where ``S`` is the normalized score grid of :class:`FastDensityScore`.
    The term is unweighted; weight it through ``ScoreType.elec_dens_fast``
    (Rosetta's cryo-EM script uses 35).

    The observed map is a score-function option, not part of the pose::

        sfxn.set_options(
            {"density_map": ElectronDensityMap, "density_resolution": 3.0}
        )

    ``density_scale_sidechains`` (default ``True``) applies Rosetta's per-residue
    side-chain scale.
    """

    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        self.density_map = None
        self.resolution = None
        self.scale_sidechains = True
        self._scorer = None

    @classmethod
    def class_name(cls):
        return "ElecDensFast"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._density_creator

        return tmol.score.terms._density_creator.DensityTermCreator.score_types()

    def n_bodies(self):
        return 1

    def set_options(self, options: dict):
        density_map = options.get("density_map")
        resolution = options.get("density_resolution")
        if density_map is not self.density_map or resolution != self.resolution:
            self._scorer = None
        self.density_map = density_map
        self.resolution = resolution
        self.scale_sidechains = bool(options.get("density_scale_sidechains", True))

    def _get_scorer(self) -> FastDensityScore:
        if self.density_map is None or self.resolution is None:
            raise ValueError(
                "elec_dens_fast needs the score-function options 'density_map' "
                "(an ElectronDensityMap) and 'density_resolution'"
            )
        if not isinstance(self.density_map, ElectronDensityMap):
            raise TypeError("'density_map' must be an ElectronDensityMap")
        if self._scorer is None:
            self._scorer = FastDensityScore(
                self.density_map.to(self.device), self.resolution
            )
        return self._scorer

    def _block_type_atom_weights(
        self, packed_block_types, scorer: FastDensityScore
    ) -> torch.Tensor:
        """Per atom of every block type: ``(a_elt / 6) * sidechain_scale``.

        Hydrogens and padding atoms get zero.
        """
        z = block_type_atomic_numbers(packed_block_types, self.device)
        weight = scorer.amplitude(z) * (z > 1)
        if self.scale_sidechains:
            scale = torch.ones_like(weight)
            for i, block_type in enumerate(packed_block_types.active_block_types):
                factor = CRYOEM_SIDECHAIN_SCALE.get(block_type.base_name)
                if factor is None:
                    continue
                for j, atom in enumerate(block_type.atoms):
                    if atom.name not in _BACKBONE_ATOM_NAMES:
                        scale[i, j] = factor
            weight = weight * scale
        return weight.to(scorer.coeffs.dtype)

    def get_pose_score_term_function(self):
        return eval_density_energy_for_pose

    def get_rotamer_score_term_function(self):
        return eval_density_energy_for_rotamers

    def get_score_term_attributes(self, pose_stack: PoseStack):
        scorer = self._get_scorer()
        pbt = pose_stack.packed_block_types
        block_type_weight = self._block_type_atom_weights(pbt, scorer)
        block_type_n_atoms = pbt.n_atoms.to(torch.int64)

        # Whole-pose scoring has one rotamer per block, so the atom list is static:
        # precompute the flat coordinate index, the owning pose and block, and the
        # weight of every scoring atom once per render.
        block_type = pose_stack.block_type_ind64
        pose_of_block, block_of_block = (block_type >= 0).nonzero(as_tuple=True)
        block_bt = block_type[pose_of_block, block_of_block]
        n_atoms = block_type_n_atoms[block_bt]
        block_of_atom = torch.repeat_interleave(
            torch.arange(n_atoms.shape[0], device=n_atoms.device), n_atoms
        )
        first_atom = torch.cumsum(n_atoms, 0) - n_atoms
        local = (
            torch.arange(block_of_atom.shape[0], device=n_atoms.device)
            - first_atom[block_of_atom]
        )
        offset = pose_stack.block_coord_offset64[pose_of_block, block_of_block][
            block_of_atom
        ]
        pose_of_atom = pose_of_block[block_of_atom]
        flat_index = pose_of_atom * pose_stack.max_n_pose_atoms + offset + local
        weight = block_type_weight[block_bt[block_of_atom], local]
        keep = weight != 0
        return [
            scorer,
            flat_index[keep],
            pose_of_atom[keep],
            block_of_block[block_of_atom][keep],
            weight[keep],
            block_type_weight,
            block_type_n_atoms,
        ]


def eval_density_energy_for_pose(
    # common args
    coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    _block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    # term args
    scorer,
    flat_index,
    pose_of_atom,
    block_of_atom,
    atom_weight,
    _block_type_weight,
    _block_type_n_atoms,
    output_block_pair_energies: bool,
):
    energy = -atom_weight.to(coords.dtype) * scorer(coords[flat_index])
    n_poses, max_n_blocks = first_rot_block_type.shape
    if output_block_pair_energies:
        score = coords.new_zeros((n_poses, max_n_blocks, max_n_blocks))
        score = score.index_put(
            (pose_of_atom, block_of_atom, block_of_atom), energy, accumulate=True
        )
    else:
        score = coords.new_zeros((n_poses,)).index_add(0, pose_of_atom, energy)
    return score.unsqueeze(0), None


def eval_density_energy_for_rotamers(
    # common args
    rot_coords,
    rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    pose_ind_for_rot,
    block_type_ind_for_rot,
    n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _lockstep_group_for_block,
    _max_n_rots_per_pose,
    # term args
    scorer,
    _flat_index,
    _pose_of_atom,
    _block_of_atom,
    _atom_weight,
    block_type_weight,
    block_type_n_atoms,
    output_block_pair_energies: bool,
):
    device = rot_coords.device
    block_type = block_type_ind_for_rot.to(torch.int64)
    n_rots = block_type.shape[0]
    n_atoms = torch.where(
        block_type >= 0,
        block_type_n_atoms[block_type.clamp_min(0)],
        torch.zeros_like(block_type),
    )
    rot_of_atom = torch.repeat_interleave(torch.arange(n_rots, device=device), n_atoms)
    first_atom = torch.cumsum(n_atoms, 0) - n_atoms
    local = torch.arange(rot_of_atom.shape[0], device=device) - first_atom[rot_of_atom]
    index = rot_coord_offset.to(torch.int64)[rot_of_atom] + local
    weight = block_type_weight[block_type[rot_of_atom], local].to(rot_coords.dtype)
    energy = -weight * scorer(rot_coords[index])
    rotamer_score = rot_coords.new_zeros((n_rots,)).index_add(0, rot_of_atom, energy)

    if output_block_pair_energies:
        indices = torch.zeros((3, n_rots), dtype=torch.int32, device=device)
        indices[0, :] = pose_ind_for_rot
        rotamer_index = torch.arange(n_rots, dtype=torch.int32, device=device)
        indices[1, :] = rotamer_index
        indices[2, :] = rotamer_index
        return rotamer_score.unsqueeze(0), indices
    pose_score = rot_coords.new_zeros(n_rots_for_pose.shape).index_add(
        0, pose_ind_for_rot.to(torch.int64), rotamer_score
    )
    return pose_score.unsqueeze(0), torch.zeros((0,), dtype=torch.int32, device=device)
