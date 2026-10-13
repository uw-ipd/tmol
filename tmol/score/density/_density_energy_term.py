import math

import numpy
import torch

from .._energy_term import EnergyTerm
from ._fast_density import FastDensityScore
from .density import ElectronDensityMap, block_type_atomic_numbers

from tmol.chemical import RefinedResidueType, l_base_name
from tmol.database import ParameterDatabase
from tmol.pose import PoseStack


def sidechain_atom_mask(block_type: RefinedResidueType) -> numpy.ndarray:
    """Atoms hanging off an interior main-chain atom (CA for a protein).

    Atoms attached through the first or last main-chain atom (H, O, OXT) are
    backbone. Non-polymers have no side chain.
    """
    mask = numpy.zeros(block_type.n_atoms, dtype=bool)
    mainchain = block_type.properties.polymer.mainchain_atoms
    if not mainchain or len(mainchain) < 3:
        return mask
    mc = [block_type.atom_to_idx[name] for name in mainchain]
    neighbors = [[] for _ in range(block_type.n_atoms)]
    for a, b in block_type.bond_indices:
        neighbors[a].append(b)
    stack = [b for a in mc[1:-1] for b in neighbors[a] if b not in mc]
    while stack:
        a = stack.pop()
        if not mask[a]:
            mask[a] = True
            stack.extend(b for b in neighbors[a] if b not in mc and not mask[b])
    return mask


class DensityEnergyTerm(EnergyTerm):
    """Rosetta-style elec_dens_fast fit-to-density score of every heavy atom.

    E = - sum_atoms (trunc(a_elt) / 6) * sc_scale * S(x_atom), where S is the
    normalized score grid of FastDensityScore.

    The observed map is pose_stack.density_map, one map per stack.

    Score-function options:
        density_periodic: treat the map as periodic (default False).
        density_scatterers: scattering table, "electron" (default) or "xray".
        density_sc_scale: residue name -> side-chain scale (default: none).
    """

    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        self.scatterer_tables = param_db.scoring.elec_dens.scatterers
        self.periodic = False
        self.scatterers = "electron"
        self.sc_scale = {}
        self._scorer = None
        self._scorer_map = None
        self._scorer_key = None

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
        self.periodic = bool(options.get("density_periodic", False))
        self.scatterers = options.get("density_scatterers", "electron")
        self.sc_scale = options.get("density_sc_scale") or {}
        if self.scatterers not in self.scatterer_tables:
            raise ValueError(
                f"density_scatterers must be one of {sorted(self.scatterer_tables)}"
            )

    def _get_scorer(self, density_map) -> FastDensityScore:
        if density_map is None:
            raise ValueError("elec_dens_fast needs a pose_stack.density_map")
        if not isinstance(density_map, ElectronDensityMap):
            raise TypeError("pose_stack.density_map must be an ElectronDensityMap")
        carbon_sigma = self.scatterer_tables[self.scatterers]["C"].sigma
        key = (self.periodic, carbon_sigma)
        if self._scorer_map is not density_map or self._scorer_key != key:
            self._scorer = FastDensityScore(
                density_map.to(self.device),
                carbon_sigma,
                periodic=self.periodic,
            )
            self._scorer_map = density_map
            self._scorer_key = key
        return self._scorer

    def _block_type_atom_weights(self, packed_block_types) -> torch.Tensor:
        """Per atom of every block type: (trunc(a_elt) / 6) * sc_scale.

        The scattering weight is truncated to an integer, as Rosetta stores it.
        Hydrogens, virtual atoms and padding get zero.
        """
        table = self.scatterer_tables[self.scatterers]
        z_for_element = {
            e.name: e.atomic_number for e in packed_block_types.chem_db.element_types
        }
        weight_for_z = {
            z_for_element[name]: math.trunc(p.weight) / 6.0 for name, p in table.items()
        }
        z = block_type_atomic_numbers(packed_block_types).numpy()
        weight = numpy.where(z > 1, weight_for_z[z_for_element["C"]], 0.0)
        for atomic_number, w in weight_for_z.items():
            weight[z == atomic_number] = w
        for i, block_type in enumerate(packed_block_types.active_block_types):
            factor = self.sc_scale.get(block_type.base_name)
            if factor is None:
                factor = self.sc_scale.get(l_base_name(block_type))
            if factor is not None:
                sc = sidechain_atom_mask(block_type)
                weight[i, : block_type.n_atoms][sc] *= factor
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)

    def get_pose_score_term_function(self):
        return eval_density_energy_for_pose

    def get_rotamer_score_term_function(self):
        return eval_density_energy_for_rotamers

    def get_score_term_attributes(self, pose_stack: PoseStack):
        scorer = self._get_scorer(pose_stack.density_map)
        pbt = pose_stack.packed_block_types
        block_type_weight = self._block_type_atom_weights(pbt)

        # whole-pose scoring has one rotamer per block, so the scoring atoms are
        # fixed per render: flat coordinate index, owning pose and block, weight
        block_type = pose_stack.block_type_ind64
        pose_of_block, block_of_block = (block_type >= 0).nonzero(as_tuple=True)
        block_bt = block_type[pose_of_block, block_of_block]
        offset = pose_stack.block_coord_offset64[pose_of_block, block_of_block]
        block_of_atom, local = _atoms_of(pbt.n_atoms.to(torch.int64)[block_bt])
        pose_of_atom = pose_of_block[block_of_atom]
        flat_index = (
            pose_of_atom * pose_stack.max_n_pose_atoms + offset[block_of_atom] + local
        )
        weight = block_type_weight[block_bt[block_of_atom], local]
        keep = weight != 0
        n_blocks = pose_stack.max_n_blocks
        block = block_of_block[block_of_atom]
        diagonal = (pose_of_atom * n_blocks + block) * n_blocks + block
        return [
            scorer,
            flat_index[keep],
            weight[keep],
            pose_of_atom[keep],
            diagonal[keep],
        ]

    def get_rotamer_score_term_attributes(self, pose_stack, rotamer_set):
        scorer = self._get_scorer(pose_stack.density_map)
        pbt = pose_stack.packed_block_types
        block_type_weight = self._block_type_atom_weights(pbt)
        rot_bt = rotamer_set.block_type_ind_for_rot
        n_atoms = torch.where(
            rot_bt >= 0, pbt.n_atoms.to(torch.int64)[rot_bt.clamp_min(0)], 0
        )
        rot_of_atom, local = _atoms_of(n_atoms)
        index = rotamer_set.coord_offset_for_rot.to(torch.int64)[rot_of_atom] + local
        weight = block_type_weight[rot_bt[rot_of_atom], local]
        keep = weight != 0
        rot_of_atom = rot_of_atom[keep]
        n_rots = rot_bt.shape[0]
        rotamer = torch.arange(n_rots, dtype=torch.int32, device=rot_bt.device)
        indices = torch.stack(
            [rotamer_set.pose_for_rot.to(torch.int32), rotamer, rotamer]
        )
        return [
            scorer,
            index[keep],
            weight[keep],
            rotamer_set.pose_for_rot[rot_of_atom],
            rot_of_atom,
            indices,
        ]


def _atoms_of(n_atoms: torch.Tensor):
    """Owner and local index of every atom of consecutive owners with n_atoms."""
    owner = torch.repeat_interleave(
        torch.arange(n_atoms.shape[0], device=n_atoms.device), n_atoms
    )
    first = torch.cumsum(n_atoms, 0) - n_atoms
    local = torch.arange(owner.shape[0], device=n_atoms.device) - first[owner]
    return owner, local


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
    atom_index,
    atom_weight,
    atom_pose,
    atom_diagonal,
    output_block_pair_energies: bool,
):
    n_poses, n_blocks = first_rot_block_type.shape
    if output_block_pair_energies:
        score = scorer.score(
            coords, atom_index, atom_diagonal, atom_weight, n_poses * n_blocks**2
        ).reshape(n_poses, n_blocks, n_blocks)
    else:
        score = scorer.score(coords, atom_index, atom_pose, atom_weight, n_poses)
    return score.unsqueeze(0), None


def eval_density_energy_for_rotamers(
    # common args
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    block_type_ind_for_rot,
    n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _lockstep_group_for_block,
    _max_n_rots_per_pose,
    # term args
    scorer,
    atom_index,
    atom_weight,
    atom_pose,
    atom_rotamer,
    indices,
    output_block_pair_energies: bool,
):
    if output_block_pair_energies:
        n_rots = block_type_ind_for_rot.shape[0]
        score = scorer.score(rot_coords, atom_index, atom_rotamer, atom_weight, n_rots)
        return score.unsqueeze(0), indices
    n_poses = n_rots_for_pose.shape[0]
    score = scorer.score(rot_coords, atom_index, atom_pose, atom_weight, n_poses)
    return score.unsqueeze(0), torch.zeros(
        (0,), dtype=torch.int32, device=rot_coords.device
    )
