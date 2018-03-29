import properties
import torch
import numpy

from tmol.properties.reactive import derived_from
from tmol.properties.array import VariableT

from .bonded_atom import BondedAtomScoreGraph

class InteratomicDistanceGraphBase(BondedAtomScoreGraph):
    """Base graph for interatomic distances.

    Graph component calculating interatomic distances. Distances are present *once* in the source
    graph for a given atomic pair; the rendered distances are equivalent to the upper triangle of
    the full interatomic distance matrix.

    Distances are rendered as two tensor properties, `atom_pair_inds` and `atom_pair_dist`,
    containing the pair of atomic indicies in `coords` and the calculated distance respectively.
    """

    atom_pair_dist_thresholds = properties.Set(
        "interaction threshold distances that *may* be used to optimize distance pair selection",
        prop=properties.Float("maximum nonzero interaction distance threshold", min=0, cast=True),
        default = set(),
        observe_mutations = True
    )

    @property
    def atom_pair_inds(self):
        raise NotImplementedError()

    @property
    def atom_pair_dist(self):
        raise NotImplementedError()

class NaiveInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    @derived_from(
        ("system_size"),
        VariableT("index pairs for all atom pairs"))
    def atom_pair_inds(self):
        return torch.stack(tuple(map(torch.LongTensor,
            numpy.triu_indices(self.system_size))))

    @derived_from(
        ("coords", "atom_pair_inds"),
        VariableT("inter-atomic pairwise distance within threshold distance"))
    def atom_pair_dist(self):
        dist = (
            self.coords[self.atom_pair_inds[0]] - self.coords[self.atom_pair_inds[1]]
        ).norm(dim=-1)

        dist.register_hook(self.nan_to_num)

        return dist

class BlockedInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    atom_pair_block_size = properties.Integer(
        "atom block size for block-neighbor optimization", min=1, cast=True, default=8)

    @property
    def interatomic_threshold_distance(self):
        if self.atom_pair_dist_thresholds:
            return min(self.atom_pair_dist_thresholds)
        else:
            return numpy.inf

    @derived_from(("coords", "atom_pair_block_size"),
        VariableT("atomic coordinate blocks"))
    def coord_blocks(self):
        return self.nan_to_num(self.coords).reshape(-1, self.atom_pair_block_size, 3)

    @derived_from(("system_size", "atom_pair_block_size"),
        VariableT("a"))
    def atom_idx_blocks(self):
        return torch.LongTensor(numpy.arange(self.system_size)).reshape(
                -1, self.atom_pair_block_size)

    @derived_from(("system_size", "atom_pair_block_size"),
        VariableT("a"))
    def block_triu_indices(self):
        return torch.stack(tuple(map(torch.LongTensor(numpy.triu_indices(self.system_size, k=1)))))

    @derived_from(("system_size", "atom_pair_block_size"),
        VariableT("a"))
    def block_tril_indices(self):
        return torch.stack(tuple(map(torch.LongTensor(numpy.tril_indices(self.system_size, k=1)))))

    @derived_from(("coords", "atom_pair_block_size"),
        VariableT("indicies of atom blocks potentially within threshold distance"))
    def block_interactions(self):
        bs = self.atom_pair_block_size

        raw_blocked_atom_mask = self.real_atoms.reshape(-1, bs)
        valid_blocks = raw_blocked_atom_mask.sum(dim=-1).nonzero().squeeze()

        coord_blocks = self.coord_blocks[valid_blocks]
        atom_mask = raw_blocked_atom_mask[valid_blocks]
        atoms_per_block = atom_mask.sum(dim=-1).float()

        block_centers = coord_blocks.sum(dim=1) / torch.Tensor(atoms_per_block.reshape((-1, 1)))

        block_radii = (
            (coord_blocks - block_centers.reshape((-1, 1, 3)))
            .norm(dim=-1)
            .where(atom_mask, torch.Tensor([0.0]))
            .max(dim=-1)[0]
        )

        interblock_center_dist = (
            (block_centers.reshape((-1, 1, 3)) - block_centers.reshape((1, -1, 3)))
            .norm(dim=-1)
        )

        interblock_min_atomic_dist = (
            interblock_center_dist - (block_radii.reshape((-1, 1)) + block_radii.reshape((1, -1))))

        return valid_blocks[
            (interblock_min_atomic_dist < self.interatomic_threshold_distance).nonzero()
        ]

    @derived_from(
        ("atom_idx_blocks", "block_interactions"),
        VariableT("index pairs for all pairs potentially within interaction distance threshold"))
    def atom_pair_inds(self):
        bs = self.atom_pair_block_size
        b_ind = self.block_interactions.transpose(0, 1)
        ai_b = self.atom_idx_blocks

        blocked_from = ai_b[b_ind[0]].reshape((-1, bs, 1)).repeat((1, 1, bs)).reshape(-1)
        blocked_to = ai_b[b_ind[1]].reshape((-1, 1, bs)).repeat((1, bs, 1)).reshape(-1)

        return torch.stack((blocked_from, blocked_to))

    @derived_from(
        ("coords", "coord_blocks", "block_interactions"),
        VariableT("inter-atomic pairwise distance within threshold distance"))
    def atom_pair_dist(self):
        bs = self.atom_pair_block_size
        b_ind = self.block_interactions.transpose(0, 1)
        c_b = self.coord_blocks

        blocked_from = c_b[b_ind[0]].reshape((-1, bs, 1, 3))
        blocked_to = c_b[b_ind[1]].reshape((-1, 1, bs, 3))

        blocked_dist = (blocked_from - blocked_to).norm(dim=-1)

        dist = blocked_dist.reshape(-1)
        dist.register_hook(self.nan_to_num)

        return dist
