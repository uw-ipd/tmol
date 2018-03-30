import properties
import torch
import numpy
import scipy.sparse.csgraph

from tmol.properties.reactive import derived_from
from tmol.properties.array import VariableT

from .bonded_atom import BondedAtomScoreGraph

from .types import RealTensor

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

    @derived_from(
        ("coords", "atom_pair_inds"),
        VariableT("inter-atomic pairwise distance within threshold distance"))
    def atom_pair_dist(self):
        dist = (
            self.coords[self.atom_pair_inds[0]] - self.coords[self.atom_pair_inds[1]]
        ).norm(dim=-1)

        if dist.requires_grad:
            dist.register_hook(self.nan_to_num)

        return dist


    def atom_pair_to_dense(self, atom_pair_term, null_value=numpy.nan):
        sp = scipy.sparse.coo_matrix(
            (atom_pair_term, tuple(self.atom_pair_inds)),
            shape=(self.system_size, self.system_size)).tocsr()
        return scipy.sparse.csgraph.csgraph_to_dense(sp, null_value=null_value)

class NaiveInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    @derived_from(
        ("system_size"),
        VariableT("index pairs for all atom pairs"))
    def atom_pair_inds(self):
        return torch.stack(tuple(map(torch.LongTensor,
            numpy.triu_indices(self.system_size, k=1))))

class BlockedInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    atom_pair_block_size = properties.Integer(
        "atom block size for block-neighbor optimization",
        min=1, max=255, cast=True, default=8)

    @property
    def interatomic_threshold_distance(self):
        if self.atom_pair_dist_thresholds:
            return min(self.atom_pair_dist_thresholds)
        else:
            return numpy.inf

    @derived_from(("coords", "atom_pair_block_size"),
        VariableT("atomic coordinate blocks"))
    def coord_blocks(self):
        return self.coords.reshape(-1, self.atom_pair_block_size, 3)

    @derived_from(("system_size", "atom_pair_block_size"),
        VariableT("a"))
    def block_triu_indices(self):
        return torch.stack(tuple(map(torch.LongTensor,
            numpy.triu_indices(self.system_size / self.atom_pair_block_size, k=1))))

    @derived_from(("system_size", "atom_pair_block_size"),
        VariableT("a"))
    def block_tril_indices(self):
        return torch.stack(tuple(map(torch.LongTensor,
            numpy.tril_indices(self.system_size / self.atom_pair_block_size, k=1))))

    @derived_from(
        ("coords", "atom_pair_block_size", "atom_pair_distance_thresholds"),
        VariableT("index pairs for all pairs potentially within interaction distance threshold"))
    def atom_pair_inds(self):
        bs = self.atom_pair_block_size
        nb = int(self.coords.shape[0] / bs)

        raw_coords = self.coords.detach()

        c_isnan = torch.isnan(raw_coords)
        coords = raw_coords.where(~c_isnan, RealTensor([0]))

        nonnan_atoms = (c_isnan.sum(dim=-1) == 0)
        atoms_per_block = nonnan_atoms.reshape(nb, bs).sum(dim=-1).type(RealTensor)
        nonnan_blocks = atoms_per_block > 0

        block_centers = coords.reshape((nb, bs, 3)).sum(dim=-2) / atoms_per_block.reshape((nb, 1)).expand((-1, 3))
        block_radii = (
            (coords.reshape((nb, bs, 3)) - block_centers.reshape((nb, 1, 3)))
                .norm(dim=-1)
                .where(nonnan_atoms.reshape((nb, bs)), RealTensor([0]))
                .max(dim=1)[0]
        )

        interblock_triu_ind = self.block_triu_indices[:,
             (nonnan_blocks[self.block_triu_indices].sum(dim=0) == 2).nonzero().squeeze(dim=-1)]

        interblock_triu_dist = (block_centers[interblock_triu_ind[0]] - block_centers[interblock_triu_ind[1]]).norm(dim=-1)
        interblock_triu_min_atomic_dist = interblock_triu_dist - block_radii[interblock_triu_ind].sum(dim=0)

        interblock_triu_matches = interblock_triu_ind[:,
              (interblock_triu_min_atomic_dist < self.interatomic_threshold_distance).nonzero().squeeze(dim=-1)]

        interblock_atom_pair_ind = (
            ((interblock_triu_matches.reshape((2, -1, 1, 1)) * bs) +
             torch.LongTensor(numpy.mgrid[:bs,:bs]).reshape((2, 1, bs, bs)))
            .reshape(2, -1)
        )

        intrablock_triu = torch.stack(tuple(map(torch.LongTensor, numpy.triu_indices(bs, k=1))))
        intrablock_atom_pair_ind = (
            (intrablock_triu.reshape(2, 1, -1) +
             (torch.LongTensor(numpy.arange(nb)).reshape(1, nb, 1) * bs))
            .reshape(2, -1)
        )

        raw_atom_pairs = torch.cat((intrablock_atom_pair_ind, interblock_atom_pair_ind), dim=-1)

        return raw_atom_pairs
