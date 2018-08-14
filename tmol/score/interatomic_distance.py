from typing import Dict

from toolz.curried import compose, map
import attr
import torch
import numpy
import scipy.sparse.csgraph

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.utility.mixins import gather_superclass_properies

from tmol.types.torch import Tensor

from .bonded_atom import BondedAtomScoreGraph


def _nan_to_num(var):
    vals = var.detach()
    zeros = torch.zeros(
        1, dtype=vals.dtype, layout=vals.layout, device=vals.device
    )
    return var.where(~torch.isnan(vals), zeros)


@reactive_attrs(auto_attribs=True)
class InteratomicDistanceGraphBase(BondedAtomScoreGraph):
    """Base graph for interatomic distances.

    Graph component calculating interatomic distances. Distances are present *once* in the source
    graph for a given atomic pair; the rendered distances are equivalent to the upper triangle of
    the full interatomic distance matrix.

    Distances are rendered as two tensor properties, ``atom_pair_inds`` and ``atom_pair_dist``,
    containing the pair of atomic indicies in ``coords`` and the calculated distance respectively.

    Components requiring access to interatomic distance components *must* make
    the component's interatomic threshold distance available by implementing
    the ``component_atom_pair_dist_threshold`` property. The generated
    interatomic distance graph will respecte the *maximum* required interatomic
    distance of all score graph components.
    """

    def __attrs_post_init__(self):
        self.atom_pair_dist_thresholds = gather_superclass_properies(
            self, "component_atom_pair_dist_threshold"
        )

        if hasattr(super(), "__attrs_post_init__"):
            super().__attrs_post_init__()

    #interaction threshold distances that *may* be used to optimize distance pair selection
    atom_pair_dist_thresholds: Dict[str, float] = attr.ib(
        repr=False, init=False
    )

    @reactive_property
    def atom_pair_delta(
            coords: Tensor("f4")[:, 3],
            atom_pair_inds: Tensor(torch.long)[:, 2],
    ) -> Tensor("f4")[:, 3]:
        """inter-atomic pairwise distance within threshold distance"""
        delta = (coords[atom_pair_inds[0]] - coords[atom_pair_inds[1]])

        if delta.requires_grad:
            delta.register_hook(_nan_to_num)

        return delta

    @reactive_property
    def atom_pair_dist(
            atom_pair_delta: Tensor("f4")[:, 3],
    ) -> Tensor("f4")[:]:
        return atom_pair_delta.norm(dim=-1)

    def atom_pair_to_dense(self, atom_pair_term, null_value=numpy.nan):
        sp = scipy.sparse.coo_matrix(
            (atom_pair_term, tuple(self.atom_pair_inds)),
            shape=(self.system_size, self.system_size)
        ).tocsr()
        return scipy.sparse.csgraph.csgraph_to_dense(sp, null_value=null_value)


@reactive_attrs(auto_attribs=True)
class NaiveInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    @reactive_property
    def atom_pair_inds(system_size: int) -> Tensor(torch.long)[:, 2]:
        """Index pairs for all atom pairs."""

        return compose(
            torch.stack,
            tuple,
            map(torch.LongTensor),
        )(numpy.triu_indices(system_size, k=1))


@attr.s(slots=True, auto_attribs=True, frozen=True)
class BlockedDistanceAnalysis:
    #atom block size for block-neighbor optimization
    block_size: int = attr.ib()

    @block_size.validator
    def _valid_block_size(self, attribute, value):
        if value < 1 or value > 255:
            raise ValueError("Invalid block size.")

    # nan-to-num-ed system coordinates
    coords: Tensor("f4")[:, 3]

    # block mean coordinates`
    block_centers: Tensor(torch.float)[:, 3]
    # maximum block coordinates<-> mean distance
    block_radii: Tensor(torch.float)[:, 3]

    # triu indicies of the inter-block interaction matrix
    block_triu_inds: Tensor(torch.long)[:, 2]

    # inter-block mean coordinate interaction distances
    block_triu_dist: Tensor(torch.float)[:, 2]
    # inter-block minimum member coordinate interaction distance
    block_triu_min_dist: Tensor(torch.float)[:, 2]

    @classmethod
    def setup(
            cls,
            block_size: int,
            coords: Tensor(torch.float)[:, 3],
    ):
        assert not coords.requires_grad

        num_blocks, _remainder = divmod(coords.shape[0], block_size)
        assert _remainder == 0

        bs = block_size
        nb = num_blocks

        c_isnan = torch.isnan(coords)
        coords = coords.where(~c_isnan, coords.new_zeros(1))

        nonnan_atoms = (c_isnan.sum(dim=-1) == 0)
        atoms_per_block = (
            nonnan_atoms.reshape(nb, bs).sum(dim=-1).to(coords.dtype)
        )

        nonnan_blocks = atoms_per_block > 0

        block_triu_inds = compose(
            torch.stack,
            tuple,
            map(torch.LongTensor),
        )(numpy.triu_indices(num_blocks, k=1))

        block_centers = (
            coords.reshape((nb, bs, 3)).sum(dim=-2) /
            atoms_per_block.reshape((nb, 1)).expand((-1, 3))
        )

        block_radii = (
            (coords.reshape((nb, bs, 3)) - block_centers.reshape((nb, 1, 3)))
            .norm(dim=-1)
            .where(
                nonnan_atoms.reshape((nb, bs)),
                coords.new_zeros(1))
            .max(dim=1)[0]
        )

        block_triu_ind = (
            block_triu_inds[:,
                            (nonnan_blocks[block_triu_inds].sum(dim=0) == 2)
                            .nonzero()
                            .squeeze(dim=-1)]
        )

        block_triu_dist = (
                block_centers[block_triu_ind[0]] - block_centers[block_triu_ind[1]]
            ).norm(dim=-1)

        block_triu_min_dist = block_triu_dist - block_radii[block_triu_ind].sum(dim=0)


        return cls(
            coords=coords,
            block_size=block_size,
            block_centers=block_centers,
            block_radii=block_radii,
            block_triu_inds=block_triu_inds,
            block_triu_dist=block_triu_dist,
            block_triu_min_dist=block_triu_min_dist,
        )


@reactive_attrs(auto_attribs=True)
class BlockedInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    # atom block size for block-neighbor optimization
    atom_pair_block_size: int = attr.ib(8, converter=int)

    @atom_pair_block_size.validator
    def _valid_block_size(self, attribute, value):
        if value < 1 or value > 255:
            raise ValueError("Invalid block size.")

    @property
    def interatomic_threshold_distance(self):
        if self.atom_pair_dist_thresholds:
            return min(self.atom_pair_dist_thresholds.values())
        else:
            return numpy.inf

    @reactive_property
    def interblock_analysis(
            coords,
            atom_pair_block_size,
    ) -> BlockedDistanceAnalysis:
        """Current inter-block distance analysis."""
        return BlockedDistanceAnalysis.setup(
            atom_pair_block_size, coords.detach()
        )

    @reactive_property
    def atom_pair_inds(
            system_size: int,
            interblock_analysis: BlockedDistanceAnalysis,
            interatomic_threshold_distance: float,
    ) -> Tensor(torch.long)[:, 2]:
        """Indices for all atom pairs potentially within interaction threshold distance."""
        ba: BlockedDistanceAnalysis = interblock_analysis
        bs: int = interblock_analysis.block_size
        nb: int = system_size / interblock_analysis.block_size

        interblock_triu_matches = ba.block_triu_inds[:, (
               ba.block_triu_min_dist < interatomic_threshold_distance
           )
           .nonzero().squeeze(dim=-1)
        ]

        interblock_atom_pair_ind = ((
            (interblock_triu_matches.reshape((2, -1, 1, 1)) * bs) +
            torch.LongTensor(numpy.mgrid[:bs, :bs]).reshape((2, 1, bs, bs))
        ).reshape(2, -1))

        intrablock_triu = compose(
            torch.stack,
            tuple,
            map(torch.LongTensor),
        )(numpy.triu_indices(bs, k=1))

        intrablock_atom_pair_ind = ((
            intrablock_triu.reshape(2, 1, -1) +
            (torch.LongTensor(numpy.arange(nb)).reshape(1, nb, 1) * bs)
        ).reshape(2, -1))

        raw_atom_pairs = torch.cat(
            (intrablock_atom_pair_ind, interblock_atom_pair_ind), dim=-1
        )

        return raw_atom_pairs
