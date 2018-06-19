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
    zeros = torch.zeros(1, dtype=vals.dtype, layout=vals.layout, device=vals.device)
    return var.where(~torch.isnan(vals), zeros)


@reactive_attrs(auto_attribs=True)
class InteratomicDistanceGraphBase(BondedAtomScoreGraph):
    """Base graph for interatomic distances.

    Graph component calculating interatomic distances. Distances are present
    *once* in the source graph for a given atomic pair; the rendered distances
    are equivalent to the upper triangle of the full interatomic distance
    matrix.

    Distances are rendered as two tensor properties, ``atom_pair_inds`` and
    ``atom_pair_dist``, containing the pair of atomic indicies in ``coords``
    and the calculated distance respectively.

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

    # interaction threshold distances that *may* be used to optimize distance
    # pair selection
    atom_pair_dist_thresholds: Dict[str, float] = attr.ib(repr=False, init=False)

    @reactive_property
    def atom_pair_delta(
        coords: Tensor("f4")[:, 3], atom_pair_inds: Tensor(torch.long)[:, 2]
    ) -> Tensor("f4")[:, 3]:
        """inter-atomic pairwise distance within threshold distance"""
        delta = coords[atom_pair_inds[0]] - coords[atom_pair_inds[1]]

        if delta.requires_grad:
            delta.register_hook(_nan_to_num)

        return delta

    @reactive_property
    def atom_pair_dist(atom_pair_delta: Tensor("f4")[:, 3],) -> Tensor("f4")[:]:
        return atom_pair_delta.norm(dim=-1)

    def atom_pair_to_dense(self, atom_pair_term, null_value=numpy.nan):
        sp = scipy.sparse.coo_matrix(
            (atom_pair_term, tuple(self.atom_pair_inds)),
            shape=(self.system_size, self.system_size),
        ).tocsr()
        return scipy.sparse.csgraph.csgraph_to_dense(sp, null_value=null_value)


def triu_indices(n, k=0, m=None) -> Tensor(torch.long)[:, 2]:
    """Repacked triu_indices, see numpy.triu_indices for details."""
    return compose(torch.stack, tuple, map(torch.from_numpy))(
        numpy.triu_indices(n, k, m)
    )


@reactive_attrs(auto_attribs=True)
class NaiveInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    @reactive_property
    def atom_pair_inds(
        system_size: int, device: torch.device
    ) -> Tensor(torch.long)[:, 2]:
        """Index pairs for all atom pairs."""

        return triu_indices(system_size, k=1).to(device)


@attr.s(slots=True, auto_attribs=True, frozen=True)
class BlockedDistanceAnalysis:
    """Sparse inter-coordinate-block mean and minimum distance analysis.

    Inter-block mean and minimum distance analysis for the upper-triangular
    component of the inter-block matrix, sparsified to only contain entries for
    which both blocks contain non-nan coordinates. All values are over non-nan
    source coordinates.

    Fields:
        block-size: atom block size for block-neighbor optimization
        coords: nan-to-num-ed system coordinate buffer
        block_centers: mean block coordinate
        block_radii: max coord<->center distance over block
        block_triu_inds: indices of the inter-block interaction matrix for
            non-nan blocks, ie. *may* be sparser than full triu indices
        block_triu_center_dist: inter-mean distance for block_triu_inds
        block_triu_min_dist: min inter-block coordinate distance, calculated by
            block-mean & block-radii triangle inequality
    """

    block_size: int = attr.ib()

    @block_size.validator
    def _valid_block_size(self, attribute, value):
        if value < 1 or value > 255:
            raise ValueError("Invalid block size.")

    coords: Tensor("f4")[:, 3]

    block_centers: Tensor(torch.float)[:, 3]
    block_radii: Tensor(torch.float)[:, 3]

    block_triu_inds: Tensor(torch.long)[:, 2]
    block_triu_center_dist: Tensor(torch.float)[:, 2]
    block_triu_min_dist: Tensor(torch.float)[:, 2]

    @classmethod
    def setup(cls, block_size: int, coords: Tensor(torch.float)[:, 3]):
        assert not coords.requires_grad

        num_blocks, _remainder = divmod(coords.shape[0], block_size)
        assert _remainder == 0

        bs = block_size
        nb = num_blocks

        c_isnan = torch.isnan(coords)
        coords = coords.where(~c_isnan, coords.new_zeros(1))

        nonnan_atoms = c_isnan.sum(dim=-1) == 0
        atoms_per_block = nonnan_atoms.reshape(nb, bs).sum(dim=-1).to(coords.dtype)

        block_centers = coords.reshape((nb, bs, 3)).sum(
            dim=-2
        ) / atoms_per_block.reshape((nb, 1)).expand((-1, 3))

        block_radii = (
            (coords.reshape((nb, bs, 3)) - block_centers.reshape((nb, 1, 3)))
            .norm(dim=-1)
            .where(nonnan_atoms.reshape((nb, bs)), coords.new_zeros(1))
            .max(dim=1)[0]
        )

        nonnan_block_ind = (atoms_per_block > 0).nonzero()[:, -1]
        block_triu_inds = nonnan_block_ind[triu_indices(len(nonnan_block_ind), k=1),]
        block_triu_center_dist = (
            block_centers[block_triu_inds[0]] - block_centers[block_triu_inds[1]]
        ).norm(dim=-1)

        block_triu_min_dist = block_triu_center_dist - block_radii[block_triu_inds].sum(
            dim=0
        )

        return cls(
            coords=coords,
            block_size=block_size,
            block_centers=block_centers,
            block_radii=block_radii,
            block_triu_inds=block_triu_inds,
            block_triu_center_dist=block_triu_center_dist,
            block_triu_min_dist=block_triu_min_dist,
        )

    @property
    def dense_min_dist(self) -> Tensor(torch.float)[:, :]:
        """[n,n] dense minimum distance matrix."""
        result = self.block_triu_min_dist.new_empty(
            (len(self.block_centers), len(self.block_centers))
        )

        dia_ind = torch.arange(len(self.block_centers), dtype=torch.long)
        result[dia_ind, dia_ind] = torch.where(
            torch.isnan(self.block_centers).sum(dim=-1) == 0,
            result.new_full((1,), 0),
            result.new_full((1,), numpy.nan),
        )

        result[
            self.block_triu_inds[0], self.block_triu_inds[1]
        ] = self.block_triu_min_dist
        result[
            self.block_triu_inds[1], self.block_triu_inds[0]
        ] = self.block_triu_min_dist

        return result


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
    def interblock_analysis(coords, atom_pair_block_size) -> BlockedDistanceAnalysis:
        """Current inter-block distance analysis."""
        return BlockedDistanceAnalysis.setup(atom_pair_block_size, coords.detach())

    @reactive_property
    def atom_pair_inds(
        system_size: int,
        interblock_analysis: BlockedDistanceAnalysis,
        interatomic_threshold_distance: float,
    ) -> Tensor(torch.long)[:, 2]:
        """Indices for atom pairs potentially within interaction threshold distance."""
        # Localize to current device of input analysis
        device = interblock_analysis.block_triu_inds.device
        new_tensor = interblock_analysis.block_triu_inds.new_tensor

        ba: BlockedDistanceAnalysis = interblock_analysis
        bs: int = interblock_analysis.block_size
        nb: int = system_size / interblock_analysis.block_size

        interblock_triu_matches = ba.block_triu_inds[
            :,
            ((ba.block_triu_min_dist < interatomic_threshold_distance))
            .nonzero()
            .squeeze(dim=-1),
        ]

        interblock_atom_pair_ind = (
            (interblock_triu_matches.reshape((2, -1, 1, 1)) * bs)
            + new_tensor(numpy.mgrid[:bs, :bs]).reshape((2, 1, bs, bs))
        ).reshape(2, -1)

        intrablock_triu = triu_indices(bs, k=1).to(device=device)
        block_start_ind = torch.arange(nb, dtype=torch.long, device=device) * bs
        intrablock_atom_pair_ind = (
            intrablock_triu.reshape(2, 1, -1) + block_start_ind.reshape(1, nb, 1)
        ).reshape(2, -1)

        raw_atom_pairs = torch.cat(
            (intrablock_atom_pair_ind, interblock_atom_pair_ind), dim=-1
        )

        return raw_atom_pairs
