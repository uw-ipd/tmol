from typing import Dict

import attr
import torch
import numpy
import scipy.sparse.csgraph

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.utility.mixins import gather_superclass_properies

from tmol.types.functional import validate_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor

from .stacked_system import StackedSystem
from .factory import Factory


def _nan_to_num(var):
    vals = var.detach()
    zeros = torch.zeros(1, dtype=vals.dtype, layout=vals.layout, device=vals.device)
    return var.where(~torch.isnan(vals), zeros)


@reactive_attrs(auto_attribs=True)
class InteratomicDistanceGraphBase(StackedSystem):
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
    interatomic distance graph will respect the *maximum* required interatomic
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
    @validate_args
    def atom_pair_delta(
        coords: Tensor("f4")[:, :, 3], atom_pair_inds: Tensor(torch.long)[:, 3]
    ) -> Tensor("f4")[:, 3]:
        """inter-atomic pairwise distance within threshold distance"""
        delta = (
            coords[atom_pair_inds[:, 0], atom_pair_inds[:, 1]]
            - coords[atom_pair_inds[:, 0], atom_pair_inds[:, 2]]
        )

        if delta.requires_grad:
            delta.register_hook(_nan_to_num)

        return delta

    @reactive_property
    @validate_args
    def atom_pair_dist(atom_pair_delta: Tensor("f4")[:, 3],) -> Tensor("f4")[:]:
        return atom_pair_delta.norm(dim=-1)

    def atom_pair_to_dense(self, atom_pair_term, null_value=numpy.nan):
        sp = scipy.sparse.coo_matrix(
            (atom_pair_term, tuple(self.atom_pair_inds)),
            shape=(self.system_size, self.system_size),
        ).tocsr()

        return scipy.sparse.csgraph.csgraph_to_dense(sp, null_value=null_value)


@validate_args
def triu_indices(n, k=0, m=None) -> Tensor(torch.long)[:, 2]:
    """Repacked triu_indices, see numpy.triu_indices for details."""
    i, j = numpy.triu_indices(n, k, m)
    return torch.stack((torch.from_numpy(i), torch.from_numpy(j)), dim=-1)


@reactive_attrs(auto_attribs=True)
class NaiveInteratomicDistanceGraph(InteratomicDistanceGraphBase):
    @reactive_property
    @validate_args
    def atom_pair_inds(
        stack_depth: int, system_size: int, device: torch.device
    ) -> Tensor(torch.long)[:, 3]:
        """Index pairs for all atom pairs."""

        layer_inds = torch.arange(stack_depth, device=device, dtype=torch.long)
        per_layer_inds = triu_indices(system_size, k=1).to(device)
        npair = per_layer_inds.shape[0]

        return torch.cat(
            (
                layer_inds[:, None, None].expand(-1, npair, 1),
                per_layer_inds[None, :, :].expand(stack_depth, -1, 2),
            ),
            dim=-1,
        ).reshape(-1, 3)


@attr.s(slots=True, auto_attribs=True, frozen=True)
class BlockedDistanceAnalysis(ValidateAttrs):
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

    coords: Tensor("f4")[:, :, 3]

    block_centers: Tensor(torch.float)[:, :, 3]
    block_radii: Tensor(torch.float)[:, :]

    block_triu_inds: Tensor(torch.long)[:, 2]
    block_triu_center_dist: Tensor(torch.float)[:, :]
    block_triu_min_dist: Tensor(torch.float)[:, :]

    @classmethod
    @validate_args
    def setup(cls, block_size: int, coords: Tensor(torch.float)[:, :, 3]):
        assert not coords.requires_grad

        num_blocks, _remainder = divmod(coords.shape[1], block_size)
        assert _remainder == 0

        nl = coords.shape[0]
        bs = block_size
        nb = int(num_blocks)

        c_isnan = torch.isnan(coords)
        coords = coords.where(~c_isnan, coords.new_zeros(1))

        nonnan_atoms = c_isnan.sum(dim=-1) == 0
        atoms_per_block = nonnan_atoms.reshape(nl, nb, bs).sum(dim=-1).to(coords.dtype)

        block_centers = coords.reshape((nl, nb, bs, 3)).sum(
            dim=-2
        ) / atoms_per_block.reshape((nl, nb, 1)).expand((-1, -1, 3))

        # yapf: enable
        block_radii = (
            (coords.reshape((nl, nb, bs, 3)) - block_centers.reshape((nl, nb, 1, 3)))
            .norm(dim=-1)
            .where(nonnan_atoms.reshape((nl, nb, bs)), coords.new_zeros(1))
            .max(dim=-1)[0]
        )

        block_triu_inds = triu_indices(nb, k=1).to(device=coords.device)
        block_triu_center_dist = (
            block_centers[:, block_triu_inds[:, 0]]
            - block_centers[:, block_triu_inds[:, 1]]
        ).norm(dim=-1)

        block_triu_min_dist = block_triu_center_dist - block_radii[
            :, block_triu_inds
        ].sum(dim=-1)

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
class BlockedInteratomicDistanceGraph(InteratomicDistanceGraphBase, Factory):
    # atom block size for block-neighbor optimization
    atom_pair_block_size: int = attr.ib()

    @atom_pair_block_size.validator
    def _valid_block_size(self, attribute, value):
        if value < 1 or value > 255:
            raise ValueError("Invalid block size.")

    def factory_for(obj, **_):
        return dict(atom_pair_block_size=8)

    @property
    def interatomic_threshold_distance(self):
        if self.atom_pair_dist_thresholds:
            return min(self.atom_pair_dist_thresholds.values())
        else:
            return numpy.inf

    @reactive_property
    @validate_args
    def interblock_analysis(
        coords: Tensor(torch.float)[:, :, 3], atom_pair_block_size: int
    ) -> BlockedDistanceAnalysis:
        """Current inter-block distance analysis."""
        return BlockedDistanceAnalysis.setup(atom_pair_block_size, coords.detach())

    @reactive_property
    def atom_pair_inds(
        stack_depth: int,
        system_size: int,
        interblock_analysis: BlockedDistanceAnalysis,
        interatomic_threshold_distance: float,
    ) -> Tensor(torch.long)[:, 3]:
        """Triu atom pairs potentially within interaction threshold distance.

        [layer, atom_i, atom_i] index tensor for all triu (upper triangular)
        per-layer atom pairs.
        """

        # Localize to current device of input analysis
        device = interblock_analysis.block_triu_inds.device
        new_tensor = interblock_analysis.block_triu_inds.new_tensor

        # Abbreviations used in indexing below
        ba: BlockedDistanceAnalysis = interblock_analysis
        bs: int = interblock_analysis.block_size
        nb: int = int(system_size / interblock_analysis.block_size)

        # Convert the triu block-min-dist table to coindexed index tensors
        # of layer index and block pair index (within the dense triu list)
        #
        # Drop the layer index for now, it will be rejoined back on after
        # expanding block indices into atom indices.
        interblock_match_indices: Tensor(bool)[:, 2] = (
            ba.block_triu_min_dist < interatomic_threshold_distance
        ).nonzero()
        interblock_layer_index = interblock_match_indices[:, 0]
        interblock_pair_index = interblock_match_indices[:, 1]
        n = len(interblock_match_indices)

        # Convert the [n] block pair indices in dense atom index tensors.
        #
        # 1) Convert block pair indices into [n, 2] (from_block, to_block)
        #    index tensor and then convert to first-atom-in-block index.
        # 2) Generate [bs, bs] atom index offset meshgrid within a block pair
        # 3) Sum first-atom-index and atom offsets into atom index block
        first_atom_indices = (  # Tensor[:, 2]
            ba.block_triu_inds[interblock_pair_index] * bs
        )

        # mgrid [2, bs, bs] -> transpose to [bs, bs, 2] -> ravel [bs*bs, 2]
        atom_index_offsets = new_tensor(numpy.mgrid[:bs, :bs].T).reshape(-1, 2)

        interblock_atom_pair_ind_blocks = (  # [n, bs * bs, 2]
            first_atom_indices[:, None, :] + atom_index_offsets[None, :, :]
        )

        # Stack into [n, bs*bs, 3]
        # then ravel to [n*bs*bs, 3] (layer, from_a, to_a) index tensor
        interblock_layer_ind_blocks = interblock_layer_index[:, None, None].expand(
            -1, bs * bs, 1
        )
        interblock_dense_ind = torch.cat(
            (interblock_layer_ind_blocks, interblock_atom_pair_ind_blocks), dim=-1
        ).reshape(n * bs * bs, 3)

        # Calculate the intra-block triu atom pair indices
        #
        # Similarly, generate first-atom-in-block index for all blocks
        # Generate the intrablock triu index offsets and sum, then ravel
        # into atom pair index tensor
        block_start_ind = torch.arange(nb, dtype=torch.long, device=device) * bs
        stack_ind = torch.arange(stack_depth, dtype=torch.long, device=device)
        m = int(bs * (bs - 1) / 2)

        intrablock_atom_pair_ind = (
            (
                block_start_ind[:, None, None]
                + triu_indices(bs, k=1).to(device=device)[None, :, :]  # [nb]  # [m, 2]
            )
            .reshape(1, nb * m, 2)
            .expand(stack_depth, -1, 2)
        )
        intrablock_layer_ind = stack_ind[:, None, None].expand(-1, nb * m, 1)
        intrablock_dense_ind = torch.cat(
            (intrablock_layer_ind, intrablock_atom_pair_ind), dim=-1
        ).reshape(stack_depth * nb * m, 3)

        raw_atom_pairs = torch.cat((interblock_dense_ind, intrablock_dense_ind), dim=0)

        return raw_atom_pairs
