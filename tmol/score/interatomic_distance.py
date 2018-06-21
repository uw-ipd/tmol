from typing import Dict

import attr
import torch
import numpy
import scipy.sparse.csgraph

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.utility.mixins import gather_superclass_properies

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

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
class Sphere(TensorGroup):
    """Mean & radii for fixed size contiguous coordinate blocks."""

    center: Tensor(torch.float)[..., 3]
    radius: Tensor(torch.float)[...]

    @classmethod
    @validate_args
    def from_coord_blocks(cls, block_size: int, coords: Tensor(torch.float)[..., :, 3]):
        assert not coords.requires_grad

        num_blocks, _remainder = map(int, divmod(coords.shape[-2], block_size))
        assert _remainder == 0

        # The "broadcast shape" component
        brs = broadcast_shape = coords.shape[:-2]

        # coord shape w/ minor access
        blocked_shape = broadcast_shape + (num_blocks, block_size)

        nonnan_coords = torch.isnan(coords).sum(dim=-1) == 0
        coords = coords.where(nonnan_coords[..., None], coords.new_zeros(1))

        blocked_coords = coords.reshape(broadcast_shape + (num_blocks, block_size, 3))
        coords_per_block = (
            nonnan_coords.reshape(blocked_shape).sum(dim=-1).to(coords.dtype)
        )

        block_centers = blocked_coords.sum(dim=-2) / coords_per_block[..., None]

        block_radii = (
            (blocked_coords - block_centers.reshape(brs + (num_blocks, 1, 3)))
            .norm(dim=-1)
            .where(nonnan_coords.reshape(blocked_shape), coords.new_zeros(1))
            .max(dim=-1)[0]
        )

        return cls(center=block_centers, radius=block_radii)


# @attr.s(slots=True, auto_attribs=True, frozen=True)
# class BlockedDistanceAnalysis(ValidateAttrs):
#     """Sparse inter-coordinate-block mean and minimum distance analysis.

#     Inter-block mean and minimum distance analysis for the upper-triangular
#     component of the inter-block matrix. All values are over non-nan
#     source coordinates.

#     Fields:
#         block_triu_inds: indices of the inter-block interaction matrix for
#             non-nan blocks, ie. *may* be sparser than full triu indices
#         block_triu_center_dist: inter-mean distance for block_triu_inds
#         block_triu_min_dist: min inter-block coordinate distance, calculated by
#             block-mean & block-radii triangle inequality
#     """
#     block_triu_inds: Tensor(torch.long)[:, 2]
#     block_triu_center_dist: Tensor(torch.float)[:, :]
#     block_triu_min_dist: Tensor(torch.float)[:, :]

#     def pdist(cls, blocks: Sphere):

#         return cls(
#             n=blocks.shape[-1],
#             m=blocks.shape[-1],
#             block_triu_inds=block_triu_inds,
#             block_triu_center_dist=block_triu_center_dist,
#             block_triu_min_dist=block_triu_min_dist,
#         )


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
    def coord_blocks(
        atom_pair_block_size: int, coords: Tensor(torch.float)[:, :, 3]
    ) -> Sphere:
        return Sphere.from_coord_blocks(
            block_size=atom_pair_block_size, coords=coords.detach()
        )

    @reactive_property
    def atom_pair_inds(
        stack_depth: int,
        system_size: int,
        atom_pair_block_size: int,
        coord_blocks: Sphere,
        interatomic_threshold_distance: float,
    ) -> Tensor(torch.long)[:, 3]:
        """Triu atom pairs potentially within interaction threshold distance.

        [layer, atom_i, atom_i] index tensor for all triu (upper triangular)
        per-layer atom pairs.
        """

        # Localize to current device of input analysis
        device = coord_blocks.center.device
        new_tensor = coord_blocks.center.new_tensor

        block_triu_inds = triu_indices(coord_blocks.shape[-1], k=1).to(
            device=coord_blocks.center.device
        )

        block_triu_center_dist = (
            coord_blocks.center[..., block_triu_inds[:, 0], :]
            - coord_blocks.center[..., block_triu_inds[:, 1], :]
        ).norm(dim=-1)

        block_triu_min_dist = block_triu_center_dist - coord_blocks.radius[
            ..., block_triu_inds
        ].sum(dim=-1)

        # Abbreviations used in indexing below
        bs: int = atom_pair_block_size
        nb: int = int(system_size / atom_pair_block_size)

        # Convert the triu block-min-dist table to coindexed index tensors
        # of layer index and block pair index (within the dense triu list)
        #
        # Drop the layer index for now, it will be rejoined back on after
        # expanding block indices into atom indices.
        interblock_match_indices: Tensor(bool)[:, 2] = (
            block_triu_min_dist < interatomic_threshold_distance
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
        first_atom_indices = block_triu_inds[interblock_pair_index] * bs  # Tensor[:, 2]

        # mgrid [2, bs, bs] -> transpose to [bs, bs, 2] -> ravel [bs*bs, 2]
        atom_index_offsets = new_tensor(
            numpy.mgrid[:bs, :bs].T, dtype=torch.long
        ).reshape(-1, 2)

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
