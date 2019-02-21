from typing import Dict

import attr
import torch
import numpy
import scipy.sparse.csgraph

from tmol.utility.reactive import reactive_property
from tmol.utility.mixins import gather_superclass_properies

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

from .score_graph import score_graph
from .stacked_system import StackedSystem


def _nan_to_num(var):
    vals = var.detach()
    zeros = torch.zeros(1, dtype=vals.dtype, layout=vals.layout, device=vals.device)
    return var.where(~torch.isnan(vals), zeros)


@score_graph
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


@score_graph
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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SphereDistance(TensorGroup):
    center_dist: Tensor(torch.float)[...]
    min_dist: Tensor(torch.float)[...]

    @classmethod
    def for_spheres(cls, a: Sphere, b: Sphere):
        center_dist = (a.center - b.center).norm(dim=-1)

        min_dist = center_dist - (a.radius + b.radius)

        return cls(center_dist=center_dist, min_dist=min_dist)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IntraLayerAtomPairs:
    inds: Tensor(torch.long)[:, 3]

    @classmethod
    def for_coord_blocks(
        cls, block_size: int, coord_blocks: Sphere, threshold_distance: float
    ):
        # Abbreviations used in indexing below
        # num_layers, num_blocks
        nl, nb = coord_blocks.shape
        bs: int = block_size

        interblock = SphereDistance.for_spheres(
            coord_blocks[:, :, None, None, None], coord_blocks[:, None, None, :, None]
        )
        assert interblock.shape == (nl, nb, 1, nb, 1)

        atom_pair_mask = interblock.min_dist.new_full(
            (nl, nb, bs, nb, bs), 0, dtype=torch.uint8
        )

        atom_pair_mask.masked_fill_(interblock.min_dist < threshold_distance, 1)
        atom_pair_mask = atom_pair_mask.reshape((nl, nb * bs, nb * bs))

        atom_pair_mask.masked_fill_(
            torch.ones_like(atom_pair_mask[0]).tril()[None, :, :], 0
        )

        return cls(atom_pair_mask.nonzero())


@attr.s(auto_attribs=True, frozen=True, slots=True)
class InterLayerAtomPairs:
    inds: Tensor(torch.long)[:, 4]

    @classmethod
    def for_coord_blocks(
        cls,
        atom_pair_block_size: int,
        coord_blocks_a: Sphere,
        coord_blocks_b: Sphere,
        interatomic_threshold_distance: float,
    ):
        # Abbreviations used in indexing below
        # num_layers_[ab], num_blocks_[ab]
        nla, nba = coord_blocks_a.shape
        nlb, nbb = coord_blocks_b.shape
        # block_size
        bs: int = atom_pair_block_size

        interblock = SphereDistance.for_spheres(
            coord_blocks_a[:, :, None, None, None, None],
            coord_blocks_b[None, None, None, :, :, None],
        )
        assert interblock.shape == (nla, nba, 1, nlb, nbb, 1)

        atom_pair_mask = interblock.min_dist.new_full(
            (nla, nba, bs, nlb, nbb, bs), 0, dtype=torch.uint8
        )
        atom_pair_mask.masked_fill_(
            interblock.min_dist < interatomic_threshold_distance, 1
        )
        atom_pair_mask = atom_pair_mask.reshape((nla, nba * bs, nlb, nbb * bs))

        return cls(torch.nonzero(atom_pair_mask))


@score_graph
class BlockedInteratomicDistanceGraph(InteratomicDistanceGraphBase):
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
        atom_pair_block_size: int,
        coord_blocks: Sphere,
        interatomic_threshold_distance: float,
    ) -> Tensor(torch.long)[:, 3]:
        """Triu atom pairs potentially within interaction threshold distance.

        [layer, atom_i, atom_i] index tensor for all triu (upper triangular)
        per-layer atom pairs.
        """
        return IntraLayerAtomPairs.for_coord_blocks(
            block_size=atom_pair_block_size,
            coord_blocks=coord_blocks,
            threshold_distance=interatomic_threshold_distance,
        ).inds
