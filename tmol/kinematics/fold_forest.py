import enum

import attr
import numpy
import torch

from tmol.pose.pose_stack import PoseStack
from tmol.types.array import NDArray


class EdgeType(enum.IntEnum):
    polymer = 0
    jump = enum.auto()
    root_jump = enum.auto()


@attr.s(auto_attribs=True, frozen=True)
class FoldForest:
    """The fold forest will define the fold trees for the poses in a PoseStack.
    Each tensor in the class has its first dimension over the number of poses.

    The primary definition of a FoldTree is the Edge. The Edge defines a connection
    between two parts of a Pose. The three types of edges are 1. polymer edges
    (analgous to the previously named "peptide edges" from Rosetta++ and Rosetta3), 2.
    jump edges which connect any pair of residues in the Pose, and 3. root-jump
    edges, which originate at the explicit virtual root and connect to a particular
    residue. A polymer edge spans a contiguous range of polymeric block types where
    the "up" connection of residue i is connected to the "down" connection of residue
    i+1 for all i in the range between the "start" and "end" blocks.

    Each edge is described by a 4-tuple of integers (type, start, end, jump-index);
    where type is one of the EdgeType enum values, start is the index of the upstream
    residue of the edge, end is the index of the downstream residue of the edge, and
    jump-index is used to assign an id to any particular jump edge; jump-edge indices
    must be unique and ascending from 0 to n_jumps-1. "Root jump" edges take their
    "identity" from the downstream residue of the edge, so they do not need an index.

    The FoldForest in tmol differs from the FoldTree in Rosetta3 in that there
    is always a virtual root at the origin and any residue (block) may be
    connected to this root by a "root jump". Such root-jump residues are defined
    by listing the residue that the root is connected to as the "end" residue;
    the "start" residue field should be left as -1. An example FoldForest for a
    ten-residue protein might be:
      (polymer, 0, 4)
      (jump   , 0, 7)
      (polymer, 7, 9)
      (polymer, 7, 6)
      (root-jump, -1, 0)
      (root-jump, -1, 5)
    where both residues are 0 and 5 are connected to the root.

    Note that in the MoveMap, the root-jumps are distinct from the non-root-jumps.
    """

    max_n_edges: int
    n_edges: NDArray[int][:]
    edges: NDArray[int][:, :, 4]

    @classmethod
    def reasonable_fold_forest(cls, pose_stack: PoseStack):
        """Create a fold tree consisting of N->C edges for each chain
        for each Pose in the input PoseStack and a root-jump to each
        chain's N-terminus.
        """
        # one edge from the root to each chain's first residue
        # and one n->c polymer edge for each chain
        pose_n_residues = torch.sum(pose_stack.block_type_ind != -1, dim=1)

        n_chains_per_pose = torch.max(pose_stack.chain_id64, dim=1)[0] + 1
        max_n_chains = torch.max(n_chains_per_pose).cpu().item()
        max_n_edges = 2 * max_n_chains
        is_pci_chain_real = torch.arange(max_n_chains, dtype=torch.int64, device=pose_stack.device).unsqueeze(0) < (
            n_chains_per_pose.unsqueeze(1)
        )
        real_chain_indices = torch.nonzero(is_pci_chain_real, as_tuple=False)

        edges = torch.full(
            (pose_stack.n_poses, max_n_edges, 4),
            -1,
            dtype=torch.int64,
            device=pose_stack.device,
        )
        n_edges = torch.zeros((pose_stack.n_poses,), dtype=int, device=pose_stack.device)

        chain_boundaries = torch.zeros(
            (pose_stack.n_poses, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        chain_boundaries[:, :-1] = pose_stack.chain_id64[:, 1:] != pose_stack.chain_id64[:, :-1]
        # last residue is end of a chain; but will not be marked as such if n_res == max_n_res
        chain_boundaries[
            torch.arange(pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device),
            pose_n_residues - 1,
        ] = True

        # the index for each chain's last residue
        chain_end_indices_ci = torch.nonzero(chain_boundaries, as_tuple=False)

        # now lets construct a pair of tensors of n_poses x max_n_chains
        # with the beginning and ending residue indices for each chain
        chain_end_indices_pci = torch.zeros(
            (pose_stack.n_poses, max_n_chains),
            dtype=torch.int64,
            device=pose_stack.device,
        )
        chain_end_indices_pci[is_pci_chain_real] = chain_end_indices_ci[:, 1]
        chain_begin_indices_pci = torch.zeros_like(chain_end_indices_pci)
        real_pci_chains = torch.nonzero(is_pci_chain_real, as_tuple=False)

        # chain_begin_indices_pci:
        # the index of the first residue in each chain.
        # This is just one greater than the last residue of the previous chain,
        # except, we have to overwrite the index of the first chain in each
        # pose to be zero.
        chain_begin_indices_pci[real_pci_chains[1:, 0], real_pci_chains[1:, 1]] = chain_end_indices_ci[:-1, 1] + 1
        chain_begin_indices_pci[:, 0] = 0

        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1], 0] = EdgeType.polymer
        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1], 1] = chain_begin_indices_pci[
            real_chain_indices[:, 0], real_chain_indices[:, 1]
        ]
        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1], 2] = chain_end_indices_pci[
            real_chain_indices[:, 0], real_chain_indices[:, 1]
        ]

        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1] + 1, 0] = EdgeType.root_jump
        # edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1]+ 1, 0] == -1 already
        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1] + 1, 2] = chain_begin_indices_pci[
            real_chain_indices[:, 0], real_chain_indices[:, 1]
        ]
        # jump number for root-jumps
        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1] + 1, 3] = (
            torch.arange(max_n_chains, dtype=torch.int64, device=pose_stack.device)
            .unsqueeze(0)
            .expand(pose_stack.n_poses, max_n_chains)[is_pci_chain_real]
        )
        edges[real_chain_indices[:, 0], 2 * real_chain_indices[:, 1] + 1, 3] = real_chain_indices[:, 1]
        edges = edges.cpu().numpy()
        n_edges = (2 * n_chains_per_pose).cpu().numpy()

        return cls(
            max_n_edges=max_n_edges,
            n_edges=n_edges,
            edges=edges,
        )

    @classmethod
    # soon! @validate_args
    def polymeric_forest(cls, n_res_per_tree: NDArray[numpy.int32][:]):
        """Create an N->C fold tree for a collection of monomers in a PoseStack.

        This will define a fold tree for each pose with a just one edge, and thus
        will be a very bad fold tree except for the not-so-common case that all
        of the Poses in the PoseStack are monomers.
        """
        n_trees = n_res_per_tree.shape[0]

        edges = numpy.full((n_trees, 2, 4), -1, dtype=int)
        edges[:, 0, 0] = EdgeType.root_jump
        edges[:, 0, 1] = -1
        edges[:, 0, 2] = 0
        n_res_gt_1 = n_res_per_tree > 1
        edges[n_res_gt_1, 1, 0] = EdgeType.polymer
        edges[n_res_gt_1, 1, 1] = 0
        edges[n_res_gt_1, 1, 2] = (n_res_per_tree[n_res_gt_1] - 1).astype(int)
        n_edges = numpy.full(n_trees, 1, dtype=int)
        n_edges[n_res_gt_1] = 2
        return cls(
            max_n_edges=2,
            n_edges=n_edges,
            edges=edges,
        )

    @classmethod
    def from_edges(cls, edges: NDArray[int][:, :, 4]):
        return cls(
            max_n_edges=edges.shape[1],
            n_edges=numpy.sum(edges[:, :, 0] != -1, axis=1),
            edges=edges,
        )
