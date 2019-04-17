import enum
import math

import attr

import torch
import pandas

from tmol.types.attrs import ConvertAttrs

from tmol.types.tensor import TensorGroup
from tmol.types.torch import Tensor

from tmol.utility.categorical import vals_to_name_cat, names_to_val_cat

from tmol.kinematics.datatypes import NodeType, KinDOF, KinTree


class DOFTypes(enum.IntEnum):
    """High-level class of kinematic DOF types."""

    jump = 0
    bond_angle = enum.auto()
    bond_distance = enum.auto()
    bond_torsion = enum.auto()


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DOFMetadata(TensorGroup, ConvertAttrs):
    """The location, type, and descriptive ids of valid dofs within a KinTree.

    Descriptive entries for dofs within a KinTree, this provides a 1-d
    structure to select and report a subset of entries within a KinDOF buffer.
    DOFMetadata sets are used to indicate mobile vs fixed dofs for KinematicOp
    dof to coordinate functions.

    DOFMetadata supports isomorphic conversion between a DataFrame and
    TensorGroup representation to support symbolic selection. This converts the
    IntEnum encoded "dof_type" entry into a string categorical column.
    """

    node_idx: Tensor(torch.long)[...]
    dof_idx: Tensor(torch.long)[...]

    dof_type: Tensor(torch.long)[...]
    parent_id: Tensor(torch.long)[...]
    child_id: Tensor(torch.long)[...]

    @classmethod
    def for_kintree(cls, kintree: KinTree):
        """Return all valid dofs within a KinTree."""

        # Setup a dof type table the same shape as the kinematic dofs,
        # marking all potential movable dofs with the abstract dof type.
        # Leaving all non-movable or invalid dofs as nan.
        parentIdx = kintree.parent.to(dtype=torch.long)
        dof_types = KinDOF.full(kintree.shape, math.nan)
        node_has_children = (
            torch.zeros_like(kintree.id).put_(
                parentIdx, torch.ones_like(kintree.parent), True
            )
            > 0
        )

        bsel = kintree.doftype == NodeType.bond

        dof_types.bond.phi_p[bsel] = DOFTypes.bond_angle
        dof_types.bond.theta[bsel] = DOFTypes.bond_angle
        dof_types.bond.d[bsel] = DOFTypes.bond_distance
        # Only flag the "child phi" if the node has affected children.
        dof_types.bond.phi_c[bsel & node_has_children] = DOFTypes.bond_torsion

        jsel = kintree.doftype == NodeType.jump
        dof_types.jump.RBx[jsel] = DOFTypes.jump
        dof_types.jump.RBy[jsel] = DOFTypes.jump
        dof_types.jump.RBz[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_alpha[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_beta[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_gamma[jsel] = DOFTypes.jump

        # Get indices of all marked dofs.
        valid_dofs = (~torch.isnan(dof_types.raw)).nonzero()
        node_idx = valid_dofs[:, 0]
        dof_idx = valid_dofs[:, 1]

        # Unpack into the node/dof index, then expand
        return cls(
            node_idx=node_idx,
            dof_idx=dof_idx,
            dof_type=dof_types.raw[node_idx, dof_idx],
            child_id=kintree.id[node_idx],
            parent_id=kintree.id[parentIdx[node_idx]],
        )

    def to_frame(self) -> pandas.DataFrame:
        assert len(self.shape) == 1

        columns = attr.asdict(self)
        columns["dof_type"] = vals_to_name_cat(DOFTypes, columns["dof_type"])
        return pandas.DataFrame(columns)

    @classmethod
    def from_frame(cls, frame):
        """Convert from DataFrame to metadata, discarding any unneeded columns."""
        cols = {n: c.values for n, c in dict(frame).items()}

        if isinstance(cols["dof_type"], pandas.Categorical):
            cols["dof_type"] = names_to_val_cat(
                DOFTypes, cols["dof_type"]
            ).codes.astype(int)

        return cls(
            node_idx=cols["node_idx"],
            dof_idx=cols["dof_idx"],
            dof_type=cols["dof_type"],
            child_id=cols["child_id"],
            parent_id=cols["parent_id"],
        )
