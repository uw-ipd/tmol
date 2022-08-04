import enum
import math

import attr

import torch
import pandas

from tmol.types.attrs import ConvertAttrs

from tmol.types.tensor import TensorGroup
from tmol.types.torch import Tensor

from tmol.utility.categorical import vals_to_name_cat, names_to_val_cat

from tmol.kinematics.datatypes import NodeType, KinDOF, KinForest


class DOFTypes(enum.IntEnum):
    """High-level class of kinematic DOF types."""

    jump = 0
    bond_angle = enum.auto()
    bond_distance = enum.auto()
    bond_torsion = enum.auto()


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DOFMetadata(TensorGroup, ConvertAttrs):
    """The location, type, and descriptive ids of valid dofs within a KinForest.

    Descriptive entries for dofs within a KinForest, this provides a 1-d
    structure to select and report a subset of entries within a KinDOF buffer.
    DOFMetadata sets are used to indicate mobile vs fixed dofs for KinematicOp
    dof to coordinate functions.

    DOFMetadata supports isomorphic conversion between a DataFrame and
    TensorGroup representation to support symbolic selection. This converts the
    IntEnum encoded "dof_type" entry into a string categorical column.

    The DOFMetadata data members, just like the KinForest, suffer from the same confusion
    about what an index represents because there are two ways to index the data:

    - The "Target Order" (TO) that refers to the index of an atom in the PoseStack
      it came from where the coordinate tensor is squashed to (N,3)
    - The "KinForest Order" (KFO) that refers to the order that an atom's node appears
      in the KinForest; this second ordering puts the index of any child atom after
      the index for any parent atom

    The DOFMetadata class indexes all available DOFs in the system. There are 9 possible
    DOFs per atom (either 3 for BondedAtoms or 9 for JumpAtoms), but in actuality,
    there are many fewer valid DOFs. The DOFMetadata class indexes valid DOFs.

    For each valid DOF i, there's:
    - node_idx[i]: the KFO index of the atom that DOF i belongs to
    - dof_idx[i]: the index between 0-8 for DOF i on its atom
    - dof_type[i]: the DOF type (either a BondDOFType or a JumpDOFType) for DOF i
    - parent_id[i]: the TO index for the parent to node_idx[i] for DOF i
    - child_id[i]: the TO index for node_idx[i] for DOF i

    The DOFMetadata class is primarily used to index into torch tensors in python, and therefore
    all of its dtypes are 64-bit integers.
    
    """

    node_idx: Tensor[torch.long][...]
    dof_idx: Tensor[torch.long][...]

    dof_type: Tensor[torch.long][...]
    parent_id: Tensor[torch.long][...]
    child_id: Tensor[torch.long][...]

    @classmethod
    def for_kinforest(cls, kinforest: KinForest):
        """Return all valid dofs within a KinForest."""

        # Setup a dof type table the same shape as the kinematic dofs,
        # marking all potential movable dofs with the abstract dof type.
        # Leaving all non-movable or invalid dofs as nan. Essentially,
        # [n-atoms x 9]
        dof_types = KinDOF.full(kinforest.shape, math.nan)

        parentIdx = kinforest.parent.to(dtype=torch.long)
        # count the number of children each KFO node has and then ask is that number > 0
        node_has_children = (
            torch.zeros_like(kinforest.id).put_(
                parentIdx, torch.ones_like(kinforest.parent), accumulate=True
            )
            > 0
        )

        bsel = kinforest.doftype == NodeType.bond

        dof_types.bond.phi_p[bsel] = DOFTypes.bond_angle
        dof_types.bond.theta[bsel] = DOFTypes.bond_angle
        dof_types.bond.d[bsel] = DOFTypes.bond_distance
        # Only flag the "child phi" if the node has affected children.
        dof_types.bond.phi_c[bsel & node_has_children] = DOFTypes.bond_torsion

        jsel = kinforest.doftype == NodeType.jump
        dof_types.jump.RBx[jsel] = DOFTypes.jump
        dof_types.jump.RBy[jsel] = DOFTypes.jump
        dof_types.jump.RBz[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_alpha[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_beta[jsel] = DOFTypes.jump
        dof_types.jump.RBdel_gamma[jsel] = DOFTypes.jump

        # Get indices of all marked dofs.
        node_idx, dof_idx = (~torch.isnan(dof_types.raw)).nonzero(as_tuple=True)

        # Unpack into the node/dof index, then expand
        return cls(
            node_idx=node_idx,
            dof_idx=dof_idx,
            dof_type=dof_types.raw[node_idx, dof_idx],
            child_id=kinforest.id[node_idx],
            parent_id=kinforest.id[parentIdx[node_idx]],
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
