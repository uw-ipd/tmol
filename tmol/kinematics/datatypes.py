import enum
import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

from tmol.types.attrs import ConvertAttrs
from tmol.types.functional import convert_args


class NodeType(enum.IntEnum):
    """KinForest node types."""

    root = 0
    jump = enum.auto()
    bond = enum.auto()


@attr.s(auto_attribs=True, frozen=True)
class KinForest(TensorGroup, ConvertAttrs):
    """A collection of atom-level kinematic trees, each of which can be processed
    in parallel.

    A kinematic description of a collection of atom locations, each atom location
    corresponding to a node within a tree. The root of each tree in this forest
    is built from a jump from the global reference frame at the origin. (The global
    reference frame will later be treated as a node in the forest, effectively
    linking all the trees, but this is a minor technical detail; best to think
    of this as several independent trees than a single tree). Every other node
    corresponds to a derived orientation, with an atomic coordinate at the
    center of the frame.

    Each node in the tree is connected by one of two "node types":

    1) Jump nodes, representing an arbitrary rigid body transform between two
    reference frames via six degrees of freedom, 3 translational and
    3 rotational.

    2) Bond nodes, representing the relationships between two atom reference
    frames via three bond degrees of freedom: the translation from the parent
    to the child along the bond axis (bond length, d), the rotation from the
    grand-parent-to-parent bond axis to the bond axis (an improper bond
    angle, theta), and the rotation about the grand-parent-to-parent bond axis
    (bond torsion, phi). Bond nodes include an additional, redundent,
    degree of freedom representing concerted rotation of all downstream atoms
    about the parent-to-self bond. These DOFs are used to represent
    the torsions that alter the location of several children. For example,
    chi1 is represented as the 4th DOF for the CB atom of LEU. A rotation
    about the CA-->CB bond axis will spin CG, HB1 and HB2. In this scheme,
    the phi DOF would be 0 for CG, 120 for HB1 and 240 for HB2. This differs
    from the Rosetta3 implementation of downstream-dihedral propagation
    where Chi1 would live as the phi DOF of CG, and CG's rotation would
    carry forward to HB1 and HB2 (requiring that CG be the first child of
    CB).

    The atoms in the `KinForest` have their own order that is distinct
    from the ordering in the target (e.g. a PoseStack) where there might
    be gaps between sets of atoms (e.g. because each Pose in the stack
    has a different number of atoms, so a contiguous block of atom indices
    from 0-100 might have a gap before the next contiguous block begins
    at 150). Whe working with a `KinForest`, remembering what order
    and array's indices is in (the kin-forest order (KFO) or the target
    order (TO)) and what a value/index read out of an array represents (is
    the index an index in KFO or TO?) is *very* challenging. The documentation
    for these arrays includes whether the arrays are indexed in KFO or TO
    and whether the values they hold are KFO or TO indices.

    The `KinForest` data structure itself is frozen and can not be modified post
    construction. The `KinematicBuilder` factory class is responsible for
    construction of a `KinForest` with valid internal structure for atomic
    systems.

    Indices::
        id = the TO index in KFO; i.e. kin_forest_order_2_target_order
        # roots = KFO index for the roots of the trees in the forest;
        #      coordinate updates for these atoms and the path they root will
        #      proceed in parallel in the first pass of the generational
        #      -segmented scan. These are listed in no particular order.
        parent = KFO index of the parent, in KFO
        frame_x = KFO index of self, in KFO 
        frame_y = KFO index of parent, in KFO
        frame_z = KFO index of grandparent, in KFO
    """

    id: Tensor[torch.int32][...]
    # roots: Tensor[torch.int32][...]
    doftype: Tensor[torch.int32][...]
    parent: Tensor[torch.int32][...]
    frame_x: Tensor[torch.int32][...]
    frame_y: Tensor[torch.int32][...]
    frame_z: Tensor[torch.int32][...]

    @classmethod
    @convert_args
    def node(
        cls,
        id: int,
        doftype: NodeType,
        parent: int,
        frame_x: int,
        frame_y: int,
        frame_z: int,
    ):
        """Construct a single node from element values."""
        return cls(
            id=torch.Tensor([id]),
            # roots=torch.Tensor([]),
            doftype=torch.Tensor([doftype]),
            parent=torch.Tensor([parent]),
            frame_x=torch.Tensor([frame_x]),
            frame_y=torch.Tensor([frame_y]),
            frame_z=torch.Tensor([frame_z]),
        )

    @classmethod
    def root_node(cls):
        """The global/root kinematic node at KinForest[0]."""
        return cls.node(
            id=-1, doftype=NodeType.root, parent=0, frame_x=0, frame_y=0, frame_z=0
        )

    # @classmethod
    # def full(cls, n_roots: int, n_kinforest_atoms: int, sentinel: int, **kwargs):
    #     return KinForest(
    #         id=torch.full((n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs),
    #         roots=torch.full((n_roots,), sentinel, dtype=torch.int32, **kwargs),
    #         doftype=torch.full(
    #             (n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs
    #         ),
    #         parent=torch.full(
    #             (n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs
    #         ),
    #         frame_x=torch.full(
    #             (n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs
    #         ),
    #         frame_y=torch.full(
    #             (n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs
    #         ),
    #         frame_z=torch.full(
    #             (n_kinforest_atoms,), sentinel, dtype=torch.int32, **kwargs
    #         ),
    #     )

    # def __len__(self):
    #     """Override the TensorGroup __len__ method since not all tensors
    #     have the same shape
    #     """
    #     return len(self.id)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinDOF(TensorGroup, ConvertAttrs):
    """Internal coordinate data.

    The KinDOF data structure holds two logical views: the "raw" view a
    sparsely populated [n,9] tensor of DOF values and a set of named property
    accessors providing access to specific entries within this array. This is
    logically equivalent a C union datatype, the interpretation of an entry in
    the DOF buffer depends on the type of the corresponding KinForest entry.
    """

    raw: Tensor[torch.double][..., 9]

    @property
    def bond(self):
        return BondDOF(raw=self.raw[..., :4])

    @property
    def jump(self):
        return JumpDOF(raw=self.raw[..., :9])

    def clone(self):
        return KinDOF(raw=self.raw.clone())


class BondDOFTypes(enum.IntEnum):
    """Indices of bond dof types within KinDOF.raw."""

    phi_p = 0
    theta = enum.auto()
    d = enum.auto()
    phi_c = enum.auto()


class JumpDOFTypes(enum.IntEnum):
    """Indices of jump dof types within KinDOF.raw."""

    RBx = 0
    RBy = enum.auto()
    RBz = enum.auto()
    RBdel_alpha = enum.auto()
    RBdel_beta = enum.auto()
    RBdel_gamma = enum.auto()
    RBalpha = enum.auto()
    RBbeta = enum.auto()
    RBgamma = enum.auto()


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BondDOF(TensorGroup, ConvertAttrs):
    """A bond dof view of KinDOF."""

    raw: Tensor[torch.double][..., 4]

    @property
    def phi_p(self):
        return self.raw[..., BondDOFTypes.phi_p]

    @property
    def theta(self):
        return self.raw[..., BondDOFTypes.theta]

    @property
    def d(self):
        return self.raw[..., BondDOFTypes.d]

    @property
    def phi_c(self):
        return self.raw[..., BondDOFTypes.phi_c]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class JumpDOF(TensorGroup, ConvertAttrs):
    """A jump dof view of KinDOF."""

    raw: Tensor[torch.double][..., 9]

    @property
    def RBx(self):
        return self.raw[..., JumpDOFTypes.RBx]

    @property
    def RBy(self):
        return self.raw[..., JumpDOFTypes.RBy]

    @property
    def RBz(self):
        return self.raw[..., JumpDOFTypes.RBz]

    @property
    def RBdel_alpha(self):
        return self.raw[..., JumpDOFTypes.RBdel_alpha]

    @property
    def RBdel_beta(self):
        return self.raw[..., JumpDOFTypes.RBdel_beta]

    @property
    def RBdel_gamma(self):
        return self.raw[..., JumpDOFTypes.RBdel_gamma]

    @property
    def RBalpha(self):
        return self.raw[..., JumpDOFTypes.RBalpha]

    @property
    def RBbeta(self):
        return self.raw[..., JumpDOFTypes.RBbeta]

    @property
    def RBgamma(self):
        return self.raw[..., JumpDOFTypes.RBgamma]
