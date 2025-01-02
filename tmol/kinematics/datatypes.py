import enum
import numpy
import torch
import attrs

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray

from tmol.types.attrs import ConvertAttrs
from tmol.types.functional import convert_args


class NodeType(enum.IntEnum):
    """KinForest node types."""

    root = 0
    jump = enum.auto()
    bond = enum.auto()


@attrs.define(auto_attribs=True, frozen=True)
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
    at 150). When working with a `KinForest`, remembering what order
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


@attrs.define(auto_attribs=True, frozen=True)
class KinForestScanData(TensorGroup, ConvertAttrs):
    nodes: Tensor[torch.int]
    scans: Tensor[torch.int]
    gens: Tensor[torch.int]


@attrs.define(auto_attribs=True, frozen=True)
class KinematicModuleData:
    forest: KinForest
    scan_data_fw: KinForestScanData
    scan_data_bw: KinForestScanData

    # some extra tensors we need to describe
    # the fold forest in its entirety
    block_in_and_first_out: Tensor[torch.int][:, :]
    keep_atom_fixed: Tensor[torch.bool][:, :]
    pose_stack_atom_for_jump: Tensor[torch.int][:, :, 2]


@attrs.define(auto_attribs=True, slots=True, frozen=True)
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


n_movalbe_bond_dof_types = 4


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


n_movable_jump_dof_types = 6


@attrs.define(auto_attribs=True, slots=True, frozen=True)
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


@attrs.define(auto_attribs=True, slots=True, frozen=True)
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


@attrs.define
class BTGenerationalSegScanPathSegs:
    jump_atom: int
    parents: NDArray[numpy.int64][:, :]  # n-input x n-atoms
    dof_type: NDArray[numpy.int64][:, :]  # n-input x n-atoms
    input_conn_atom: NDArray[numpy.int64][:]  # n-input
    n_gens: NDArray[numpy.int64][:, :]  # n-input x n-output
    n_nodes_for_gen: NDArray[numpy.int64][:, :, :]
    nodes_for_gen: NDArray[numpy.int64][
        :, :, :, :
    ]  # n-input x n-output x max-n-gen x max-n-nodes-per-gen
    n_scan_path_segs: NDArray[numpy.int64][:, :, :]  # n-input x n-output x n-gen
    scan_path_seg_that_builds_output_conn: NDArray[numpy.int64][
        :, :, :, 2
    ]  # n-input x n-output x n-conn x 2
    scan_path_seg_starts: NDArray[numpy.int64][:, :, :, :]
    scan_path_seg_is_real: NDArray[bool][:, :, :, :]
    scan_path_seg_is_inter_block: NDArray[bool][:, :, :, :]
    scan_path_seg_lengths: NDArray[numpy.int64][:, :, :, :]
    uaid_for_torsion: NDArray[numpy.int64][:, :, 3]  # n-input x n-torsions x 3
    torsion_direction: NDArray[numpy.int64][:, :]  # n-input x n-torsions

    @classmethod
    def empty(
        cls,
        n_input_types: int,
        n_output_types: int,
        n_atoms: int,
        n_conn: int,
        max_n_gens: int,
        max_n_scan_path_segs_per_gen: int,
        max_n_nodes_per_gen: int,
        n_torsions: int,
    ):
        io = (n_input_types, n_output_types)
        return cls(
            jump_atom=-1,
            parents=numpy.full(
                (n_input_types, n_atoms), -1, dtype=int
            ),  # independent of primary output
            dof_type=numpy.full(
                (n_input_types, n_atoms), -1, dtype=int
            ),  # independent of primary output
            input_conn_atom=numpy.full(n_input_types, -1, dtype=int),
            n_gens=numpy.zeros(io, dtype=int),
            n_nodes_for_gen=numpy.zeros(io + (max_n_gens,), dtype=int),
            nodes_for_gen=numpy.full(
                io + (max_n_gens, max_n_nodes_per_gen), -1, dtype=int
            ),
            n_scan_path_segs=numpy.zeros(io + (max_n_gens,), dtype=int),
            scan_path_seg_that_builds_output_conn=numpy.full(
                io + (n_conn, 2), -1, dtype=int
            ),
            scan_path_seg_starts=numpy.full(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), -1, dtype=int
            ),
            scan_path_seg_is_real=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=bool
            ),
            scan_path_seg_is_inter_block=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=bool
            ),
            scan_path_seg_lengths=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=int
            ),
            uaid_for_torsion=numpy.full((n_input_types, n_torsions, 3), -1, dtype=int),
            torsion_direction=numpy.full((n_input_types, n_torsions), 1, dtype=int),
        )


@attrs.define
class PBTGenerationalSegScanPathSegs:
    jump_atom: NDArray[numpy.int64][:]  # n-bt
    parents: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-atoms
    dof_type: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-atoms
    input_conn_atom: Tensor[torch.int32][:, :]  # n-bt x n-input
    n_gens: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-output
    n_nodes_for_gen: Tensor[torch.int32][:, :, :, :]
    nodes_for_gen: Tensor[torch.int32][
        :, :, :, :, :
    ]  # n-input x n-output x max-n-gen x max-n-nodes-per-gen
    n_scan_path_segs: Tensor[torch.int32][
        :, :, :, :
    ]  # n-bt x n-input x n-output x n-gen
    scan_path_seg_that_builds_output_conn: NDArray[numpy.int64][
        :, :, :, :, 2
    ]  # n-bt x n-input x n-output x n-conn x 2
    scan_path_seg_starts: Tensor[torch.int32][:, :, :, :, :]
    scan_path_seg_is_real: Tensor[bool][:, :, :, :, :]
    scan_path_seg_is_inter_block: Tensor[bool][:, :, :, :, :]
    scan_path_seg_lengths: Tensor[torch.int32][:, :, :, :, :]
    uaid_for_torsion: NDArray[numpy.int64][
        :, :, :, 3
    ]  # n-bt x n-input x n-torsions x 3
    torsion_direction: NDArray[numpy.int64][:, :, :]  # n-bt x n-input x n-torsions

    @classmethod
    def empty(
        cls,
        device,
        n_bt: int,
        max_n_input_types: int,
        max_n_output_types: int,
        max_n_atoms: int,
        max_n_conn: int,
        max_n_gens: int,
        max_n_scan_path_segs_per_gen: int,
        max_n_nodes_per_gen: int,
        max_n_torsions: int,
    ):
        io = (n_bt, max_n_input_types, max_n_output_types)
        return cls(
            jump_atom=torch.full((n_bt,), -1, dtype=torch.int32, device=device),
            parents=torch.full(
                (n_bt, max_n_input_types, max_n_atoms),
                -1,
                dtype=torch.int32,
                device=device,
            ),  # independent of primary output
            dof_type=torch.full(
                (n_bt, max_n_input_types, max_n_atoms),
                -1,
                dtype=torch.int32,
                device=device,
            ),  # independent of primary output
            input_conn_atom=torch.full(
                (n_bt, max_n_input_types), -1, dtype=torch.int32, device=device
            ),
            n_gens=torch.zeros(io, dtype=torch.int32, device=device),
            n_nodes_for_gen=torch.zeros(
                io + (max_n_gens,), dtype=torch.int32, device=device
            ),
            nodes_for_gen=torch.full(
                io + (max_n_gens, max_n_nodes_per_gen),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            n_scan_path_segs=torch.zeros(
                io + (max_n_gens,), dtype=torch.int32, device=device
            ),
            scan_path_seg_that_builds_output_conn=torch.full(
                io + (max_n_conn, 2), -1, dtype=torch.int32, device=device
            ),
            scan_path_seg_starts=torch.full(
                io + (max_n_gens, max_n_scan_path_segs_per_gen),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            scan_path_seg_is_real=torch.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen),
                dtype=torch.bool,
                device=device,
            ),
            scan_path_seg_is_inter_block=torch.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen),
                dtype=bool,
                device=device,
            ),
            scan_path_seg_lengths=torch.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen),
                dtype=torch.int32,
                device=device,
            ),
            uaid_for_torsion=torch.full(
                (n_bt, max_n_input_types, max_n_torsions, 3), -1, dtype=torch.int32
            ),
            torsion_direction=torch.full(
                (n_bt, max_n_input_types, max_n_torsions), 1, dtype=torch.int32
            ),
        )
