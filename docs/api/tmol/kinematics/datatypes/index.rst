tmol.kinematics.datatypes
=========================

.. py:module:: tmol.kinematics.datatypes


Attributes
----------

.. autoapisummary::

   tmol.kinematics.datatypes.n_movable_bond_dof_types
   tmol.kinematics.datatypes.n_movable_jump_dof_types


Classes
-------

.. autoapisummary::

   tmol.kinematics.datatypes.NodeType
   tmol.kinematics.datatypes.KinForest
   tmol.kinematics.datatypes.KinForestScanData
   tmol.kinematics.datatypes.KinematicModuleData
   tmol.kinematics.datatypes.KinDOF
   tmol.kinematics.datatypes.BondDOFTypes
   tmol.kinematics.datatypes.JumpDOFTypes
   tmol.kinematics.datatypes.BondDOF
   tmol.kinematics.datatypes.JumpDOF
   tmol.kinematics.datatypes.BTGenerationalSegScanPathSegs
   tmol.kinematics.datatypes.PBTGenerationalSegScanPathSegs


Module Contents
---------------

.. py:class:: NodeType

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      KinForest node types.
      

   .. py:attribute:: root
      :value: 0



   .. py:attribute:: jump


   .. py:attribute:: bond


.. py:class:: KinForest

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      A collection of atom-level kinematic trees, each of which can be processed
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
      construction. `KinForests` are not typically built directly by users,
      but rather are constructed as part of building the `KinModuleData` class
      by `construct_kin_module_data` that lives in scan_ordering.py.
      The `_KinematicBuilder` factory class is now deprecated but use to be
      responsible for construction of `KinForests`
      
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
      

   .. py:attribute:: id
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: doftype
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: parent
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: frame_x
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: frame_y
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: frame_z
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:method:: node(id: int, doftype: NodeType, parent: int, frame_x: int, frame_y: int, frame_z: int)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Construct a single node from element values.
         


   .. py:method:: root_node()
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         The global/root kinematic node at KinForest[0].
         


.. py:class:: KinForestScanData

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: nodes
      :type:  tmol.types.torch.Tensor[torch.int]


   .. py:attribute:: scans
      :type:  tmol.types.torch.Tensor[torch.int]


   .. py:attribute:: gens
      :type:  tmol.types.torch.Tensor[torch.int]


.. py:class:: KinematicModuleData

   .. py:attribute:: forest
      :type:  KinForest


   .. py:attribute:: scan_data_fw
      :type:  KinForestScanData


   .. py:attribute:: scan_data_bw
      :type:  KinForestScanData


   .. py:attribute:: block_in_and_first_out
      :type:  tmol.types.torch.Tensor[torch.int][:, :]


   .. py:attribute:: keep_atom_fixed
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: pose_stack_atom_for_jump
      :type:  tmol.types.torch.Tensor[torch.int][:, :, 2]


   .. py:attribute:: pose_stack_atom_for_root_jump
      :type:  tmol.types.torch.Tensor[torch.int][:, 2]


.. py:class:: KinDOF

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Internal coordinate data.
      
      The KinDOF data structure holds two logical views: the "raw" view a
      sparsely populated [n,9] tensor of DOF values and a set of named property
      accessors providing access to specific entries within this array. This is
      logically equivalent a C union datatype, the interpretation of an entry in
      the DOF buffer depends on the type of the corresponding KinForest entry.
      

   .. py:attribute:: raw
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 9]


   .. py:property:: bond


   .. py:property:: jump


   .. py:method:: clone()


.. py:class:: BondDOFTypes

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      Indices of bond dof types within KinDOF.raw.
      

   .. py:attribute:: phi_p
      :value: 0



   .. py:attribute:: theta


   .. py:attribute:: d


   .. py:attribute:: phi_c


.. py:data:: n_movable_bond_dof_types
   :value: 4


.. py:class:: JumpDOFTypes

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      Indices of jump dof types within KinDOF.raw.
      

   .. py:attribute:: RBx
      :value: 0



   .. py:attribute:: RBy


   .. py:attribute:: RBz


   .. py:attribute:: RBdel_alpha


   .. py:attribute:: RBdel_beta


   .. py:attribute:: RBdel_gamma


   .. py:attribute:: RBalpha


   .. py:attribute:: RBbeta


   .. py:attribute:: RBgamma


.. py:data:: n_movable_jump_dof_types
   :value: 6


.. py:class:: BondDOF

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      A bond dof view of KinDOF.
      

   .. py:attribute:: raw
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 4]


   .. py:property:: phi_p


   .. py:property:: theta


   .. py:property:: d


   .. py:property:: phi_c


.. py:class:: JumpDOF

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      A jump dof view of KinDOF.
      

   .. py:attribute:: raw
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 9]


   .. py:property:: RBx


   .. py:property:: RBy


   .. py:property:: RBz


   .. py:property:: RBdel_alpha


   .. py:property:: RBdel_beta


   .. py:property:: RBdel_gamma


   .. py:property:: RBalpha


   .. py:property:: RBbeta


   .. py:property:: RBgamma


.. py:class:: BTGenerationalSegScanPathSegs

   .. py:attribute:: jump_atom
      :type:  int


   .. py:attribute:: parents
      :type:  tmol.types.array.NDArray[numpy.int64][:, :]


   .. py:attribute:: dof_type
      :type:  tmol.types.array.NDArray[numpy.int64][:, :]


   .. py:attribute:: input_conn_atom
      :type:  tmol.types.array.NDArray[numpy.int64][:]


   .. py:attribute:: n_gens
      :type:  tmol.types.array.NDArray[numpy.int64][:, :]


   .. py:attribute:: n_nodes_for_gen
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :]


   .. py:attribute:: nodes_for_gen
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, :]


   .. py:attribute:: n_scan_path_segs
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :]


   .. py:attribute:: scan_path_seg_that_builds_output_conn
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, 2]


   .. py:attribute:: scan_path_seg_starts
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, :]


   .. py:attribute:: scan_path_seg_is_real
      :type:  tmol.types.array.NDArray[bool][:, :, :, :]


   .. py:attribute:: scan_path_seg_is_inter_block
      :type:  tmol.types.array.NDArray[bool][:, :, :, :]


   .. py:attribute:: scan_path_seg_lengths
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, :]


   .. py:attribute:: uaid_for_torsion_by_inconn
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, 3]


   .. py:attribute:: torsion_direction
      :type:  tmol.types.array.NDArray[numpy.int64][:, :]


   .. py:method:: empty(n_input_types: int, n_output_types: int, n_atoms: int, n_conn: int, max_n_gens: int, max_n_scan_path_segs_per_gen: int, max_n_nodes_per_gen: int, n_torsions: int)
      :classmethod:



.. py:class:: PBTGenerationalSegScanPathSegs

   .. py:attribute:: jump_atom
      :type:  tmol.types.array.NDArray[numpy.int64][:]


   .. py:attribute:: parents
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: dof_type
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: input_conn_atom
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: n_gens
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: n_nodes_for_gen
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :]


   .. py:attribute:: nodes_for_gen
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :, :]


   .. py:attribute:: n_scan_path_segs
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :]


   .. py:attribute:: scan_path_seg_that_builds_output_conn
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, :, 2]


   .. py:attribute:: scan_path_seg_starts
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :, :]


   .. py:attribute:: scan_path_seg_is_real
      :type:  tmol.types.torch.Tensor[bool][:, :, :, :, :]


   .. py:attribute:: scan_path_seg_is_inter_block
      :type:  tmol.types.torch.Tensor[bool][:, :, :, :, :]


   .. py:attribute:: scan_path_seg_lengths
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :, :]


   .. py:attribute:: uaid_for_torsion_by_inconn
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :, 3]


   .. py:attribute:: torsion_direction
      :type:  tmol.types.array.NDArray[numpy.int64][:, :, :]


   .. py:method:: empty(device, n_bt: int, max_n_input_types: int, max_n_output_types: int, max_n_atoms: int, max_n_conn: int, max_n_gens: int, max_n_scan_path_segs_per_gen: int, max_n_nodes_per_gen: int, max_n_torsions: int)
      :classmethod:



