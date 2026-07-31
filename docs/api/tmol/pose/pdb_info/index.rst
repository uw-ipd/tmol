tmol.pose.pdb_info
==================

.. py:module:: tmol.pose.pdb_info


Attributes
----------

.. autoapisummary::

   tmol.pose.pdb_info.DEFAULT_ATOM_OCCUPANCY
   tmol.pose.pdb_info.DEFAULT_ATOM_B_FACTOR


Classes
-------

.. autoapisummary::

   tmol.pose.pdb_info.PDBInfo


Module Contents
---------------

.. py:data:: DEFAULT_ATOM_OCCUPANCY
   :value: 1.0


.. py:data:: DEFAULT_ATOM_B_FACTOR
   :value: 0.0


.. py:class:: PDBInfo

   .. rubric:: Docstring

   .. code-block:: text

      Holds other information about a structure as it's read in from a file.
      
      The data held in this class has no impact on structure calculations, e.g.
      the energy of a conformation, but it is useful for preserving information
      about input structures for later output. If the information starts to
      diverge from the actual conformation held in a PoseStack, e.g. if residues
      are added or deleted, then it is the responsibility of the code that makes
      those changes to also update the PDBInfo object accordingly.
      
      Datamembers:
      residue_labels: numpy array of strings giving residue ids for each residue.
          shape: [n_poses x max_n_residues]
      residue_insertion_codes: numpy array of strings giving insertion codes
          for each residue.
          shape: [n_poses x max_n_residues]
      chain_labels: numpy array of strings giving chain labels for each residue.
          shape: [n_poses x max_n_residues]
      atom_occupancy: numpy array of floats giving occupancy for each atom.
          shape: [n_poses x max_n_atoms_per_pose]
      atom_b_factor: numpy array of floats giving B-factors for each atom.
          shape: [n_poses x max_n_atoms_per_pose]
      

   .. py:attribute:: residue_labels
      :type:  tmol.types.array.NDArray[int][:, :]


   .. py:attribute:: residue_insertion_codes
      :type:  tmol.types.array.NDArray[object][:, :]


   .. py:attribute:: chain_labels
      :type:  tmol.types.array.NDArray[object][:, :]


   .. py:attribute:: atom_occupancy
      :type:  tmol.types.array.NDArray[numpy.float32][:, :]


   .. py:attribute:: atom_b_factor
      :type:  tmol.types.array.NDArray[numpy.float32][:, :]


   .. py:method:: split(index) -> PDBInfo

      .. rubric:: Docstring

      .. code-block:: text

         Split out a single pose's worth of PDBInfo from a batch.
         


