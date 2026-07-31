tmol.kinematics.script_modules
==============================

.. py:module:: tmol.kinematics.script_modules


Classes
-------

.. autoapisummary::

   tmol.kinematics.script_modules.PoseStackKinematicsModule


Module Contents
---------------

.. py:class:: PoseStackKinematicsModule(pose_stack: tmol.PoseStack, fold_forest: tmol.kinematics.fold_forest.FoldForest)

   Bases: :py:obj:`torch.nn.Module`


   .. rubric:: Docstring

   .. code-block:: text

      torch.autograd compatible forward kinematic operator for PoseStack.
      
      Perform forward (dof to coordinate) kinematics within torch.autograd
      compute graph. Provides support for forward kinematics over of a subset of
      source dofs, as specified by the provided DOFMetadata entries.
      
      The kinematic system maps between the natm x 9 internal coordinate frame
      and the natm x 3 coordinate frame.  Some of this natm x 9 array is unused
      or is redundant but this is not known by the kinematic module.
      
      See KinDOF for a description of the internal coordinate representation.
      

   .. py:attribute:: kmd


   .. py:attribute:: kinforest


   .. py:attribute:: nodes_f


   .. py:attribute:: scans_f


   .. py:attribute:: gens_f


   .. py:attribute:: nodes_b


   .. py:attribute:: scans_b


   .. py:attribute:: gens_b


   .. py:method:: forward(dofs)


