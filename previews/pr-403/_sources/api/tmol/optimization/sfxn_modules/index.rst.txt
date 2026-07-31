tmol.optimization.sfxn_modules
==============================

.. py:module:: tmol.optimization.sfxn_modules


Classes
-------

.. autoapisummary::

   tmol.optimization.sfxn_modules.CartesianSfxnNetwork
   tmol.optimization.sfxn_modules.KinForestSfxnNetwork


Module Contents
---------------

.. py:class:: CartesianSfxnNetwork(score_function: tmol.score.score_function.ScoreFunction, pose_stack: tmol.pose.pose_stack.PoseStack, coord_mask=None)

   Bases: :py:obj:`torch.nn.Module`


   .. rubric:: Docstring

   .. code-block:: text

      Base class for all neural network modules.
      
      Your models should also subclass this class.
      
      Modules can also contain other Modules, allowing them to be nested in
      a tree structure. You can assign the submodules as regular attributes::
      
          import torch.nn as nn
          import torch.nn.functional as F
      
      
          class Model(nn.Module):
              def __init__(self) -> None:
                  super().__init__()
                  self.conv1 = nn.Conv2d(1, 20, 5)
                  self.conv2 = nn.Conv2d(20, 20, 5)
      
              def forward(self, x):
                  x = F.relu(self.conv1(x))
                  return F.relu(self.conv2(x))
      
      Submodules assigned in this way will be registered, and will also have their
      parameters converted when you call :meth:`to`, etc.
      
      .. note::
          As per the example above, an ``__init__()`` call to the parent class
          must be made before assignment on the child.
      
      :ivar training: Boolean represents whether this module is in training or
                      evaluation mode.
      :vartype training: bool
      

   .. py:attribute:: whole_pose_scoring_module


   .. py:attribute:: pose_stack


   .. py:attribute:: full_coords


   .. py:attribute:: coord_mask
      :value: None



   .. py:attribute:: masked_coords


   .. py:attribute:: count
      :value: 0



   .. py:method:: forward()


   .. py:method:: pose_stack_from_dofs()


.. py:class:: KinForestSfxnNetwork(score_function: tmol.score.score_function.ScoreFunction, pose_stack: tmol.pose.pose_stack.PoseStack, kin_module: tmol.kinematics.script_modules.PoseStackKinematicsModule, dof_mask=None, kin_dtype=torch.float32)

   Bases: :py:obj:`torch.nn.Module`


   .. rubric:: Docstring

   .. code-block:: text

      Base class for all neural network modules.
      
      Your models should also subclass this class.
      
      Modules can also contain other Modules, allowing them to be nested in
      a tree structure. You can assign the submodules as regular attributes::
      
          import torch.nn as nn
          import torch.nn.functional as F
      
      
          class Model(nn.Module):
              def __init__(self) -> None:
                  super().__init__()
                  self.conv1 = nn.Conv2d(1, 20, 5)
                  self.conv2 = nn.Conv2d(20, 20, 5)
      
              def forward(self, x):
                  x = F.relu(self.conv1(x))
                  return F.relu(self.conv2(x))
      
      Submodules assigned in this way will be registered, and will also have their
      parameters converted when you call :meth:`to`, etc.
      
      .. note::
          As per the example above, an ``__init__()`` call to the parent class
          must be made before assignment on the child.
      
      :ivar training: Boolean represents whether this module is in training or
                      evaluation mode.
      :vartype training: bool
      

   .. py:attribute:: pose_stack


   .. py:attribute:: kin_module


   .. py:attribute:: whole_pose_scoring_module


   .. py:attribute:: full_coords


   .. py:attribute:: flat_coords


   .. py:attribute:: orig_coords_shape


   .. py:attribute:: id


   .. py:attribute:: full_dofs


   .. py:attribute:: dof_mask
      :value: None



   .. py:attribute:: masked_dofs


   .. py:attribute:: count
      :value: 0



   .. py:method:: forward()


   .. py:method:: pose_stack_from_dofs()


