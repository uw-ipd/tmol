tmol.score.common.scoring_module
================================

.. py:module:: tmol.score.common.scoring_module


Classes
-------

.. autoapisummary::

   tmol.score.common.scoring_module.TermScoringModule
   tmol.score.common.scoring_module.TermPoseScoringModule
   tmol.score.common.scoring_module.TermWholePoseScoringModule
   tmol.score.common.scoring_module.TermBlockPairScoringModule
   tmol.score.common.scoring_module.TermRotamerScoringModule


Module Contents
---------------

.. py:class:: TermScoringModule(classname, term_parameters, term_score_poses)

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
      

   .. py:attribute:: classname


   .. py:attribute:: term_parameters
      :value: []



   .. py:attribute:: term_score_poses


   .. py:method:: add_parameters(table, params)


.. py:class:: TermPoseScoringModule(classname, pose_stack, term_parameters, term_score_poses)

   Bases: :py:obj:`TermScoringModule`


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
      

   .. py:attribute:: common_parameters
      :value: []



   .. py:attribute:: n_poses


   .. py:attribute:: max_n_blocks


.. py:class:: TermWholePoseScoringModule(classname, pose_stack, term_parameters, term_score_poses)

   Bases: :py:obj:`TermPoseScoringModule`


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
      

   .. py:attribute:: count
      :value: 0



   .. py:method:: forward(coords)


.. py:class:: TermBlockPairScoringModule(classname, pose_stack, term_parameters, term_score_poses)

   Bases: :py:obj:`TermPoseScoringModule`


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
      

   .. py:method:: forward(coords)


.. py:class:: TermRotamerScoringModule(classname, rotamer_set, term_parameters, term_score_poses)

   Bases: :py:obj:`TermScoringModule`


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
      

   .. py:attribute:: common_parameters
      :value: []



   .. py:attribute:: n_poses


   .. py:attribute:: n_rots


   .. py:method:: forward(coords)

      .. rubric:: Docstring

      .. code-block:: text

         Return (scores, indices) without creating any sparse tensor.
         
         scores:  [n_subterms, nnz] float32
         indices: [3, nnz]          int32  (pose, rot_i, rot_j in global rot numbering)
         


   .. py:method:: forward_split(coords)


