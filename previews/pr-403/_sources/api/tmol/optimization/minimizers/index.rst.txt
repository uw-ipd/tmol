tmol.optimization.minimizers
============================

.. py:module:: tmol.optimization.minimizers


Functions
---------

.. autoapisummary::

   tmol.optimization.minimizers.build_kinforest_network
   tmol.optimization.minimizers.run_min
   tmol.optimization.minimizers.run_kin_min
   tmol.optimization.minimizers.run_cart_min


Module Contents
---------------

.. py:function:: build_kinforest_network(pose_stack: tmol.pose.pose_stack.PoseStack, sfxn: tmol.score.score_function.ScoreFunction, ff: tmol.kinematics.fold_forest.FoldForest, mm: tmol.kinematics.move_map.MoveMap, verbose=False, kin_dtype=torch.float32)

.. py:function:: run_min(sfxn_module, optimizer_cls=LBFGS_Armijo, optimizer_kwargs=None, verbose=False)

   .. rubric:: Docstring

   .. code-block:: text

      Run minimization on any sfxn module (Cartesian or KinForest).
      
      The sfxn_module must be a torch.nn.Module whose forward() returns
      per-pose energies and which provides a pose_stack_from_dofs() method
      to extract the optimized PoseStack.
      
      :param sfxn_module: A CartesianSfxnNetwork, KinForestSfxnNetwork, or
                          any nn.Module with a compatible interface.
      :param optimizer_cls: A torch.optim.Optimizer class. Must support a
                            closure-based step() call (e.g. LBFGS_Armijo, torch LBFGS).
      :param optimizer_kwargs: Dict of keyword arguments passed to the optimizer
                               constructor.
      :param verbose: Print timing information.
      
      :returns: A new PoseStack with optimized coordinates.
      

.. py:function:: run_kin_min(pose_stack: tmol.pose.pose_stack.PoseStack, sfxn: tmol.score.score_function.ScoreFunction, ff: tmol.kinematics.fold_forest.FoldForest, mm: tmol.kinematics.move_map.MoveMap, optimizer_cls=LBFGS_Armijo, optimizer_kwargs=None, verbose=False, kin_dtype=torch.float32)

   .. rubric:: Docstring

   .. code-block:: text

      Run minimization on a PoseStack in internal DOF space.
      
      Builds a KinForestSfxnNetwork and delegates to run_min().
      

.. py:function:: run_cart_min(pose_stack: tmol.pose.pose_stack.PoseStack, sfxn: tmol.score.score_function.ScoreFunction, coord_mask=None, optimizer_cls=LBFGS_Armijo, optimizer_kwargs=None, verbose=False)

   .. rubric:: Docstring

   .. code-block:: text

      Run minimization on a PoseStack in Cartesian coordinate space.
      
      Builds a CartesianSfxnNetwork and delegates to run_min().
      

