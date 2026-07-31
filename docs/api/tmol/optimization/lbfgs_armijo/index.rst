tmol.optimization.lbfgs_armijo
==============================

.. py:module:: tmol.optimization.lbfgs_armijo


Classes
-------

.. autoapisummary::

   tmol.optimization.lbfgs_armijo.LBFGS_Armijo


Functions
---------

.. autoapisummary::

   tmol.optimization.lbfgs_armijo.lbfgs_two_loop
   tmol.optimization.lbfgs_armijo.armijo_linesearch


Module Contents
---------------

.. py:function:: lbfgs_two_loop(grad, dirs, stps)

   .. rubric:: Docstring

   .. code-block:: text

      L-BFGS search direction H_k @ grad via the compact
      representation of Byrd, Nocedal & Schnabel, (Math. Prog. 63 (1994)):
          H_0 = I
          M = [[ R^-T (D + Y^T Y) R^-1, -R^-T ], [ -R^-1, 0 ]]
          H_k g = g + [S Y] M [S^T g ; Y^T g]
      
      Algebraically identical to the classic two-loop recursion, but all O(N*m)
      ops are parallelized.
      

.. py:function:: armijo_linesearch(func, derphi0, old_fval, alpha0=1, factor=0.5, sigma_decrease=0.1, sigma_increase=0.8, minstep=1e-06)

   .. rubric:: Docstring

   .. code-block:: text

      Minimize over alpha, the function ``f(xk+alpha pk)``.
      
      :param f: Function to be minimized, f(step)
      :type f: callable
      :param derphi0: (float) directional derivative
      :param fval0: (float) func(0), the value of the function at the origin
      :param alpha0: (float) the initial stepsize
      :param sigma_increase: (float) initial stepsize
                             [must be in (0,1) and >=sigma_decrease]
      :param sigma_decrease: (float) initial stepsize
                             [must be in (0,1) and <=sigma_increase]
      :param factor: (float) scalefactor in modifying stepsize [must be in (0,1)]
      :param minstep: (float) minimum stepsize to take
      
      :returns: stepsize - accepted stepsize
                f_val - final function value
      
      Notes
          See D.P. Bertsekas, Nonlinear Programming, 2nd ed, 1999, page 29.
      
          (fd) A few notes about this specific implementation:
          0) I believe this method was originally from Jim Havranek
          1) 'factor' corresponds roughly to 'beta', BUT on a successful initial step,
             factor is used to increase the stepsize.  When factor is used to decrease
             stepsize, factor^2 is used
          2) The stopping critera used is that in the paper, the first integer m>=0 s.t.:
               f(x_k) - f(x_k+beta^m*s*d_k) >= -sigma * beta^m * s * grad{f}(x_k) * d_k
             however, the two different values of sigma are used:
                * sigma_increase (0.8) is used to trigger an _increased_ stepsize
                * sigma_decrease (0.1) is _required_ or the step size is decreased
          3) in the code
                * 'alpha' corresponds to 's' in the text
                * 'factor' corresponds roughly to 'beta' in the text (see point 1)
      

.. py:class:: LBFGS_Armijo(params, lr=1, max_iter=200, rtol=None, atol=None, gradtol=1.0, history_size=128, minstep=1e-12, verbose=False)

   Bases: :py:obj:`torch.optim.Optimizer`


   .. rubric:: Docstring

   .. code-block:: text

      Implements L-BFGS algorithm with Armijo line search.
      All scaling and parameters taken directly from Rosetta
      
      :param lr: learning rate (default: 1)
      :type lr: float
      :param max_iter: maximal number of iterations (default: 200)
      :type max_iter: int
      :param rtol: relative tolerance (default: 1e-6)
      :type rtol: float
      :param atol: absolute tolerance (default: 0)
      :type atol: float
      :param gradtol: an absolute tolerance on max_i df/dx_i (default: 1e-4)
      :type gradtol: float
      :param history_size: update history size (default: 128).
      :type history_size: int
      

   .. py:attribute:: verbose
      :value: False



   .. py:method:: step(closure)

      .. rubric:: Docstring

      .. code-block:: text

         The LBFGS minimization algorithm. Despite the name, this performs the full
         LBFGS minimization trajectory.
         
         :param func: a function that evaluates energy
         :type func: callable
         
         :returns: the energy (loss) following optimization
         :rtype: orig_loss
         


