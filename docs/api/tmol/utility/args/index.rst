tmol.utility.args
=================

.. py:module:: tmol.utility.args


Functions
---------

.. autoapisummary::

   tmol.utility.args.bind_to_args
   tmol.utility.args.ignore_unused_kwargs


Module Contents
---------------

.. py:function:: bind_to_args(f, *args, **kwargs)

   .. rubric:: Docstring

   .. code-block:: text

      Bind args/kwargs for function into positional arguments.
      

.. py:function:: ignore_unused_kwargs(func)

   .. rubric:: Docstring

   .. code-block:: text

      Ignore kwargs not present in func signature.
      
      Decorate func with wrapper dropping any kwargs not present in the func
      signature.
      
      .. rubric:: Example
      
      Allows function invocation with kwargs bags that are a superset
      of required args::
      
          >>> @ignore_unused_kwargs(lambda a, b: a + b)(a=1, b=2, c=5)
          3
      

