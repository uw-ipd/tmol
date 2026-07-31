tmol.utility.cpp_extension
==========================

.. py:module:: tmol.utility.cpp_extension


Attributes
----------

.. autoapisummary::

   tmol.utility.cpp_extension.arch_list


Functions
---------

.. autoapisummary::

   tmol.utility.cpp_extension.get_torch_version
   tmol.utility.cpp_extension.cuda_if_available
   tmol.utility.cpp_extension.load
   tmol.utility.cpp_extension.load_inline
   tmol.utility.cpp_extension.relpaths
   tmol.utility.cpp_extension.modulename


Module Contents
---------------

.. py:function:: get_torch_version()

.. py:data:: arch_list

.. py:function:: cuda_if_available(sources)

   .. rubric:: Docstring

   .. code-block:: text

      Filter cuda sources if cuda is not available.
      

.. py:function:: load(name, sources, **kwargs)

   .. rubric:: Docstring

   .. code-block:: text

      Jit-compile torch cpp_extension with tmol paths.
      

.. py:function:: load_inline(name, sources, **kwargs)

   .. rubric:: Docstring

   .. code-block:: text

      Jit-compile torch cpp_extension with tmol paths.
      

.. py:function:: relpaths(src_path, paths)

   .. rubric:: Docstring

   .. code-block:: text

      Paths relative to the parent of given src file.
      
      Used to indiciate paths relative to a module's __file__.
      
      .. rubric:: Example
      
      srcs = relpaths(__file__, ["sibling.cpp", "sibling.cu"])
      

.. py:function:: modulename(src_name)

   .. rubric:: Docstring

   .. code-block:: text

      Adapt module name to valid cpp extension name.
      
      Used to adapt a module __name__ to a valid extension name.
      
      .. rubric:: Example
      
      name = modulename(__name__)
      

