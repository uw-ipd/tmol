tmol.utility.log
================

.. py:module:: tmol.utility.log


Attributes
----------

.. autoapisummary::

   tmol.utility.log.ClassLogger


Classes
-------

.. autoapisummary::

   tmol.utility.log.LoggerMixin


Functions
---------

.. autoapisummary::

   tmol.utility.log.classlogger_for
   tmol.utility.log.logger_for_class


Module Contents
---------------

.. py:function:: classlogger_for(instance: object) -> logging.Logger

   .. rubric:: Docstring

   .. code-block:: text

      Get {module}.{class name} named logger for object.
      

.. py:function:: logger_for_class(cls: type) -> logging.Logger

   .. rubric:: Docstring

   .. code-block:: text

      Get {module}.{name} named logger for class.
      

.. py:data:: ClassLogger

.. py:class:: LoggerMixin

   .. py:property:: logger
      :type: logging.Logger



