.. _datatypes:

=================
Datatypes
=================

TMol utilizes a combination tensor & dataclass datatypes for internal data
structures. `torch.Tensor <torch:torch.Tensor>` is generally preferred for
representation of numeric data, with `numpy.ndarray` reserved for
representation of multidimensional `str` or `object` containers. `attrs
<attrs:index>`-based dataclasses are used for python type declarations, with
strong preference toward a functional-object paradigm.
