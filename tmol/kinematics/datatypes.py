import enum
import numpy


class DOFType(enum.IntEnum):
    root = 0
    jump = enum.auto()
    bond = enum.auto()


# data structure describing the atom-level kinematics of a molecular system
kintree_node_dtype = numpy.dtype([
    ("id", numpy.int),
    ("doftype", numpy.int),
    ("parent", numpy.int),
    ("frame_x", numpy.int),
    ("frame_y", numpy.int),
    ("frame_z", numpy.int),
])
