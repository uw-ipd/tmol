import attr
import numpy

from tmol.types.array import NDArray

DEFAULT_ATOM_OCCUPANCY = 1.0
DEFAULT_ATOM_B_FACTOR = 0.0


@attr.s(auto_attribs=True)
class PDBInfo:
    """Holds other information about a structure as it's read in from a file.

    The data held in this class has no impact on structure calculations, e.g.
    the energy of a conformation, but it is useful for preserving information
    about input structures for later output. If the information starts to
    diverge from the actual conformation held in a PoseStack, e.g. if residues
    are added or deleted, then it is the responsibility of the code that makes
    those changes to also update the PDBInfo object accordingly.

    Datamembers:
    residue_labels: numpy array of strings giving residue ids for each residue.
        shape: [n_poses x max_n_residues]
    residue_insertion_codes: numpy array of strings giving insertion codes
        for each residue.
        shape: [n_poses x max_n_residues]
    chain_labels: numpy array of strings giving chain labels for each residue.
        shape: [n_poses x max_n_residues]
    atom_occupancy: numpy array of floats giving occupancy for each atom.
        shape: [n_poses x max_n_atoms_per_pose]
    atom_b_factor: numpy array of floats giving B-factors for each atom.
        shape: [n_poses x max_n_atoms_per_pose]
    """

    residue_labels: NDArray[int][:, :]
    residue_insertion_codes: NDArray[object][:, :]
    chain_labels: NDArray[object][:, :]
    atom_occupancy: NDArray[numpy.float32][:, :]
    atom_b_factor: NDArray[numpy.float32][:, :]
