import attr
import torch
import numpy
from typing import Optional

from tmol.types.torch import Tensor
from tmol.types.array import NDArray


@attr.s(auto_attribs=True, frozen=False, slots=True)
class CanonicalForm:
    """This class holds the data that describe a (stack of) structure(s) in a poised, ready-to-use state.

    This datastructure holds the information necessary to determine the chemical
    identities of the residues in the structure(s), which may be under-determined
    from tmol's perspective by the source of the structure (e.g. OpenFold does not
    explicitly model termini). The atoms that are present are represented with
    non-NaN coordinates in the `coords` array; the order in which those atoms appear
    is given by a particular CanonicalOrdering object.

    The datastructure also holds convenience information such as author-provided
    residue labels (ints), chain labels (strings) & insertion codes (strings) as well
    as the occupancy and B-factor of each atom. These are not strictly necessary
    but are often useful when processing structures.
    """

    # n_poses x max_n_res
    chain_id: Tensor[torch.int64][:, :]
    # n_poses x max_n_res
    res_types: Tensor[torch.int64][:, :]
    # n_poses x max_n_res x max_n_canonical_atoms x 3
    coords: Tensor[torch.float32][:, :, :, 3]
    # n_poses x max_n_res
    res_labels: NDArray[int][:, :]
    # n_poses x max_n_res
    residue_insertion_codes: NDArray[object][:, :]
    # n_poses x max_n_res
    chain_labels: NDArray[object][:, :]
    # n_poses x max_n_res x max_n_canonical_atoms
    atom_occupancy: Optional[NDArray[numpy.float32][:, :, :]]
    # n_poses x max_n_res x max_n_canonical_atoms
    atom_b_factor: Optional[NDArray[numpy.float32][:, :, :]]
    # n_disulfides x 3
    disulfides: Optional[Tensor[torch.int64][:, 3]]
    # n_poses x max_n_res x 2
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]]

    def __iter__(self):
        yield self.chain_id
        yield self.res_types
        yield self.coords
        yield self.res_labels
        yield self.residue_insertion_codes
        yield self.chain_labels
        yield self.atom_occupancy
        yield self.atom_b_factor
        yield self.disulfides
        yield self.res_not_connected

    def as_dict(self):
        return {
            "chain_id": self.chain_id,
            "res_types": self.res_types,
            "coords": self.coords,
            "res_labels": self.res_labels,
            "residue_insertion_codes": self.residue_insertion_codes,
            "chain_labels": self.chain_labels,
            "atom_occupancy": self.atom_occupancy,
            "atom_b_factor": self.atom_b_factor,
            "disulfides": self.disulfides,
            "res_not_connected": self.res_not_connected,
        }
