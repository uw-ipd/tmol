import attr
import cattr

import numpy
import pandas
import torch

from enum import IntEnum


from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.elec import ElecDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecGlobalParams(TensorGroup, ValidateAttrs):
    elec_min_dis: Tensor("f")[...]
    elec_max_dis: Tensor("f")[...]
    elec_sigmoidal_die_D: Tensor("f")[...]
    elec_sigmoidal_die_D0: Tensor("f")[...]
    elec_sigmoidal_die_S: Tensor("f")[...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ElecParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    global_params: ElecGlobalParams

    device: torch.device

    # map (AA,atom) to atom
    cp_reps: dict

    # map (AA,atom) to partial charge
    partial_charges: dict

    def resolve_partial_charge(
        self, res_names: NDArray(object)[...], atom_names: NDArray(object)[...]
    ) -> NDArray("f")[...]:
        """Convert array of atom type names to partial charges.
        """
        pcs = numpy.vectorize(lambda a, b: self.partial_charges[(a, b)])(
            res_names, atom_names
        )
        return torch.from_numpy(pcs).to(device=self.device)

    def remap_bonded_path_lengths(
        self,
        bonded_path_lengths: NDArray(object)[...],
        res_names: NDArray(object)[...],
        res_indices: NDArray(object)[...],
        atom_names: NDArray(object)[...],
    ) -> NDArray(object)[...]:
        """remap bonded path length to use representative atoms
        """
        mapped_atoms = numpy.vectorize(
            lambda a, b: self.cp_reps[(a, b)] if (a, b) in self.cp_reps else b
        )(res_names, atom_names)

        def remap(a, b):
            if numpy.isnan(a):
                return -1
            return numpy.where((res_indices == a) & (atom_names == b))[0][0]

        mapped_indices = numpy.vectorize(remap)(res_indices, mapped_atoms)

        remap_bonded_path_lengths = bonded_path_lengths.copy()
        remap_bonded_path_lengths[mapped_indices, :] = remap_bonded_path_lengths
        remap_bonded_path_lengths[:, mapped_indices] = remap_bonded_path_lengths
        return torch.from_numpy(remap_bonded_path_lengths).to(device=self.device)

    @classmethod
    @validate_args
    def from_database(cls, elec_database: ElecDatabase, device: torch.device):
        """Initialize param resolver for all atoms defined in database."""
        # Load global params, coerce to 1D Tensors
        global_params = ElecGlobalParams(
            **{
                n: torch.tensor(v, device=device)
                for n, v in cattr.unstructure(elec_database.global_parameters).items()
            }
        )

        # Read countpair reps
        # Note: cp_flip=False (not default) flips inner & outer atoms
        cp_reps = {
            (x.res, x.atm_outer): x.atm_inner
            for x in elec_database.atom_cp_reps_parameters
        }

        # Read partial charges
        partial_charges = {
            (x.res, x.atom): x.charge for x in elec_database.atom_charge_parameters
        }
        partial_charges[(None, None)] = 0.000

        return cls(
            global_params=global_params,
            partial_charges=partial_charges,
            cp_reps=cp_reps,
            device=device,
        )
