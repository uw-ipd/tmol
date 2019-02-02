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
        if not isinstance(atom_names, numpy.ndarray):
            atom_names = numpy.array(atom_names, dtype=object)
        pcs = numpy.vectorize(lambda a, b: self.partial_charges[(a, b)])(
            res_names, atom_names
        )
        return torch.from_numpy(pcs).to(device=self.device)

    # def countpair_reps(self, res_names: NDArray(object)[...], atom_names: NDArray(object)[...]) -> NDArray(object)[...]:
    #    """Convert array of atom type names to partial charges.
    #    """
    #    if not isinstance(atom_names, numpy.ndarray):
    #        atom_names = numpy.array(atom_names, dtype=object)
    #    return numpy.vectorize(lambda a,b: cp_reps[(a,b)])(res_names,atom_names)

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
        # Note
        #    1) cp_flip=False (not default) reverses inner & outer
        #    2) R3 is smart about checking atom existance here with patched residues
        #       -- we are not currently
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
