import attr
import cattr

import numpy
import torch

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.elec import ElecDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecGlobalParams(TensorGroup, ValidateAttrs):
    elec_min_dis: Tensor(torch.float32)[...]
    elec_max_dis: Tensor(torch.float32)[...]
    elec_sigmoidal_die_D: Tensor(torch.float32)[...]
    elec_sigmoidal_die_D0: Tensor(torch.float32)[...]
    elec_sigmoidal_die_S: Tensor(torch.float32)[...]


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
        self, res_names: NDArray(object)[:, :], atom_names: NDArray(object)[:, :]
    ) -> NDArray(numpy.float32)[...]:
        """Convert array of atom type names to partial charges.
        """
        pcs = numpy.vectorize(
            lambda a, b: self.partial_charges[(a, b)], otypes=[numpy.float32]
        )(res_names, atom_names)
        return pcs

    def remap_bonded_path_lengths(
        self,
        bonded_path_lengths: NDArray(object)[...],
        res_names: NDArray(object)[...],
        res_indices: NDArray(object)[...],
        atom_names: NDArray(object)[...],
    ) -> NDArray(object)[...]:
        """remap bonded path length to use representative atoms
        """
        assert bonded_path_lengths.shape[0] == res_names.shape[0]
        assert bonded_path_lengths.shape[0] == res_indices.shape[0]
        assert bonded_path_lengths.shape[0] == atom_names.shape[0]

        mapped_atoms = numpy.vectorize(
            lambda a, b: self.cp_reps[(a, b)] if (a, b) in self.cp_reps else b
        )(res_names, atom_names)

        nstacks = bonded_path_lengths.shape[0]
        remap_bonded_path_lengths = bonded_path_lengths.copy()
        for i in range(nstacks):
            natms = len(res_names[i, ...])
            mapped_indices = numpy.vectorize(
                lambda a, b, c: c
                if a is None or numpy.isnan(a)
                else (
                    numpy.where((res_indices[i, ...] == a) & (atom_names[i, ...] == b))[
                        0
                    ]
                )
            )(res_indices[i, ...], mapped_atoms[i, ...], numpy.arange(natms))

            # fmt: off
            remap_bonded_path_lengths[i, mapped_indices, :] = (
                remap_bonded_path_lengths[i, ...])
            remap_bonded_path_lengths[i, :, mapped_indices] = (
                remap_bonded_path_lengths[i, ...])
            # fmt: on

        return remap_bonded_path_lengths

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
