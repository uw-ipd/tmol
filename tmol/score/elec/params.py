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

from tmol.chemical.restypes import RefinedResidueType


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecGlobalParams(TensorGroup, ValidateAttrs):
    elec_min_dis: Tensor[torch.float32][...]
    elec_max_dis: Tensor[torch.float32][...]
    elec_sigmoidal_die_D: Tensor[torch.float32][...]
    elec_sigmoidal_die_D0: Tensor[torch.float32][...]
    elec_sigmoidal_die_S: Tensor[torch.float32][...]


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

    def get_partial_charges_for_block(self, block_type: RefinedResidueType):
        # find patch variant (if exists in DB) else fall back to basename
        res, *vars = block_type.name.split(":")
        vars.append("")  # unpatched last

        res_found = res in self.partial_charges

        def lookup_charge(atm):
            if not res_found:
                return 0.0
            if atm.name not in self.partial_charges[res]:
                raise KeyError(
                    "Elec charge for atom "
                    + block_type.name
                    + ","
                    + atm.name
                    + " not found"
                )
            for vi in vars:
                if vi in self.partial_charges[res][atm.name]:
                    return self.partial_charges[res][atm.name][vi]

        partial_charge = numpy.vectorize(lookup_charge, otypes=[numpy.float32])(
            block_type.atoms
        )

        return partial_charge

    def get_bonded_path_length_mapping_for_block(self, block_type: RefinedResidueType):
        """remap bonded path length for a residue block"""
        representative_mapping = numpy.arange(block_type.n_atoms, dtype=numpy.int32)

        # find patch variant (if exists in DB) else fall back to basename
        res, *vars = block_type.name.split(":")
        vars.append("")  # unpatched last

        if res not in self.cp_reps:
            # some residues may not have a need for
            # elec's count-pair-representative logic.
            # just return the default representatives
            return representative_mapping

        for outer in block_type.atom_to_idx.keys():
            if outer not in self.cp_reps[res]:
                continue

            inner = None

            for v in vars:
                if v not in self.cp_reps[res][outer]:
                    continue
                inner = self.cp_reps[res][outer][v]
                break

            if inner is None:
                continue

            if inner not in block_type.atom_to_idx:
                raise KeyError(
                    "Invalid elec cp mapping: " + res + " " + outer + "->" + str(inner)
                )

            representative_mapping[block_type.atom_to_idx[inner]] = (
                block_type.atom_to_idx[outer]
            )

        return representative_mapping

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

        def res_patch_from_line(line):
            tag = line.res.split(":")
            assert (
                len(tag) <= 2
            ), "Each atom charge can only be specialized by one patch!"
            if len(tag) == 1:
                return tag[0], ""
            return tag[0], tag[1]

        # dicts of the form dict[res][atm][patch] = value
        #   with patch = '' for unpatched
        def add_to_dict(dict, i, j, k, value):
            if i not in dict:
                dict[i] = {}
            if j not in dict[i]:
                dict[i][j] = {}
            dict[i][j][k] = value

        # Read partial charges
        partial_charges = {}
        for x in elec_database.atom_charge_parameters:
            res, var = res_patch_from_line(x)
            add_to_dict(partial_charges, res, x.atom, var, x.charge)

        # Read countpair reps
        # note that the "inner" and "outer" atoms are flipped relative to
        # the natural interpretation in the file. That if one inner:outer
        # pair is "N": "1H" and another inner:outer pair is "N": "2H", one
        # would naturally conclude that 1H's representative is N and that
        # 2H's representative is also N. However, in actuallity, N's
        # representative will be 2H; N's representative starts out 1H, but
        # then it is overwritten when the 2H entry is parsed.
        #
        # In general, the approach is to use the further atom for the closer
        # atom so that more interactions are counted (because the closer atom
        # will interact with fewer other atoms; the further out something is,
        # the more other atoms will be at least 4 chemical bonds from it.
        # As long as all the atoms j that are listed as representatives for
        # a particular atom i are chemically bound to i, then one atom
        # overriding another as the representative will have no effect.
        cp_reps = {}
        for x in elec_database.atom_cp_reps_parameters:
            res, var = res_patch_from_line(x)
            add_to_dict(cp_reps, res, x.atm_outer, var, x.atm_inner)

        return cls(
            global_params=global_params,
            partial_charges=partial_charges,
            cp_reps=cp_reps,
            device=device,
        )
