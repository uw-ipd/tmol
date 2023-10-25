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

    # fd this can go away when old scoresystem is removed
    def resolve_partial_charge(
        self, res_names: NDArray[object][:, :], atom_names: NDArray[object][:, :]
    ) -> NDArray[numpy.float32][...]:
        """Convert array of atom type names to partial charges."""

        def lookup_charge(res, atm):
            if res is None:
                return 0.0
            tag, *vars = res.split(":")

            # preserve order invariance!  Make sure only one patch wants
            # to set the charge of this atom
            npatches_modifying_charge = sum(
                [vj in self.partial_charges[tag][atm] for vj in vars]
            )
            if npatches_modifying_charge > 1:
                assert False, (
                    "Multiple patches in "
                    + (",".join(vars))
                    + " modifying charge of atom "
                    + atm
                    + " in res "
                    + res
                )

            vars.append("")  # fallback to base type charge
            for vi in vars:
                if vi in self.partial_charges[tag][atm]:
                    return self.partial_charges[tag][atm][vi]
            assert False, "Elec charge for atom " + res + "," + atm + " not found"
            return 0.0

        pcs = numpy.vectorize(lookup_charge, otypes=[numpy.float32])(
            res_names, atom_names
        )
        return pcs

    # fd this can go away when old scoresystem is removed
    def remap_bonded_path_lengths(
        self,
        bonded_path_lengths: NDArray[object][...],
        res_names: NDArray[object][...],
        res_indices: NDArray[object][...],
        atom_names: NDArray[object][...],
    ) -> NDArray[object][...]:
        """remap bonded path length to use representative atoms"""
        assert bonded_path_lengths.shape[0] == res_names.shape[0]
        assert bonded_path_lengths.shape[0] == res_indices.shape[0]
        assert bonded_path_lengths.shape[0] == atom_names.shape[0]

        def lookup_mapping(res, atm):
            # print("lookup mapping", res, atm)
            if res is None:
                return 0.0
            tag, *vars = res.split(":")
            vars.append("")
            if atm in self.cp_reps[tag]:
                for vi in vars:
                    if vi in self.cp_reps[tag][atm]:
                        # print("self.cp_reps[tag][atm][vi]", self.cp_reps[tag][atm][vi])
                        return self.cp_reps[tag][atm][vi]
            return atm

        # print("res_names", res_names.shape)
        # print("atom_names", atom_names.shape)
        mapped_atoms = numpy.vectorize(lookup_mapping)(res_names, atom_names)
        # mapped_atoms = numpy.empty_like(res_names, dtype=object)
        # for i in range(res_names.shape[0]):
        #     for j in range(res_names.shape[1]):
        #         mapped_atoms[i, j] = lookup_mapping(res_names[i, j], atom_names[i, j])

        # mapped_atoms = numpy.array(list(map(lookup_mapping, res_names.tolist(), atom_names.tolist())), dtype=object)
        # print(mapped_atoms)
        # mapped_atoms = [lookup_mapping(x, y) for x, y in zip(res_names, atom_names)]
        # mapped_atoms = numpy.array(mapped_atoms)

        # fd this could probably be made more efficient but it is going away very soon....
        nstacks = bonded_path_lengths.shape[0]
        remap_bonded_path_lengths = bonded_path_lengths.copy()
        for i in range(nstacks):
            natms = len(res_names[i, ...])
            for j in range(natms):
                matches = numpy.where(
                    (
                        (res_indices[i, ...] == res_indices[i, j])
                        & (atom_names[i, ...] == atom_names[i, j])
                    )
                )
                n_matches = len(matches[0])
                if n_matches == 0 and atom_names[i, j] is not None:
                    print(res_indices[i, j], atom_names[i, j], n_matches, matches)

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

    def get_partial_charges_for_block(self, block_type: RefinedResidueType):
        # find patch variant (if exists in DB) else fall back to basename
        res, *vars = block_type.name.split(":")
        vars.append("")  # unpatched last

        def lookup_charge(atm):
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
            raise KeyError(
                "No elec count-pair representative definition for base name " + res
            )

        for outer in block_type.atom_to_idx.keys():
            if outer not in self.cp_reps[res]:
                continue

            inner = None
            for v in vars:
                if v not in self.cp_reps[res][outer]:
                    continue
                inner = self.cp_reps[res][outer][v]
                break

            if inner not in block_type.atom_to_idx:
                raise KeyError(
                    "Invalid elec cp mapping: " + res + " " + outer + "->" + inner
                )

            representative_mapping[
                block_type.atom_to_idx[inner]
            ] = block_type.atom_to_idx[outer]

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
