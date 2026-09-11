import attr
import cattr

import numpy
import torch

from tmol.types import (
    Tensor,
    TensorGroup,
    ValidateAttrs,
    validate_args,
)
from tmol.database.scoring import ElecDatabase

from tmol.chemical import RefinedResidueType


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

    @staticmethod
    def _lookup_order(name):
        """Exact variant, individual patches in name order, then the base.

        Combined records match that complete name only. A partial combined
        record is not inherited by a type with additional patches.
        """
        base, _, patches = name.partition(":")
        return base, tuple(dict.fromkeys((patches, *patches.split(":"), "")))

    def get_partial_charges_for_block(self, block_type: RefinedResidueType):
        res, variants = self._lookup_order(block_type.name)
        if res not in self.partial_charges:
            return numpy.zeros(len(block_type.atoms), dtype=numpy.float32)
        residue_charges = self.partial_charges[res]

        def lookup_charge(atm):
            charges = residue_charges.get(atm.name, {})
            for variant in variants:
                if variant in charges:
                    return charges[variant]
            raise KeyError(
                f"Elec charge for atom {block_type.name},{atm.name} not found"
            )

        return numpy.fromiter(
            (lookup_charge(atom) for atom in block_type.atoms),
            dtype=numpy.float32,
            count=len(block_type.atoms),
        )

    def get_bonded_path_length_mapping_for_block(self, block_type: RefinedResidueType):
        """remap bonded path length for a residue block"""
        representative_mapping = numpy.arange(block_type.n_atoms, dtype=numpy.int32)

        res, variants = self._lookup_order(block_type.name)

        if res not in self.cp_reps:
            # some residues may not have a need for
            # elec's count-pair-representative logic.
            # just return the default representatives
            return representative_mapping

        # Different outer atoms can nominate a representative for the same
        # inner atom. Resolve specificity across those rows too, retaining
        # the historical last-outer-wins rule within one specificity level.
        best_rank = {}
        for outer in block_type.atom_to_idx.keys():
            if outer not in self.cp_reps[res]:
                continue

            inner = None

            for rank, v in enumerate(variants):
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

            if rank <= best_rank.get(inner, len(variants)):
                best_rank[inner] = rank
                representative_mapping[block_type.atom_to_idx[inner]] = (
                    block_type.atom_to_idx[outer]
                )

        return representative_mapping

    @classmethod
    @validate_args
    def from_database(cls, elec_database: ElecDatabase, device: torch.device):
        """Initialize param resolver for all atoms defined in database."""
        # Load global params, coerce to 1D Tensors
        values = cattr.unstructure(elec_database.global_parameters)
        tensor = torch.tensor(list(values.values()), dtype=torch.float32, device=device)
        global_params = ElecGlobalParams(**dict(zip(values, tensor.unbind())))

        def res_patch_from_line(line):
            base, _, patches = line.res.partition(":")
            return base, patches

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
