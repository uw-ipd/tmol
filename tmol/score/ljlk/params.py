import attr
import cattr

import pandas

import torch


from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.ljlk import LJLKDatabase
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase

from ..chemical_database import AtomTypeParamResolver


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParams(TensorGroup):
    max_dis: Tensor[torch.float32][...]
    spline_start: Tensor[torch.float32][...]
    lj_hbond_OH_donor_dis: Tensor[torch.float32][...]
    lj_hbond_dis: Tensor[torch.float32][...]
    lj_hbond_hdis: Tensor[torch.float32][...]
    lj_switch_dis2sigma: Tensor[torch.float32][...]
    lk_min_dis2sigma: Tensor[torch.float32][...]

    # not clear if this belongs here or in a separate class
    lkb_water_dist: Tensor[torch.float32][...]
    lkb_water_angle_sp2: Tensor[torch.float32][...]
    lkb_water_angle_sp3: Tensor[torch.float32][...]
    lkb_water_angle_ring: Tensor[torch.float32][...]
    lkb_water_tors_sp2: Tensor[torch.float32][..., :]
    lkb_water_tors_sp3: Tensor[torch.float32][..., :]
    lkb_water_tors_ring: Tensor[torch.float32][..., :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LJLKTypeParams(TensorGroup):
    lj_radius: Tensor[torch.float32][...]
    lj_wdepth: Tensor[torch.float32][...]
    lk_dgfree: Tensor[torch.float32][...]
    lk_lambda: Tensor[torch.float32][...]
    lk_volume: Tensor[torch.float32][...]

    # Parameters imported from chemical database
    is_acceptor: Tensor[bool][...]
    acceptor_hybridization: Tensor[torch.int][...]

    is_donor: Tensor[bool][...]

    is_hydroxyl: Tensor[bool][...]
    is_polarh: Tensor[bool][...]

    is_hydrogen: Tensor[bool][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LJLKParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    # str->int index from atom type name to index within parameter tensors
    atom_type_index: pandas.Index

    # shape [1] global parameters
    global_params: LJLKGlobalParams

    # shape [n] per-type parameters, source params used to calculate pair parameters
    type_params: LJLKTypeParams

    device: torch.device

    @classmethod
    @validate_args
    def from_database(
        cls,
        chemical_database: PatchedChemicalDatabase,
        ljlk_database: LJLKDatabase,
        device: torch.device,
    ):
        """Initialize param resolver for all atom types in database."""
        return cls.from_param_resolver(
            AtomTypeParamResolver.from_database(chemical_database, device=device),
            ljlk_database,
        )

    @classmethod
    @validate_args
    def from_param_resolver(
        cls, atom_type_resolver: AtomTypeParamResolver, ljlk_database: LJLKDatabase
    ):
        # Reference existing atom type index from atom_type_resolver
        atom_type_index = atom_type_resolver.index
        device = atom_type_resolver.device

        # Convert float entries into 1-d tensors
        def at_least_1d(t):
            if t.dim() == 0:
                return t.expand((1,))
            else:
                return t

        global_params = LJLKGlobalParams(
            **{
                n: at_least_1d(torch.tensor(v, device=device))
                for n, v in cattr.unstructure(ljlk_database.global_parameters).items()
            }
        )

        # Pack the tuple of type parameters into a dataframe and reindex via
        # the param resolver type index. This appends a "nan" row at the end of
        # the frame for the invalid/None entry added above.
        param_records = (
            pandas.DataFrame.from_records(
                cattr.unstructure(ljlk_database.atom_type_parameters)
            )
            .set_index("name")
            .reindex(index=atom_type_index)
        )

        # Convert the param record dataframe into typed TensorGroup
        type_params = LJLKTypeParams(
            # Reference parameters from atom_type_resolver
            is_acceptor=atom_type_resolver.params.is_acceptor,
            acceptor_hybridization=atom_type_resolver.params.acceptor_hybridization,
            is_donor=atom_type_resolver.params.is_donor,
            is_hydroxyl=atom_type_resolver.params.is_hydroxyl,
            is_polarh=atom_type_resolver.params.is_polarh,
            **{
                f.name: torch.tensor(
                    param_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(LJLKTypeParams)
                if f.name
                in ("lj_radius", "lj_wdepth", "lk_dgfree", "lk_lambda", "lk_volume")
            },
            is_hydrogen=atom_type_resolver.params.is_hydrogen,
        )

        return cls(
            atom_type_index=atom_type_index,
            global_params=global_params,
            type_params=type_params,
            device=device,
        )
