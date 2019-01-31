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

from tmol.database.scoring.ljlk import LJLKDatabase


class AcceptorHybridization(IntEnum):
    none = 0
    sp2 = 1
    sp3 = 2
    ring = 3


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParams(TensorGroup, ValidateAttrs):
    max_dis: Tensor("f")[...]
    spline_start: Tensor("f")[...]
    lj_hbond_OH_donor_dis: Tensor("f")[...]
    lj_hbond_dis: Tensor("f")[...]
    lj_hbond_hdis: Tensor("f")[...]
    lj_switch_dis2sigma: Tensor("f")[...]
    lk_min_dis2sigma: Tensor("f")[...]

    lkb_water_dist: Tensor("f")[...]
    lkb_water_angle_sp2: Tensor("f")[...]
    lkb_water_angle_sp3: Tensor("f")[...]
    lkb_water_angle_ring: Tensor("f")[...]
    lkb_water_tors_sp2: Tensor("f")[..., :]
    lkb_water_tors_sp3: Tensor("f")[..., :]
    lkb_water_tors_ring: Tensor("f")[..., :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LJLKTypeParams(TensorGroup, ValidateAttrs):
    lj_radius: Tensor("f")[...]
    lj_wdepth: Tensor("f")[...]
    lk_dgfree: Tensor("f")[...]
    lk_lambda: Tensor("f")[...]
    lk_volume: Tensor("f")[...]

    acceptor_hybridization: Tensor("i4")[...]
    is_acceptor: Tensor(bool)[...]
    is_donor: Tensor(bool)[...]
    is_hydroxyl: Tensor(bool)[...]
    is_polarh: Tensor(bool)[...]


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

    def type_idx(self, atom_types: NDArray(object)[...]) -> NDArray("i8")[...]:
        """Convert array of atom type names to parameter indices.

        pandas.Index.get_indexer only operates on 1-d input arrays. Coerces
        higher-dimensional arrays, as may be produced via broadcasting, into
        lower-dimensional views to resolver parameter indices.
        """
        if not isinstance(atom_types, numpy.ndarray):
            atom_types = numpy.array(atom_types, dtype=object)
        return self.atom_type_index.get_indexer(atom_types.ravel()).reshape(
            atom_types.shape
        )

    @classmethod
    @validate_args
    def from_database(cls, ljlk_database: LJLKDatabase, device: torch.device):
        """Initialize param resolver for all atom types in database."""

        # Generate a full atom type index, appending a "None" value at index -1
        # to generate nan parameter entries if an atom type is not present in
        # the index.
        atom_type_names = [p.name for p in ljlk_database.atom_type_parameters]
        atom_type_index = pandas.Index(atom_type_names + [None])

        # Convert float entries into 1-d tensors
        global_params = LJLKGlobalParams(
            **{
                n: torch.tensor(v, device=device)
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

        # Convert boolean types to uint8 for torch, setting the invalid/None
        # entry to 0/false
        for field in attr.fields(LJLKTypeParams):
            if field.type.dtype == torch.uint8:
                (param_records[field.name]) = (
                    param_records[field.name].fillna(value=0).astype("u1")
                )
        param_records["acceptor_hybridization"] = param_records[
            "acceptor_hybridization"
        ].map(
            lambda v: int(
                AcceptorHybridization.__members__.get(v, AcceptorHybridization.none)
            )
        )

        # Convert the param record dataframe into typed TensorGroup
        type_params = LJLKTypeParams(
            **{
                f.name: torch.tensor(
                    param_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(LJLKTypeParams)
            }
        )

        return cls(
            atom_type_index=atom_type_index,
            global_params=global_params,
            type_params=type_params,
            device=device,
        )
