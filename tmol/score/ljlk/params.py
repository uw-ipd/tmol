import attr
import cattr

import pandas

import torch
import numpy


from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.ljlk import LJLKDatabase
from tmol.database.chemical import ChemicalDatabase

from ..chemical_database import AtomTypeParamResolver


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParams(TensorGroup):
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
class LJLKTypeParams(TensorGroup):
    lj_radius: Tensor("f")[...]
    lj_wdepth: Tensor("f")[...]
    lk_dgfree: Tensor("f")[...]
    lk_lambda: Tensor("f")[...]
    lk_volume: Tensor("f")[...]

    # Parameters imported from chemical database
    is_acceptor: Tensor(bool)[...]
    acceptor_hybridization: Tensor("i4")[...]

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

    def type_idx(self, atom_types: NDArray(object)[...]) -> Tensor("i8")[...]:
        """Convert array of atom type names to parameter indices.

        pandas.Index.get_indexer only operates on 1-d input arrays. Coerces
        higher-dimensional arrays, as may be produced via broadcasting, into
        lower-dimensional views to resolver parameter indices.
        """
        if not isinstance(atom_types, numpy.ndarray):
            atom_types = numpy.array(atom_types, dtype=object)

        return torch.from_numpy(
            self.atom_type_index.get_indexer(atom_types.ravel()).reshape(
                atom_types.shape
            )
        ).to(self.device)

    @classmethod
    @validate_args
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
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
        )

        return cls(
            atom_type_index=atom_type_index,
            global_params=global_params,
            type_params=type_params,
            device=device,
        )
