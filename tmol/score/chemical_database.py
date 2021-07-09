import typing

import attr
import cattr

import pandas

import torch
import numpy

from tmol.database import ParameterDatabase
from tmol.database.chemical import ChemicalDatabase

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs

from tmol.utility.reactive import reactive_property
from .score_graph import score_graph

from .database import ParamDB
from .device import TorchDevice

from enum import IntEnum


class AcceptorHybridization(IntEnum):
    none = 0
    sp2 = 1
    sp3 = 2
    ring = 3

    _index: typing.ClassVar[pandas.Index]


AcceptorHybridization._index = pandas.Index([None, "sp2", "sp3", "ring"])


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomTypeParams(TensorGroup, ValidateAttrs):
    is_acceptor: Tensor[bool][...]
    acceptor_hybridization: Tensor[torch.int32][...]

    is_donor: Tensor[bool][...]

    is_hydrogen: Tensor[bool][...]
    is_hydroxyl: Tensor[bool][...]
    is_polarh: Tensor[bool][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class AtomTypeParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    # str->int index from atom type name to index within parameter tensors
    index: pandas.Index

    # shape [n] per-type parameters, source params used to calculate pair parameters
    params: AtomTypeParams

    device: torch.device

    def type_idx(self, atom_types: NDArray[object][...]) -> Tensor[torch.int64][...]:
        """Convert array of atom type names to parameter indices.

        pandas.Index.get_indexer only operates on 1-d input arrays. Coerces
        higher-dimensional arrays, as may be produced via broadcasting, into
        lower-dimensional views to resolver parameter indices.
        """
        if not isinstance(atom_types, numpy.ndarray):
            atom_types = numpy.array(atom_types, dtype=object)

        return torch.from_numpy(
            self.index.get_indexer(atom_types.ravel()).reshape(atom_types.shape)
        ).to(self.device)

    @classmethod
    def from_database(cls, chemical_database: ChemicalDatabase, device: torch.device):
        """Initialize param resolver for all atom types in database."""

        # Generate a full atom type index, appending a "None" value at index -1
        # to generate nan parameter entries if an atom type is not present in
        # the index.
        atom_type_names = [p.name for p in chemical_database.atom_types]
        atom_type_index = pandas.Index(atom_type_names + [None])

        # Pack the tuple of type parameters into a dataframe and reindex via
        # the param resolver type index. This appends a "nan" row at the end of
        # the frame for the invalid/None entry added above.
        param_records = (
            pandas.DataFrame.from_records(
                cattr.unstructure(chemical_database.atom_types)
            )
            .set_index("name")
            .reindex(index=atom_type_index)
        )

        # Map element to is_hydrogen flag
        param_records["is_hydrogen"] = param_records["element"] == "H"

        # Map acceptor hybridization from string space to index space
        param_records["acceptor_hybridization"].loc[None] = None

        param_records[
            "acceptor_hybridization"
        ] = AcceptorHybridization._index.get_indexer_for(
            param_records["acceptor_hybridization"]
        )

        # Convert boolean types for torch, setting the invalid/None
        # entry to 0/false
        for field in attr.fields(AtomTypeParams):
            if field.type.dtype == torch.bool:
                (param_records[field.name]) = (
                    param_records[field.name].fillna(value=0).astype(bool)
                )

        assert not (param_records["acceptor_hybridization"] == -1).any()

        # Convert the param record dataframe into typed TensorGroup
        atom_type_params = AtomTypeParams(
            **{
                f.name: torch.tensor(
                    param_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(AtomTypeParams)
            }
        )

        return cls(index=atom_type_index, params=atom_type_params, device=device)


@score_graph
class ChemicalDB(ParamDB, TorchDevice):
    """Graph component for chemical parameter dispatch."""

    @reactive_property
    def atom_type_params(parameter_database: ParameterDatabase, device: torch.device):
        return AtomTypeParamResolver.from_database(parameter_database.chemical, device)
