import attr
import cattr

import numpy
import pandas
import torch

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.cartbonded import CartBondedDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartHarmonicParams(TensorGroup, ConvertAttrs):
    K: Tensor(torch.float)[...]
    x0: Tensor(torch.float)[...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartSimpleSinusoidalParams(TensorGroup, ConvertAttrs):
    K: Tensor(torch.float)[...]
    x0: Tensor(torch.float)[...]
    period: Tensor(torch.float)[...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CartSinusoidalParams(TensorGroup, ValidateAttrs):
    k1: Tensor(torch.float)[...]
    k2: Tensor(torch.float)[...]
    k3: Tensor(torch.float)[...]
    phi1: Tensor(torch.float)[...]
    phi2: Tensor(torch.float)[...]
    phi3: Tensor(torch.float)[...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CartBondedParamResolver(ValidateAttrs):
    bondlength_index: pandas.Index
    bondangle_index: pandas.Index
    torsion_index: pandas.Index
    improper_index: pandas.Index
    hxltorsion_index: pandas.Index

    bondlength_params: CartHarmonicParams
    bondangle_params: CartHarmonicParams
    torsion_params: CartSimpleSinusoidalParams
    improper_params: CartSimpleSinusoidalParams
    hxltorsion_params: CartSinusoidalParams

    device: torch.device

    @validate_args
    def resolve_lengths(
        self,
        resnames: NDArray(object)[:, :],
        atm1s: NDArray(object)[:, :],
        atm2s: NDArray(object)[:, :],
    ) -> NDArray("i8")[...]:
        """Resolve string triplets into integer bondlength parameter indices."""
        assert resnames.shape[0] == atm1s.shape[0]
        assert resnames.shape[0] == atm2s.shape[0]
        assert resnames.shape[1] == atm1s.shape[1]
        assert resnames.shape[1] == atm2s.shape[1]

        inds = numpy.full((resnames.shape[0], resnames.shape[1]), -9999, dtype=numpy.int64)
        real = resnames.astype(bool)
        inds[real] = self.bondlength_index.get_indexer([resnames[real], atm1s[real], atm2s[real]])
        inds[inds == -1] = self.bondlength_index.get_indexer([resnames[inds==-1], atm2s[inds==-1], atm1s[inds==-1]])
        wildcard = numpy.full_like(resnames, "_")
        inds[inds == -1] = self.bondlength_index.get_indexer(
            [wildcard[inds == -1], atm1s[inds == -1], atm2s[inds == -1]])
        inds[inds == -1] = self.bondlength_index.get_indexer(
            [wildcard[inds == -1], atm2s[inds == -1], atm1s[inds == -1]])
        inds[inds == -9999] = -1

        return inds

    @validate_args
    def resolve_angles(
        self,
        resnames: NDArray(object)[:, :],
        atm1s: NDArray(object)[:, :],
        atm2s: NDArray(object)[:, :],
        atm3s: NDArray(object)[:, :],
    ) -> NDArray("i8")[...]:
        """Resolve string quads into integer bondangle parameter indices."""

        inds = numpy.full(resnames.shape[0:2], -9999, dtype=numpy.int64)
        real = resnames.astype(bool)
        inds[real] = self.bondangle_index.get_indexer([resnames[real], atm1s[real], atm2s[real], atm3s[real]])
        inds[inds == -1] = self.bondangle_index.get_indexer(
            [
                resnames[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )
        wildcard = numpy.full_like(resnames, "_")
        inds[inds == -1] = self.bondangle_index.get_indexer(
            [
                wildcard[inds == -1],
                atm1s[inds == -1],
                atm2s[inds == -1],
                atm3s[inds == -1],
            ]
        )
        inds[inds == -1] = self.bondangle_index.get_indexer(
            [
                wildcard[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )

        inds[inds == -9999] = -1
        return inds

    @validate_args
    def resolve_torsions(
        self,
        resnames: NDArray(object)[:, :],
        atm1s: NDArray(object)[:, :],
        atm2s: NDArray(object)[:, :],
        atm3s: NDArray(object)[:, :],
        atm4s: NDArray(object)[:, :],
    ) -> NDArray("i8")[...]:
        """Resolve string quints into integer torsion parameter indices."""

        inds = numpy.full_like(resnames, -9999, dtype=numpy.int64)
        real = resnames.astype(bool)
        inds[real] = self.torsion_index.get_indexer(
            [resnames[real], atm1s[real], atm2s[real], atm3s[real], atm4s[real]]
        )
        inds[inds == -1] = self.torsion_index.get_indexer(
            [
                resnames[inds == -1],
                atm4s[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )
        wildcard = numpy.full_like(resnames, "_")
        inds[inds == -1] = self.torsion_index.get_indexer(
            [
                wildcard[inds == -1],
                atm1s[inds == -1],
                atm2s[inds == -1],
                atm3s[inds == -1],
                atm4s[inds == -1],
            ]
        )
        inds[inds == -1] = self.torsion_index.get_indexer(
            [
                wildcard[inds == -1],
                atm4s[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )

        inds[inds == -9999] = -1
        return inds

    @validate_args
    def resolve_impropers(
        self,
        resnames: NDArray(object),
        atm1s: NDArray(object),
        atm2s: NDArray(object),
        atm3s: NDArray(object),
        atm4s: NDArray(object),
    ) -> NDArray("i8")[...]:
        """Resolve string quints into integer improper torsion parameter indices."""
        # impropers have a defined ordering

        inds = numpy.full_like(resnames, -9999, dtype=numpy.int64)
        real = resnames.astype(bool)

        inds[real] = self.improper_index.get_indexer(
            [resnames[real], atm1s[real], atm2s[real], atm3s[real], atm4s[real]]
        )
        wildcard = numpy.full_like(resnames, "_")
        inds[inds == -1] = self.improper_index.get_indexer(
            [
                wildcard[inds == -1],
                atm1s[inds == -1],
                atm2s[inds == -1],
                atm3s[inds == -1],
                atm4s[inds == -1],
            ]
        )

        inds[inds == -9999] = -1
        return inds

    @validate_args
    def resolve_hxltorsions(
        self,
        resnames: NDArray(object),
        atm1s: NDArray(object),
        atm2s: NDArray(object),
        atm3s: NDArray(object),
        atm4s: NDArray(object),
    ) -> NDArray("i8")[...]:
        """Resolve string quints into integer hydroxyl torsion parameter indices."""
        inds = numpy.full_like(resnames, -9999, dtype=numpy.int64)
        real = resnames.astype(bool)
        inds[real] = self.hxltorsion_index.get_indexer(
            [resnames[real], atm1s[real], atm2s[real], atm3s[real], atm4s[real]]
        )
        inds[inds == -1] = self.hxltorsion_index.get_indexer(
            [
                resnames[inds == -1],
                atm4s[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )
        wildcard = numpy.full_like(resnames, "_")
        inds[inds == -1] = self.hxltorsion_index.get_indexer(
            [
                wildcard[inds == -1],
                atm1s[inds == -1],
                atm2s[inds == -1],
                atm3s[inds == -1],
                atm4s[inds == -1],
            ]
        )
        inds[inds == -1] = self.hxltorsion_index.get_indexer(
            [
                wildcard[inds == -1],
                atm4s[inds == -1],
                atm3s[inds == -1],
                atm2s[inds == -1],
                atm1s[inds == -1],
            ]
        )

        inds[inds == -9999] = -1
        return inds

    @classmethod
    @validate_args
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        """Initialize param resolver for all bonded params in database."""

        # 1) cartbonded length params
        length_records = pandas.DataFrame.from_records(
            cattr.unstructure(cb_database.length_parameters)
        )
        bondlength_index = pandas.Index(length_records[["res", "atm1", "atm2"]])
        bondlength_params = CartHarmonicParams(
            **{
                f.name: torch.tensor(
                    length_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(CartHarmonicParams)
            }
        )

        # 2) cartbonded length params
        angle_records = pandas.DataFrame.from_records(
            cattr.unstructure(cb_database.angle_parameters)
        )
        bondangle_index = pandas.Index(angle_records[["res", "atm1", "atm2", "atm3"]])
        bondangle_params = CartHarmonicParams(
            **{
                f.name: torch.tensor(
                    angle_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(CartHarmonicParams)
            }
        )

        # 3) cartbonded torsion params
        torsion_records = pandas.DataFrame.from_records(
            cattr.unstructure(cb_database.torsion_parameters)
        )
        torsion_index = pandas.Index(
            torsion_records[["res", "atm1", "atm2", "atm3", "atm4"]]
        )
        torsion_params = CartSimpleSinusoidalParams(
            **{
                f.name: torch.tensor(
                    torsion_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(CartSimpleSinusoidalParams)
            }
        )

        # 4) cartbonded improper params
        improper_records = pandas.DataFrame.from_records(
            cattr.unstructure(cb_database.improper_parameters)
        )
        improper_index = pandas.Index(
            improper_records[["res", "atm1", "atm2", "atm3", "atm4"]]
        )
        improper_params = CartSimpleSinusoidalParams(
            **{
                f.name: torch.tensor(
                    improper_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(CartSimpleSinusoidalParams)
            }
        )

        # 5) cartbonded hxltorsion params
        hxltorsion_records = pandas.DataFrame.from_records(
            cattr.unstructure(cb_database.hxltorsion_parameters)
        )
        hxltorsion_index = pandas.Index(
            hxltorsion_records[["res", "atm1", "atm2", "atm3", "atm4"]]
        )
        hxltorsion_params = CartSinusoidalParams(
            **{
                f.name: torch.tensor(
                    hxltorsion_records[f.name].values, dtype=f.type.dtype, device=device
                )
                for f in attr.fields(CartSinusoidalParams)
            }
        )

        return cls(
            bondlength_index=bondlength_index,
            bondlength_params=bondlength_params,
            bondangle_index=bondangle_index,
            bondangle_params=bondangle_params,
            torsion_index=torsion_index,
            torsion_params=torsion_params,
            improper_index=improper_index,
            improper_params=improper_params,
            hxltorsion_index=hxltorsion_index,
            hxltorsion_params=hxltorsion_params,
            device=device,
        )
