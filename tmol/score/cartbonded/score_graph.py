import attr
from typing import Optional

import torch

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from tmol.database import ParameterDatabase
from tmol.database.scoring import CartBondedDatabase
from .identification import CartBondedIdentification
from .params import CartBondedParamResolver
from .torch_op import (
    CartBondedLengthOp,
    CartBondedAngleOp,
    CartBondedTorsionOp,
    CartBondedImproperOp,
    CartBondedHxlTorsionOp,
)


from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup


@attr.s(auto_attribs=True)
class CartBondedLengthParams(TensorGroup):
    atom_indices: Tensor(torch.int32)[..., 2]
    param_indices: Tensor(torch.int32)[...]


@attr.s(auto_attribs=True)
class CartBondedAngleParams(TensorGroup):
    atom_indices: Tensor(torch.int32)[..., 3]
    param_indices: Tensor(torch.int32)[...]


# all of torsion/imroper/hxltorsion use this struct
@attr.s(auto_attribs=True)
class CartBondedTorsionParams(TensorGroup):
    atom_indices: Tensor(torch.int32)[..., 4]
    param_indices: Tensor(torch.int32)[...]


@reactive_attrs
class CartBondedIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def total_cartbonded_length(cartbonded_length):
        """total cartbonded length score"""
        score_val = cartbonded_length
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded_length(target):
        return target.cartbonded_length_op.score(
            target.coords[0, ...],
            target.cartbonded_lengths.atom_indices,
            target.cartbonded_lengths.param_indices,
        )

    @reactive_property
    @validate_args
    def total_cartbonded_angle(cartbonded_angle):
        """total cartbonded angle score"""
        score_val = cartbonded_angle
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded_angle(target):
        return target.cartbonded_angle_op.score(
            target.coords[0, ...],
            target.cartbonded_angles.atom_indices,
            target.cartbonded_angles.param_indices,
        )

    @reactive_property
    @validate_args
    def total_cartbonded_torsion(cartbonded_torsion):
        """total cartbonded torsion score"""
        score_val = cartbonded_torsion
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded_torsion(target):
        return target.cartbonded_torsion_op.score(
            target.coords[0, ...],
            target.cartbonded_torsions.atom_indices,
            target.cartbonded_torsions.param_indices,
        )

    @reactive_property
    @validate_args
    def total_cartbonded_improper(cartbonded_improper):
        """total cartbonded improper score"""
        score_val = cartbonded_improper
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded_improper(target):
        return target.cartbonded_improper_op.score(
            target.coords[0, ...],
            target.cartbonded_impropers.atom_indices,
            target.cartbonded_impropers.param_indices,
        )

    @reactive_property
    @validate_args
    def total_cartbonded_hxltorsion(cartbonded_hxltorsion):
        """total cartbonded hxltorsion score"""
        score_val = cartbonded_hxltorsion
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded_hxltorsion(target):
        if target.cartbonded_hxltorsions.atom_indices.shape[0] == 0:
            return torch.tensor([], device=target.device)
        else:
            return target.cartbonded_hxltorsion_op.score(
                target.coords[0, ...],
                target.cartbonded_hxltorsions.atom_indices,
                target.cartbonded_hxltorsions.param_indices,
            )


@score_graph
class CartBondedScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    """Compute graph for the CartBonded term.
    """

    total_score_components = [
        ScoreComponentClasses(
            "cartbonded_length",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_angle",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_torsion",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_improper",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_hxltorsion",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        cartbonded_database: Optional[CartBondedDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.
        """

        if cartbonded_database is None:
            if getattr(val, "cartbonded_database", None):
                cartbonded_database = val.cartbonded_database
            else:
                cartbonded_database = parameter_database.scoring.cartbonded

        return dict(cartbonded_database=cartbonded_database)

    cartbonded_database: CartBondedDatabase

    @reactive_property
    @validate_args
    def cartbonded_param_resolver(
        cartbonded_database: CartBondedDatabase, device: torch.device
    ) -> CartBondedParamResolver:
        "cartbonded tuple resolver"
        return CartBondedParamResolver.from_database(cartbonded_database, device)

    @reactive_property
    @validate_args
    def cartbonded_param_identifier(
        cartbonded_database: CartBondedDatabase, bonds: NDArray(int)[:, 3]
    ) -> CartBondedIdentification:
        return CartBondedIdentification.setup(
            cartbonded_database=cartbonded_database, bonds=bonds[:, 1:]
        )

    @reactive_property
    @validate_args
    def cartbonded_length_op(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedLengthOp:
        return CartBondedLengthOp.from_param_resolver(cartbonded_param_resolver)

    @reactive_property
    @validate_args
    def cartbonded_angle_op(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedAngleOp:
        return CartBondedAngleOp.from_param_resolver(cartbonded_param_resolver)

    @reactive_property
    @validate_args
    def cartbonded_torsion_op(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedTorsionOp:
        return CartBondedTorsionOp.from_param_resolver(cartbonded_param_resolver)

    @reactive_property
    @validate_args
    def cartbonded_improper_op(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedImproperOp:
        return CartBondedImproperOp.from_param_resolver(cartbonded_param_resolver)

    @reactive_property
    @validate_args
    def cartbonded_hxltorsion_op(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedHxlTorsionOp:
        return CartBondedHxlTorsionOp.from_param_resolver(cartbonded_param_resolver)

    @reactive_property
    @validate_args
    def cartbonded_lengths(
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedLengthParams:
        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = cartbonded_param_identifier.lengths
        bondlength_indices = cartbonded_param_resolver.resolve_lengths(
            res_names[0, bondlength_atom_indices[:, 0]],  # use atm1 for resid
            atom_names[0, bondlength_atom_indices[:, 0]],
            atom_names[0, bondlength_atom_indices[:, 1]],
        )
        # remove undefined indices
        bondlength_defined = bondlength_indices != -1
        tbondlength_atom_indices = torch.from_numpy(
            bondlength_atom_indices[bondlength_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        tbondlength_indices = torch.from_numpy(
            bondlength_indices[bondlength_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)

        return CartBondedLengthParams(
            atom_indices=tbondlength_atom_indices, param_indices=tbondlength_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_angles(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedAngleParams:
        # combine resolved atom indices and bondangle indices
        bondangle_atom_indices = cartbonded_param_identifier.angles
        bondangle_indices = cartbonded_param_resolver.resolve_angles(
            res_names[0, bondangle_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[0, bondangle_atom_indices[:, 0]],
            atom_names[0, bondangle_atom_indices[:, 1]],
            atom_names[0, bondangle_atom_indices[:, 2]],
        )

        # remove undefined indices
        bondangle_defined = bondangle_indices != -1
        tbondangle_atom_indices = torch.from_numpy(
            bondangle_atom_indices[bondangle_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        tbondangle_indices = torch.from_numpy(bondangle_indices[bondangle_defined]).to(
            device=cartbonded_param_resolver.device, dtype=torch.int32
        )

        return CartBondedAngleParams(
            atom_indices=tbondangle_atom_indices, param_indices=tbondangle_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_torsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # combine resolved atom indices and bondangle indices
        torsion_atom_indices = cartbonded_param_identifier.torsions
        torsion_indices = cartbonded_param_resolver.resolve_torsions(
            res_names[0, torsion_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[0, torsion_atom_indices[:, 0]],
            atom_names[0, torsion_atom_indices[:, 1]],
            atom_names[0, torsion_atom_indices[:, 2]],
            atom_names[0, torsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        torsion_defined = torsion_indices != -1
        ttorsion_atom_indices = torch.from_numpy(
            torsion_atom_indices[torsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        ttorsion_indices = torch.from_numpy(torsion_indices[torsion_defined]).to(
            device=cartbonded_param_resolver.device, dtype=torch.int32
        )

        return CartBondedTorsionParams(
            atom_indices=ttorsion_atom_indices, param_indices=ttorsion_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_impropers(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # combine resolved atom indices and bondangle indices
        improper_atom_indices = cartbonded_param_identifier.impropers
        improper_indices = cartbonded_param_resolver.resolve_impropers(
            res_names[0, improper_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[0, improper_atom_indices[:, 0]],
            atom_names[0, improper_atom_indices[:, 1]],
            atom_names[0, improper_atom_indices[:, 2]],
            atom_names[0, improper_atom_indices[:, 3]],
        )

        # remove undefined indices
        improper_defined = improper_indices != -1
        timproper_atom_indices = torch.from_numpy(
            improper_atom_indices[improper_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        timproper_indices = torch.from_numpy(improper_indices[improper_defined]).to(
            device=cartbonded_param_resolver.device, dtype=torch.int32
        )

        return CartBondedTorsionParams(
            atom_indices=timproper_atom_indices, param_indices=timproper_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_hxltorsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # same identification as regular torsions, but resolved against a different DB
        hxltorsion_atom_indices = cartbonded_param_identifier.torsions
        hxltorsion_indices = cartbonded_param_resolver.resolve_hxltorsions(
            res_names[0, hxltorsion_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[0, hxltorsion_atom_indices[:, 0]],
            atom_names[0, hxltorsion_atom_indices[:, 1]],
            atom_names[0, hxltorsion_atom_indices[:, 2]],
            atom_names[0, hxltorsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        hxltorsion_defined = hxltorsion_indices != -1
        thxltorsion_atom_indices = torch.from_numpy(
            hxltorsion_atom_indices[hxltorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        thxltorsion_indices = torch.from_numpy(
            hxltorsion_indices[hxltorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)

        return CartBondedTorsionParams(
            atom_indices=thxltorsion_atom_indices, param_indices=thxltorsion_indices
        )
