import attr
from typing import Optional

import torch
import numpy

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory
from ..score_components import ScoreComponent, ScoreComponentClasses, IntraScore

from tmol.database import ParameterDatabase
from tmol.database.scoring import CartBondedDatabase
from .identification import CartBondedIdentification
from .params import CartBondedParamResolver
from .torch_op import CartBondedOp


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
    atom_indices: Tensor(torch.int32)[..., 2]
    param_indices: Tensor(torch.int32)[...]


# all of torsion/imroper/hxltorsion use this struct
@attr.s(auto_attribs=True)
class CartBondedTorsionParams(TensorGroup):
    atom_indices: Tensor(torch.int32)[..., 4]
    param_indices: Tensor(torch.int32)[...]


@reactive_attrs
class CartBondedLengthScore(IntraScore):
    @reactive_property
    @validate_args
    def total_cartbonded(cartbonded):
        """total hbond score"""
        score_ind, score_val = cartbonded
        return score_val.sum()

    @reactive_property
    @validate_args
    def cartbonded(target):
        return target.cartbonded_length_op.score(
            target.cartbonded_lengths.atom_indices[0][..., 0],
            target.cartbonded_lengths.atom_indices[0][..., 1],
            target.cartbonded_lengths.param_indices[0],
            target.coords,
        )


@reactive_attrs(auto_attribs=True)
class CartBondedGraph(
    BondedAtomScoreGraph, ScoreComponent, ParamDB, TorchDevice, Factory
):
    """Compute graph for the CartBonded term.
    """

    total_score_components = [
        ScoreComponentClasses(
            "cartbonded", intra_container=CartBondedIntraScore, inter_container=None
        )
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        hbond_database: Optional[CartBondedDatabase] = None,
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
        cartbonded_database: CartBondedDatabase,
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedLengthOp:
        return CartBondedLengthOp.from_database(
            cartbonded_database, cartbonded_param_resolver
        )

    @reactive_property
    @validate_args
    def cartbonded_length_indices(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedLengthParams:
        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = cartbonded_param_identifier.lengths
        bondlength_indices = cartbonded_param_resolver.resolve_lengths(
            res_names[bondlength_atom_indices[:, 0]],  # use atm1 for resid
            atom_names[bondlength_atom_indices[:, 0]],
            atom_names[bondlength_atom_indices[:, 1]],
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
    def cartbonded_angle_indices(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedAngleParams:
        # combine resolved atom indices and bondangle indices
        bondangle_atom_indices = cartbonded_param_identifier.angles
        bondangle_indices = cartbonded_param_resolver.resolve_angles(
            res_names[bondangle_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[bondangle_atom_indices[:, 0]],
            atom_names[bondangle_atom_indices[:, 1]],
            atom_names[bondangle_atom_indices[:, 2]],
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
    def cartbonded_torsion_indices(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # combine resolved atom indices and bondangle indices
        bondtorsion_atom_indices = cartbonded_param_identifier.torsions
        bondtorsion_indices = cartbonded_param_resolver.resolve_torsions(
            res_names[bondtorsion_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[bondtorsion_atom_indices[:, 0]],
            atom_names[bondtorsion_atom_indices[:, 1]],
            atom_names[bondtorsion_atom_indices[:, 2]],
            atom_names[bondtorsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        bondtorsion_defined = bondtorsion_indices != -1
        tbondtorsion_atom_indices = torch.from_numpy(
            bondtorsion_atom_indices[bondtorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        tbondtorsion_indices = torch.from_numpy(
            bondtorsion_indices[bondtorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)

        return CartBondedTorsionParams(
            atom_indices=tbondtorsion_atom_indices, param_indices=tbondtorsion_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_improper_indices(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # combine resolved atom indices and bondangle indices
        bondimproper_atom_indices = cartbonded_param_identifier.impropers
        bondimproper_indices = cartbonded_param_resolver.resolve_impropers(
            res_names[bondimproper_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[bondimproper_atom_indices[:, 0]],
            atom_names[bondimproper_atom_indices[:, 1]],
            atom_names[bondimproper_atom_indices[:, 2]],
            atom_names[bondimproper_atom_indices[:, 3]],
        )

        # remove undefined indices
        bondimproper_defined = bondimproper_indices != -1
        tbondimproper_atom_indices = torch.from_numpy(
            bondimproper_atom_indices[bondimproper_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        tbondimproper_indices = torch.from_numpy(
            bondimproper_indices[bondimproper_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)

        return CartBondedTorsionParams(
            atom_indices=tbondimproper_atom_indices, param_indices=tbondimproper_indices
        )

    @reactive_property
    @validate_args
    def cartbonded_hxltorsion_indices(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> CartBondedTorsionParams:
        # combine resolved atom indices and bondangle indices
        bondhxltorsion_atom_indices = cartbonded_param_identifier.hxltorsions
        bondhxltorsion_indices = cartbonded_param_resolver.resolve_hxltorsions(
            res_names[bondhxltorsion_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[bondhxltorsion_atom_indices[:, 0]],
            atom_names[bondhxltorsion_atom_indices[:, 1]],
            atom_names[bondhxltorsion_atom_indices[:, 2]],
            atom_names[bondhxltorsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        bondhxltorsion_defined = bondhxltorsion_indices != -1
        tbondhxltorsion_atom_indices = torch.from_numpy(
            bondhxltorsion_atom_indices[bondhxltorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)
        tbondhxltorsion_indices = torch.from_numpy(
            bondhxltorsion_indices[bondhxltorsion_defined]
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int32)

        return CartBondedTorsionParams(
            atom_indices=tbondhxltorsion_atom_indices,
            param_indices=tbondhxltorsion_indices,
        )
