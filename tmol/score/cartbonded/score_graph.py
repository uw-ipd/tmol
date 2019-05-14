import attr
from typing import Optional

import torch

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph, IndexedBonds
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from tmol.database import ParameterDatabase
from tmol.database.scoring import CartBondedDatabase
from .identification import CartBondedIdentification
from .params import CartBondedParamResolver
from .script_modules import (
    CartBondedLengthModule,
    CartBondedAngleModule,
    CartBondedTorsionModule,
    CartBondedImproperModule,
    CartBondedHxlTorsionModule,
)


from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup


@reactive_attrs
class CartBondedIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def total_cartbonded_length(target):
        """total cartbonded length score"""
        return target.cartbonded_length_module(
            target.coords[0], target.cartbonded_lengths
        )

    @reactive_property
    @validate_args
    def total_cartbonded_angle(target):
        """total cartbonded angle score"""
        return target.cartbonded_angle_module(
            target.coords[0], target.cartbonded_angles
        )

    @reactive_property
    @validate_args
    def total_cartbonded_torsion(target):
        """total cartbonded torsion score"""
        return target.cartbonded_torsion_module(
            target.coords[0], target.cartbonded_torsions
        )

    @reactive_property
    @validate_args
    def total_cartbonded_improper(target):
        """total cartbonded improper score"""
        return target.cartbonded_improper_module(
            target.coords[0], target.cartbonded_impropers
        )

    @reactive_property
    @validate_args
    def total_cartbonded_hxltorsion(target):
        """total cartbonded hxltorsion score"""
        return target.cartbonded_hxltorsion_module(
            target.coords[0], target.cartbonded_hxltorsions
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
    def cartbonded_param_resolver(
        cartbonded_database: CartBondedDatabase, device: torch.device
    ) -> CartBondedParamResolver:
        "cartbonded tuple resolver"
        return CartBondedParamResolver.from_database(cartbonded_database, device)

    @reactive_property
    def cartbonded_param_identifier(
        cartbonded_database: CartBondedDatabase, indexed_bonds: IndexedBonds
    ) -> CartBondedIdentification:
        return CartBondedIdentification.setup(
            cartbonded_database=cartbonded_database, indexed_bonds=indexed_bonds
        )

    @reactive_property
    def cartbonded_length_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedLengthModule:
        return CartBondedLengthModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_angle_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedAngleModule:
        return CartBondedAngleModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_torsion_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedTorsionModule:
        return CartBondedTorsionModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_improper_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedImproperModule:
        return CartBondedImproperModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_hxltorsion_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedHxlTorsionModule:
        return CartBondedHxlTorsionModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_lengths(
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = cartbonded_param_identifier.lengths
        bondlength_indices = cartbonded_param_resolver.resolve_lengths(
            res_names[0, bondlength_atom_indices[:, 0]],  # use atm1 for resid
            atom_names[0, bondlength_atom_indices[:, 0]],
            atom_names[0, bondlength_atom_indices[:, 1]],
        )

        # remove undefined indices
        bondlength_defined = bondlength_indices != -1

        cbl = torch.cat(
            [
                torch.tensor(bondlength_atom_indices[bondlength_defined]),
                torch.tensor(bondlength_indices[bondlength_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbl

    @reactive_property
    def cartbonded_angles(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
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

        cba = torch.cat(
            [
                torch.tensor(bondangle_atom_indices[bondangle_defined]),
                torch.tensor(bondangle_indices[bondangle_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cba

    @reactive_property
    def cartbonded_torsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
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

        cbt = torch.cat(
            [
                torch.tensor(torsion_atom_indices[torsion_defined]),
                torch.tensor(torsion_indices[torsion_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbt

    @reactive_property
    def cartbonded_impropers(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
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

        cbi = torch.cat(
            [
                torch.tensor(improper_atom_indices[improper_defined]),
                torch.tensor(improper_indices[improper_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbi

    @reactive_property
    def cartbonded_hxltorsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
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

        cbht = torch.cat(
            [
                torch.tensor(hxltorsion_atom_indices[hxltorsion_defined]),
                torch.tensor(hxltorsion_indices[hxltorsion_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbht
