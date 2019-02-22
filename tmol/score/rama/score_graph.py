import attr

import torch
import numpy
import attr
from typing import Optional

import torch

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from tmol.database import ParameterDatabase
from tmol.database.scoring import RamaDatabase
from .identification import RamaIdentification
from .params import RamaParamResolver
from .torch_op import RamaOp


@attr.s(auto_attribs=True)
class RamaParams(TensorGroup):
    atom_indices: Tensor(torch.int32)[..., 4]
    param_indices: Tensor(torch.int32)[...]


@reactive_attrs
class RamaIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def rama_db(target) -> CompactedRamaDatabase:
        return target.rama_db

    @reactive_property
    @validate_args
    def rama_table_inds(target) -> Tensor(torch.long)[:, :]:
        return target.rama_table_inds

    @reactive_property
    @validate_args
    def phi_tor(target) -> Tensor(torch.float)[:, :]:
        return target.phi_tor

    @reactive_property
    @validate_args
    def psi_tor(target) -> Tensor(torch.float)[:, :]:
        return target.psi_tor

    @reactive_property
    @validate_args
    def rama_scores(
        rama_db: CompactedRamaDatabase,
        rama_table_inds: Tensor(torch.long)[:, :],
        phi_tor: Tensor(torch.float)[:, :],
        psi_tor: Tensor(torch.float)[:, :],
    ) -> Tensor(torch.float)[:, :]:
        assert rama_table_inds.shape[0] == 1
        assert phi_tor.shape[0] == 1
        assert psi_tor.shape[0] == 1

        has_rama = (
            ~torch.isnan(phi_tor) & ~torch.isnan(psi_tor) & (rama_table_inds != -1)
        )
        phi_psi = torch.cat(
            (phi_tor[has_rama].reshape(-1, 1), psi_tor[has_rama].reshape(-1, 1)), dim=1
        )
        # shift range of [-pi,pi) to [0,36)
        phi_psi = (18 / numpy.pi) * phi_psi + 18

        rama_inds = rama_table_inds[has_rama].reshape(-1, 1)
        return rama_db.bspline.interpolate(phi_psi, rama_inds).unsqueeze(0)

    @reactive_property
    @validate_args
    def total_rama(rama_scores: Tensor(torch.float)[:, :]) -> Tensor(torch.float):
        return rama_scores.sum()


@reactive_attrs(auto_attribs=True)
class RamaScoreGraph(
    AlphaAABackboneTorsionProvider,
    ResidueProperties,
    ScoreComponent,
    PolymericBonds,
    ParamDB,
    TorchDevice,
    Factory,
):

    # Data member instructing the ScoreComponent class which classes to construct when
    # attempting to evaluate "one body" vs "two body" energies with the Rama term.
    total_score_components = [
        ScoreComponentClasses(
            "rama", intra_container=RamaIntraScore, inter_container=None
        )
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        residue_properties: List[List[str]],
        upper: Tensor(torch.long)[:],
        **_,
    ):
        """Request the CompoactedRamaDatabase "held" (memoized) in the
        parameter database. We only want a single copy of the parameter
        database to live on the CPU or on the device.
        """
        rama_db = CompactedRamaDatabase.from_ramadb(
            parameter_database.scoring.rama, device
        )

        # Calculate all of the table indices on the CPU, then transfer those
        # indices to the device. Perhaps this can be made to run on the GPU?
        assert upper.shape[1] == len(residue_properties)
        upper_cpu = upper.cpu()
        inds = [-1] * len(residue_properties)
        for i in range(1, len(residue_properties) - 1):
            if upper_cpu[0, i] == -1:
                continue
            i_props = residue_properties[i]
            i_next_props = residue_properties[upper_cpu[0, i]]
            inds[i] = rama_db.mapper.table_ind_for_res(i_props, i_next_props)

        rama_table_inds = torch.tensor(inds, dtype=torch.long, device=device)
        rama_table_inds = rama_table_inds.reshape(1, -1)
        print(rama_table_inds.type())

        return dict(rama_db=rama_db, rama_table_inds=rama_table_inds)

    rama_db: CompactedRamaDatabase = attr.ib()
    rama_table_inds: Tensor(torch.long)[:, :] = attr.ib()

    @reactive_property
    @validate_args
    def total_rama(rama_scores: Tensor(torch.float)[:, :]) -> Tensor(torch.float):
        """total ramachandran score"""
        return rama_scores.sum()
