import attr

import torch
import numpy
from typing import List

from ..database import ParamDB
from ..device import TorchDevice
from ..total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from ..factory import Factory
from ..residue_properties import ResidueProperties
from ..torsions import AlphaAABackboneTorsionProvider
from ..polymeric_bonds import PolymericBonds

from tmol.database import ParameterDatabase

# from tmol.database.chemical import AAType
from tmol.database.scoring.rama import CompactedRamaDatabase

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor


@reactive_attrs(auto_attribs=True)
class RamaScoreGraph(
    AlphaAABackboneTorsionProvider,
    ResidueProperties,
    TotalScoreComponentsGraph,
    PolymericBonds,
    ParamDB,
    TorchDevice,
    Factory,
):
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

    @property
    def component_total_score_terms(self):
        """Expose rama score sum as total_score term."""
        return ScoreComponentAttributes(name="rama", total="total_rama")

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
        """total ramachandran score"""
        return rama_scores.sum()
