import attr

import torch
import numpy

from ..database import ParamDB
from ..device import TorchDevice
from ..total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from ..factory import Factory
from ..torsions import AlphaAABackboneTorsionProvider
from ..polymeric_bonds import PolymericBonds

from tmol.database import ParameterDatabase
from tmol.database.chemical import AAType
from tmol.database.scoring.rama import CompactedRamaDatabase

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor


@reactive_attrs(auto_attribs=True)
class RamaScoreGraph(
    AlphaAABackboneTorsionProvider,
    TotalScoreComponentsGraph,
    PolymericBonds,
    ParamDB,
    TorchDevice,
    Factory,
):
    @staticmethod
    def factory_for(
        val, parameter_database: ParameterDatabase, device: torch.device, **_
    ):
        """Request the CompoactedRamaDatabase "held" (memoized) in the
        parameter database. We only want a single copy of the parameter
        database to live on the CPU or on the device.
        """
        rama_db = CompactedRamaDatabase.from_ramadb(
            parameter_database.scoring.rama, device
        )
        return dict(rama_db=rama_db)

    rama_db: CompactedRamaDatabase = attr.ib()

    @rama_db.default
    def _default_rama_database(self):
        return self.parameter_database.scoring.get_compacted_rama_database(
            torch.device("cpu")
        )

    @property
    def component_total_score_terms(self):
        """Expose rama score sum as total_score term."""
        return ScoreComponentAttributes(name="rama", total="total_rama")

    @reactive_property
    @validate_args
    def rama_scores(
        rama_db: CompactedRamaDatabase,
        upper: Tensor(torch.long)[:],
        res_aas: Tensor(torch.long)[:],
        phi_tor: Tensor(torch.float)[:],
        psi_tor: Tensor(torch.float)[:],
    ) -> Tensor(torch.float):
        has_rama = ~torch.isnan(phi_tor) & ~torch.isnan(psi_tor)
        phi_psi = torch.cat(
            (phi_tor[has_rama].reshape(-1, 1), psi_tor[has_rama].reshape(-1, 1)), dim=1
        )
        # shift range of [-pi,pi) to [0,36)
        phi_psi = (18 / numpy.pi) * phi_psi + 18

        has_upper = upper != -1
        upper_is_pro = (res_aas[upper[has_upper & has_rama]] == AAType.aa_pro).type(
            torch.long
        )
        rama_inds = torch.cat(
            (
                (res_aas[has_upper & has_rama]).reshape(-1, 1),
                upper_is_pro.reshape(-1, 1),
            ),
            dim=1,
        )
        return rama_db.bspline.interpolate(phi_psi, rama_inds)

    @reactive_property
    @validate_args
    def total_rama(rama_scores: Tensor(torch.float)[:]) -> Tensor(torch.float):
        """total ramachandran score"""
        return rama_scores.sum()
