from ._cartbonded import (  # noqa: F401
    AngleGroup,
    CartRes,
    HxlTorsionGroup,
    ImproperGroup,
    LengthGroup,
    TorsionGroup,
)  # noqa: F401
from ._disulfide import DisulfideGlobalParameters  # noqa: F401
from ._dunbrack_libraries import (  # noqa: F401
    DunMappingParams,
    RotamericAADunbrackLibrary,
    RotamericDataForAA,
    SemiRotamericAADunbrackLibrary,
)  # noqa: F401
from ._elec import CountPairReps, GlobalParams, PartialCharges  # noqa: F401
from ._genbonded import GenBondedImproperEntry, GenBondedTorsionEntry  # noqa: F401
from ._hbond import (  # noqa: F401
    AcceptorAtomType,
    AcceptorTypeParam,
    DonorAtomType,
    DonorTypeParam,
    HBondDatabaseRaw,
    PairParameters,
    PolynomialParameters,
)  # noqa: F401
from ._ljlk import LJLKAtomTypeParameters, LJLKGlobalParameters  # noqa: F401
from ._na_torsion import NaTorsionGlobalParams, NaTorsionWells  # noqa: F401
from ._omega_bbdep import OmegaBBDepMappingParams, OmegaBBDepTables  # noqa: F401
from ._rama import RamaMappingParams, RamaTables  # noqa: F401

import os
import attr

from ._cartbonded import CartBondedDatabase  # noqa: F401
from ._genbonded import GenBondedDatabase  # noqa: F401
from ._disulfide import DisulfideDatabase  # noqa: F401
from ._na_torsion import NaTorsionDatabase  # noqa: F401
from ._dunbrack_libraries import DunbrackRotamerLibrary  # noqa: F401
from ._elec import ElecDatabase  # noqa: F401
from ._hbond import HBondDatabase  # noqa: F401
from ._ljlk import LJLKDatabase  # noqa: F401
from ._omega_bbdep import OmegaBBDepDatabase  # noqa: F401
from ._rama import RamaDatabase  # noqa: F401
from ._ref import RefDatabase  # noqa: F401


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    cartbonded: CartBondedDatabase
    genbonded: GenBondedDatabase
    disulfide: DisulfideDatabase
    na_torsion: NaTorsionDatabase
    dun: DunbrackRotamerLibrary
    elec: ElecDatabase
    hbond: HBondDatabase
    ljlk: LJLKDatabase
    omega_bbdep: OmegaBBDepDatabase
    rama: RamaDatabase
    ref: RefDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):  # noqa

        return cls(
            cartbonded=CartBondedDatabase.from_file(
                os.path.join(path, "cartbonded.yaml")
            ),
            genbonded=GenBondedDatabase.from_file(os.path.join(path, "genbonded.yaml")),
            disulfide=DisulfideDatabase.from_file(os.path.join(path, "disulfide.yaml")),
            na_torsion=NaTorsionDatabase.from_file(
                os.path.join(path, "na_torsion.yaml")
            ),
            dun=DunbrackRotamerLibrary.from_file(os.path.join(path, "dunbrack.bin")),
            elec=ElecDatabase.from_file(os.path.join(path, "elec.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            omega_bbdep=OmegaBBDepDatabase.from_file(
                os.path.join(path, "omega_bbdep.zip")
            ),
            rama=RamaDatabase.from_file(os.path.join(path, "rama.zip")),
            ref=RefDatabase.from_file(os.path.join(path, "ref.yaml")),
        )
