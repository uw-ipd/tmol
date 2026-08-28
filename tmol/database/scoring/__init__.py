"""Typed schemas and loaders for TMol scoring parameters."""

from ._cartbonded import (  # noqa: F401
    AngleGroup,
    CartRes,
    HxlTorsionGroup,
    ImproperGroup,
    LengthGroup,
    TorsionGroup,
    CartBondedDatabase,
)
from ._disulfide import (  # noqa: F401
    DisulfideGlobalParameters,
    DisulfideDatabase,
)
from ._dunbrack_libraries import (  # noqa: F401
    DunMappingParams,
    RotamericAADunbrackLibrary,
    RotamericDataForAA,
    SemiRotamericAADunbrackLibrary,
    DunbrackRotamerLibrary,
)
from ._elec import (  # noqa: F401
    CountPairReps,
    GlobalParams,
    PartialCharges,
    ElecDatabase,
)
from ._genbonded import (  # noqa: F401
    GenBondedImproperEntry,
    GenBondedTorsionEntry,
    GenBondedDatabase,
)
from ._hbond import (  # noqa: F401
    AcceptorAtomType,
    AcceptorTypeParam,
    DonorAtomType,
    DonorTypeParam,
    HBondDatabaseRaw,
    PairParameters,
    PolynomialParameters,
    HBondDatabase,
)
from ._ljlk import (  # noqa: F401
    LJLKAtomTypeParameters,
    LJLKGlobalParameters,
    LJLKDatabase,
)
from ._na_torsion import (  # noqa: F401
    NaTorsionGlobalParams,
    NaTorsionWells,
    NaTorsionDatabase,
)
from ._omega_bbdep import (  # noqa: F401
    OmegaBBDepMappingParams,
    OmegaBBDepTables,
    OmegaBBDepDatabase,
)
from ._rama import (  # noqa: F401
    RamaMappingParams,
    RamaTables,
    RamaDatabase,
)

import os
import attr

from ._ref import RefDatabase  # noqa: F401


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    """Immutable parameters for all standard TMol energy terms."""

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
    def from_file(
        cls, path: str | os.PathLike[str] = os.path.dirname(__file__)
    ) -> "ScoringDatabase":
        """Load all scoring-term databases rooted at ``path``."""

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
