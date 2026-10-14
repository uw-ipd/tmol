"""Typed schemas and loaders for TMol scoring parameters."""

from ._cartbonded import CartBondedDatabase  # noqa: F401
from ._disulfide import DisulfideDatabase  # noqa: F401
from ._dunbrack_libraries import DunbrackRotamerLibrary  # noqa: F401
from ._elec import ElecDatabase  # noqa: F401
from ._genbonded import GenBondedDatabase  # noqa: F401
from ._hbond import HBondDatabase  # noqa: F401
from ._ljlk import LJLKDatabase  # noqa: F401
from ._metal_coordination import MetalCoordinationDatabase  # noqa: F401
from ._na_torsion import NaTorsionDatabase  # noqa: F401
from ._omega_bbdep import OmegaBBDepDatabase  # noqa: F401
from ._rama import RamaDatabase  # noqa: F401

import os
import attr

from ._ref import RefDatabase  # noqa: F401

# residue sets written by a support script, merged into the databases at load
GENERATED_ELEC_FILES = (
    "elec_d_amino_acids.yaml",
    "elec_metal_ions.yaml",
    "elec_metal_clusters.yaml",
)
GENERATED_CARTBONDED_FILES = (
    "cartbonded_d_amino_acids.yaml",
    "cartbonded_metal_clusters.yaml",
)
GENERATED_REF_FILES = ("ref_d_amino_acids.yaml",)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    """Immutable parameters for all standard TMol energy terms."""

    cartbonded: CartBondedDatabase
    genbonded: GenBondedDatabase
    disulfide: DisulfideDatabase
    metal_coordination: MetalCoordinationDatabase
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
                os.path.join(path, "cartbonded.yaml"),
                generated=[os.path.join(path, f) for f in GENERATED_CARTBONDED_FILES],
            ),
            genbonded=GenBondedDatabase.from_file(os.path.join(path, "genbonded.yaml")),
            disulfide=DisulfideDatabase.from_file(os.path.join(path, "disulfide.yaml")),
            metal_coordination=MetalCoordinationDatabase.from_file(
                os.path.join(path, "metal_coordination.yaml")
            ),
            na_torsion=NaTorsionDatabase.from_file(
                os.path.join(path, "na_torsion.yaml")
            ),
            dun=DunbrackRotamerLibrary.from_file(
                os.path.join(path, "dunbrack.bin"),
                lookup=os.path.join(path, "dunbrack.yaml"),
            ),
            elec=ElecDatabase.from_file(
                os.path.join(path, "elec.yaml"),
                generated=[os.path.join(path, f) for f in GENERATED_ELEC_FILES],
            ),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            omega_bbdep=OmegaBBDepDatabase.from_file(
                os.path.join(path, "omega_bbdep.zip")
            ),
            rama=RamaDatabase.from_file(os.path.join(path, "rama.zip")),
            ref=RefDatabase.from_file(
                os.path.join(path, "ref.yaml"),
                generated=[os.path.join(path, f) for f in GENERATED_REF_FILES],
            ),
        )
