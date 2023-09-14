import os
import attr

from .cartbonded import CartBondedDatabase
from .disulfide import DisulfideDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary
from .elec import ElecDatabase
from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .omega import OmegaDatabase  # once scoring refactoring is done this can be removed
from .rama import RamaDatabase
from .ref import RefDatabase
from .omega_bbdep import OmegaBBDepDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    cartbonded: CartBondedDatabase
    disulfide: DisulfideDatabase
    dun: DunbrackRotamerLibrary
    elec: ElecDatabase
    hbond: HBondDatabase
    ljlk: LJLKDatabase
    omega: OmegaDatabase
    omega_bbdep: OmegaBBDepDatabase
    rama: RamaDatabase
    ref: RefDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):  # noqa
        return cls(
            cartbonded=CartBondedDatabase.from_file(
                os.path.join(path, "cartbonded.yaml")
            ),
            disulfide=DisulfideDatabase.from_file(os.path.join(path, "disulfide.yaml")),
            dun=DunbrackRotamerLibrary.from_zarr_archive(
                os.path.join(path, "dunbrack.yaml"), os.path.join(path, "dunbrack.bin")
            ),
            elec=ElecDatabase.from_file(os.path.join(path, "elec.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            omega=OmegaDatabase.from_file(os.path.join(path, "omega.yaml")),
            omega_bbdep=OmegaBBDepDatabase.from_files(
                os.path.join(path, "omega_bbdep.yaml"),
                os.path.join(path, "omega_bbdep.zip"),
            ),
            rama=RamaDatabase.from_files(
                os.path.join(path, "rama.yaml"), os.path.join(path, "rama.zip")
            ),
            ref=RefDatabase.from_file(os.path.join(path, "ref.yaml")),
        )
