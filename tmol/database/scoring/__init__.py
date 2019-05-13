import os
import attr

from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .elec import ElecDatabase
from .rama import RamaDatabase
from .cartbonded import CartBondedDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:

    ljlk: LJLKDatabase
    elec: ElecDatabase
    hbond: HBondDatabase
    rama: RamaDatabase
    cartbonded: CartBondedDatabase
    dun: DunbrackRotamerLibrary

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):  # noqa
        return cls(
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            elec=ElecDatabase.from_file(os.path.join(path, "elec.yaml")),
            rama=RamaDatabase.from_files(
                os.path.join(path, "rama.yaml"), os.path.join(path, "rama.zip")
            ),
            cartbonded=CartBondedDatabase.from_file(
                os.path.join(path, "cartbonded.yaml")
            ),
            dun=DunbrackRotamerLibrary.from_zarr_archive(
                os.path.join(path, "dunbrack.yaml"), os.path.join(path, "dunbrack.bin")
            ),
        )
