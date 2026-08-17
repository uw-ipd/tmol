import os
import attr

from .cartbonded import CartBondedDatabase
from .genbonded import GenBondedDatabase
from .disulfide import DisulfideDatabase
from .na_torsion import NaTorsionDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary
from .elec import ElecDatabase
from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .omega_bbdep import OmegaBBDepDatabase
from .rama import RamaDatabase
from .ref import RefDatabase

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "LengthGroup": ("cartbonded", "LengthGroup"),
    "AngleGroup": ("cartbonded", "AngleGroup"),
    "TorsionGroup": ("cartbonded", "TorsionGroup"),
    "ImproperGroup": ("cartbonded", "ImproperGroup"),
    "HxlTorsionGroup": ("cartbonded", "HxlTorsionGroup"),
    "CartRes": ("cartbonded", "CartRes"),
    "DisulfideGlobalParameters": ("disulfide", "DisulfideGlobalParameters"),
    "RotamericDataForAA": ("dunbrack_libraries", "RotamericDataForAA"),
    "RotamericAADunbrackLibrary": ("dunbrack_libraries", "RotamericAADunbrackLibrary"),
    "SemiRotamericAADunbrackLibrary": (
        "dunbrack_libraries",
        "SemiRotamericAADunbrackLibrary",
    ),
    "DunMappingParams": ("dunbrack_libraries", "DunMappingParams"),
    "GlobalParams": ("elec", "GlobalParams"),
    "CountPairReps": ("elec", "CountPairReps"),
    "PartialCharges": ("elec", "PartialCharges"),
    "GenBondedTorsionEntry": ("genbonded", "GenBondedTorsionEntry"),
    "GenBondedImproperEntry": ("genbonded", "GenBondedImproperEntry"),
    "DonorAtomType": ("hbond", "DonorAtomType"),
    "AcceptorAtomType": ("hbond", "AcceptorAtomType"),
    "DonorTypeParam": ("hbond", "DonorTypeParam"),
    "AcceptorTypeParam": ("hbond", "AcceptorTypeParam"),
    "PolynomialParameters": ("hbond", "PolynomialParameters"),
    "PairParameters": ("hbond", "PairParameters"),
    "HBondDatabaseRaw": ("hbond", "HBondDatabaseRaw"),
    "LJLKGlobalParameters": ("ljlk", "LJLKGlobalParameters"),
    "LJLKAtomTypeParameters": ("ljlk", "LJLKAtomTypeParameters"),
    "NaTorsionGlobalParams": ("na_torsion", "NaTorsionGlobalParams"),
    "NaTorsionWells": ("na_torsion", "NaTorsionWells"),
    "OmegaBBDepMappingParams": ("omega_bbdep", "OmegaBBDepMappingParams"),
    "OmegaBBDepTables": ("omega_bbdep", "OmegaBBDepTables"),
    "RamaMappingParams": ("rama", "RamaMappingParams"),
    "RamaTables": ("rama", "RamaTables"),
    "CartBondedDatabase": ("cartbonded", "CartBondedDatabase"),
    "DisulfideDatabase": ("disulfide", "DisulfideDatabase"),
    "DunbrackRotamerLibrary": ("dunbrack_libraries", "DunbrackRotamerLibrary"),
    "ElecDatabase": ("elec", "ElecDatabase"),
    "HBondDatabase": ("hbond", "HBondDatabase"),
    "LJLKDatabase": ("ljlk", "LJLKDatabase"),
    "NaTorsionDatabase": ("na_torsion", "NaTorsionDatabase"),
    "OmegaBBDepDatabase": ("omega_bbdep", "OmegaBBDepDatabase"),
    "RamaDatabase": ("rama", "RamaDatabase"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
