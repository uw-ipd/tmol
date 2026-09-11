import os

import attr
import cattr
from tmol.database._yaml import safe_load
import json
import hashlib

from typing import Tuple


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LengthGroup:
    atm1: str
    atm2: str
    x0: float
    K: float
    type: int = 0


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AngleGroup:
    atm1: str
    atm2: str
    atm3: str
    x0: float
    K: float
    type: int = 1


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TorsionGroup:
    atm1: str
    atm2: str
    atm3: str
    atm4: str
    k1: float = 0.0
    phi1: float = 0.0
    k2: float = 0.0
    phi2: float = 0.0
    k3: float = 0.0
    phi3: float = 0.0
    type: int = 2


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ImproperGroup(TorsionGroup):
    type: int = 3


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HxlTorsionGroup(TorsionGroup):
    type: int = 4


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartRes:
    length_parameters: Tuple[LengthGroup, ...]
    angle_parameters: Tuple[AngleGroup, ...]
    torsion_parameters: Tuple[TorsionGroup, ...]
    improper_parameters: Tuple[ImproperGroup, ...]
    hxltorsion_parameters: Tuple[HxlTorsionGroup, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ConnectionCartRes:
    """Complete length/angle parameters for one pair of named connections.

    Block names identify exact patched residue types. Atom names prefixed by
    '+' belong to block_type2; other names belong to block_type1. The record
    replaces legacy length/angle lookup for this pair, in either orientation.
    It must cover the bond and every two-block angle through it exactly once. Proper
    and improper torsions retain their existing independent ownership.
    """

    block_type1: str
    connection1: str
    block_type2: str
    connection2: str
    length_parameters: Tuple[LengthGroup, ...]
    angle_parameters: Tuple[AngleGroup, ...]
    provenance: str = ""


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartBondedDatabase:
    residue_params: dict[str, CartRes]
    hash: str
    connection_params: Tuple[ConnectionCartRes, ...] = ()

    @classmethod
    def from_file(cls, path, generated=()):
        with open(path, "r") as infile:
            resparam_dict = safe_load(infile)
        for extra in generated:
            if not os.path.exists(extra):
                continue
            with open(extra, "r") as infile:
                extension = safe_load(infile)
                resparam_dict["residue_params"].update(
                    extension.get("residue_params", {})
                )
                if extension.get("connection_params"):
                    resparam_dict.setdefault("connection_params", []).extend(
                        extension["connection_params"]
                    )
        # A serialized cache key is derived data, not part of its own input.
        resparam_dict.pop("hash", None)
        resparam_dict["hash"] = cls._generate_hash(resparam_dict)

        return cattr.structure(resparam_dict, cls)

    @classmethod
    def from_cartres_dict(cls, cartres_dict: dict[str, CartRes], connection_params=()):
        resparam_dict = cattr.unstructure(cartres_dict)
        if connection_params:
            resparam_dict = {
                "residue_params": resparam_dict,
                "connection_params": cattr.unstructure(connection_params),
            }
        hash = cls._generate_hash(resparam_dict)
        return cls(
            residue_params=cartres_dict,
            hash=hash,
            connection_params=tuple(connection_params),
        )

    @classmethod
    def _generate_hash(cls, resparam_dict):
        serialized = json.dumps(resparam_dict, sort_keys=True)
        hash_value = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return hash_value
