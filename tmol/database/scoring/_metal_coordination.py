from typing import Optional, Tuple

import attr
import cattr

from tmol.database._yaml import safe_load


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MetalCoordinationGlobalParameters:
    radial_sd: float
    lateral_sd: float
    fan_sd: float
    # width, in degrees, of the wall below a bridge floor
    bridge_width: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MetalIonWidths:
    atom_type: str
    radial_sd: Optional[float] = None
    lateral_sd: Optional[float] = None


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MetalWellDepth:
    atom_type: str
    donor: str
    depth: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MetalBridgeFloor:
    donor: str
    # smallest metal-donor-metal angle, degrees
    floor: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class MetalCoordinationDatabase:
    global_parameters: MetalCoordinationGlobalParameters
    ions: Tuple[MetalIonWidths, ...] = ()
    well_depths: Tuple[MetalWellDepth, ...] = ()
    bridge_floors: Tuple[MetalBridgeFloor, ...] = ()

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = safe_load(infile)
        return cattr.structure(raw, cls)

    def widths(self, atom_type: str) -> Tuple[float, float]:
        """Radial and lateral sd for a metal atom type."""
        radial = self.global_parameters.radial_sd
        lateral = self.global_parameters.lateral_sd
        for ion in self.ions:
            if ion.atom_type == atom_type:
                radial = ion.radial_sd if ion.radial_sd is not None else radial
                lateral = ion.lateral_sd if ion.lateral_sd is not None else lateral
        return radial, lateral

    def bridge_floor(self, donor: str) -> Optional[float]:
        """Smallest metal-donor-metal angle, degrees, for a donor key; None if unset."""
        for row in self.bridge_floors:
            if row.donor == donor:
                return row.floor
        return None

    def well_depth(self, atom_type: str, donor: str) -> float:
        for well in self.well_depths:
            if well.atom_type == atom_type and well.donor == donor:
                return well.depth
        return 0.0
