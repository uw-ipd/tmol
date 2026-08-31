import attr
import numpy
import torch

from tmol.database.scoring import NaTorsionDatabase
from tmol.types import (
    Tensor,
    ValidateAttrs,
)

# torsions this term reads, by chemical-layer name; order fixes tensor layout
BACKBONE_TORSIONS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
SUGAR_TORSIONS = ("delta", "nu4", "nu0", "nu1")
CHI_TORSION = "chi1"

TORSION_NAMES = BACKBONE_TORSIONS + ("nu4", "nu0", "nu1", CHI_TORSION)
TORSION_IND = {name: i for i, name in enumerate(TORSION_NAMES)}

# one index space for both polymers: DNA occupies the first block and RNA the
# second, so the polymer of a base is base // N_BASE_PER_POLYMER and the tables
# that vary only by polymer are gathered on that.
POLYMERS = ("dna", "rna")
POLYMER_IND = {name: i for i, name in enumerate(POLYMERS)}
# "x" is a nucleotide whose base is not one of the four: averaged over them, so
# a residue with an unrecognized or absent base is still placed on the polymer's
# distribution rather than left unscored
UNKNOWN_BASE = "x"
BASES = {"dna": ("a", "c", "g", "t", "x"), "rna": ("a", "c", "g", "u", "x")}
N_BASE_PER_POLYMER = 5
N_BASE = len(POLYMERS) * N_BASE_PER_POLYMER
BASE_FOR_NAME3 = {
    "DA": 0, "DC": 1, "DG": 2, "DT": 3, "DX": 4,
    "A": 5, "C": 6, "G": 7, "U": 8, "RX": 9,
}  # fmt: skip
# the reference a nucleotide falls back to when its base cannot be identified
UNKNOWN_BASE_REFERENCE = {"dna": "DX", "rna": "RX"}
N_PUCKER = 10

# pucker states on the C3'-endo side; the rest are south
NORTH_PUCKERS = (0, 1, 2, 3, 4)

# chi is scored against a fixed syn mean inside this window
SYN_MEAN = 50.0
SYN_RANGE = (20.0, 100.0)


N_TORSION = len(TORSION_NAMES)
DELTA = TORSION_IND["delta"]
CHI = TORSION_IND[CHI_TORSION]
# every torsion is optional and masked at scoring time: the backbone ones are
# absent at a terminus, chi wherever the base is not attached through a torsion
# we can name, and the sugar ones wherever the ring does not resolve
REQUIRED_TORSIONS: list = []


def sugar_ring_atoms(block_type, element_for_atom_type):
    """Ordered sugar ring, derived from the nu torsions rather than named.

    nu0 and nu1 each span four consecutive ring atoms offset by one, so
    together they give the whole cycle in order. The pucker slot arithmetic
    is defined relative to a cycle ending on the ring heteroatom, so rotate
    to put it last.
    """
    nu0 = block_type.torsion_to_uaids.get("nu0")
    nu1 = block_type.torsion_to_uaids.get("nu1")
    if nu0 is None or nu1 is None:
        return None
    cycle = [uaid[0] for uaid in nu0] + [nu1[-1][0]]
    if len(set(cycle)) != 5 or any(a < 0 for a in cycle):
        return None

    def element(atom_index):
        return element_for_atom_type[block_type.atoms[atom_index].atom_type]

    hetero = [i for i, a in enumerate(cycle) if element(a) != "C"]
    if len(hetero) != 1:
        return None
    k = hetero[0]
    return cycle[k + 1 :] + cycle[: k + 1]


def block_type_params(block_type, element_for_atom_type):
    """Per-block-type indices this term needs, shared by scoring and packing.

    base is -1 for anything this term does not handle. A nucleotide whose sugar
    ring or glycosidic torsion cannot be resolved still scores: those subterms
    are masked, and what remains of its backbone is scored as usual.
    """
    # a modified nucleotide may borrow a canonical base table
    base = BASE_FOR_NAME3.get(block_type.na_base_reference or block_type.name3, -1)
    uaids = numpy.full((N_TORSION, 4, 3), -1, dtype=numpy.int32)
    ring = numpy.full((5,), -1, dtype=numpy.int32)
    if base >= 0:
        for i, name in enumerate(TORSION_NAMES):
            tor = block_type.torsion_to_uaids.get(name)
            if tor is not None:
                uaids[i] = numpy.array(tor, dtype=numpy.int32)
        ring_atoms = sugar_ring_atoms(block_type, element_for_atom_type)
        if ring_atoms is not None:
            ring[:] = ring_atoms
        if (uaids[REQUIRED_TORSIONS, :, 0] < 0).any():
            base = -1
    down = block_type.connection_to_cidx.get("down", -1)
    return dict(base=base, uaids=uaids, ring=ring, down=down)


def _circular_mean(degrees):
    """Mean of angles, over the leading axis, in degrees.

    Averaging angles directly puts the mean of 179 and -179 at zero, which is
    the opposite side of the circle from either.
    """
    radians = degrees * (torch.pi / 180.0)
    mean = torch.atan2(radians.sin().mean(0), radians.cos().mean(0))
    # the database writes its means in [0, 360); wrap_degrees does not care,
    #    but a row in a different range invites a wrong comparison later
    return (mean * (180.0 / torch.pi)) % 360.0


def polymer_index(base):
    """0 for DNA, 1 for RNA. Non-nucleotides (base < 0) fall to 0 and are masked."""
    return base.clamp_min(0) // N_BASE_PER_POLYMER


@attr.s(auto_attribs=True, frozen=True, slots=True)
class NaTorsionParams(ValidateAttrs):
    # means in degrees; 3 bins for alpha/gamma, 2 for beta/epsilon/zeta
    backbone_means: Tensor[torch.float32][2, 6, 3]
    backbone_n_bins: Tensor[torch.int32][2, 6]
    backbone_sdev: Tensor[torch.float32][2, 6]

    sugar_means: Tensor[torch.float32][2, N_PUCKER, 4]  # delta nu4 nu0 nu1
    chi_means: Tensor[torch.float32][N_BASE, N_PUCKER]

    # bin-population energies; each bin assignment is charged once
    well_pucker: Tensor[torch.float32][2, N_PUCKER]
    well_alpha_gamma: Tensor[torch.float32][2, 3, 3]
    well_bibii_pucker: Tensor[torch.float32][2, 2, 2]
    well_alphanext_bibii: Tensor[torch.float32][2, 3, 2]
    well_chi_syn: Tensor[torch.float32][2, N_PUCKER, N_BASE]
    is_north: Tensor[torch.bool][N_PUCKER]

    # per-polymer scalars, gathered like the tables above
    sdev_sugar: Tensor[torch.float32][2]
    sdev_chi: Tensor[torch.float32][2]
    weight_bb: Tensor[torch.float32][2]
    weight_chi: Tensor[torch.float32][2]
    weight_sugar: Tensor[torch.float32][2]

    # shared: these set the shape of the soft bin assignment, not its content
    pucker_temperature: float
    bin_blend_sdev: float

    @classmethod
    def from_database(
        cls, database: NaTorsionDatabase, device: torch.device
    ):  # noqa: C901
        def t(v, dtype=torch.float32):
            return torch.tensor(v, dtype=dtype, device=device)

        def stack(build):
            return torch.stack([build(p) for p in POLYMERS])

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)

        def backbone(poly):
            means = zeros(6, 3)
            for i, name in enumerate(BACKBONE_TORSIONS):
                vals = database.backbone_means[poly].get(name)
                if vals is None:  # delta lives in the sugar term
                    continue
                means[i, : len(vals)] = t(vals)
            return means

        def backbone_bins(poly):
            n = zeros(6, dtype=torch.int32)
            for i, name in enumerate(BACKBONE_TORSIONS):
                vals = database.backbone_means[poly].get(name)
                n[i] = 0 if vals is None else len(vals)
            return n

        def sugar(poly):
            out = zeros(N_PUCKER, 4)
            for j, name in enumerate(SUGAR_TORSIONS):
                out[:, j] = t(database.sugar_means[poly][name]["all"])
            return out

        chi = zeros(N_BASE, N_PUCKER)
        chi_syn = zeros(2, N_PUCKER, N_BASE)
        for pi, poly in enumerate(POLYMERS):
            w = database.well_energies[poly]
            named = [b for b in BASES[poly] if b != UNKNOWN_BASE]
            for b, base in enumerate(BASES[poly]):
                ind = pi * N_BASE_PER_POLYMER + b
                if base == UNKNOWN_BASE:
                    # chi means are angles, so they average on the circle; the
                    #    well energies are energies and average directly
                    chi[ind] = _circular_mean(
                        torch.stack(
                            [t(database.sugar_means[poly]["chi"][n]) for n in named]
                        )
                    )
                    for si, state in enumerate(("anti", "syn")):
                        chi_syn[si, :, ind] = torch.stack(
                            [t(w.chi_syn_given_pucker[state][n]) for n in named]
                        ).mean(0)
                    continue
                chi[ind] = t(database.sugar_means[poly]["chi"][base])
                for si, state in enumerate(("anti", "syn")):
                    chi_syn[si, :, ind] = t(w.chi_syn_given_pucker[state][base])

        is_north = zeros(N_PUCKER, dtype=torch.bool)
        is_north[list(NORTH_PUCKERS)] = True

        def well(field):
            return stack(lambda p: t(getattr(database.well_energies[p], field)))

        def scalar(field):
            return t([getattr(database.global_parameters[p], field) for p in POLYMERS])

        # shared across polymers by construction; the generator writes one value
        shared = database.global_parameters[POLYMERS[0]]

        return cls(
            backbone_means=stack(backbone),
            backbone_n_bins=stack(backbone_bins),
            backbone_sdev=stack(
                lambda p: t(database.global_parameters[p].sdev_backbone)
            ),
            sugar_means=stack(sugar),
            chi_means=chi,
            well_pucker=well("pucker"),
            well_alpha_gamma=well("alpha_gamma"),
            well_bibii_pucker=well("bibii_given_pucker"),
            well_alphanext_bibii=well("alphanext_given_bibii"),
            well_chi_syn=chi_syn,
            is_north=is_north,
            sdev_sugar=scalar("sdev_sugar"),
            sdev_chi=scalar("sdev_chi"),
            weight_bb=scalar("weight_bb"),
            weight_chi=scalar("weight_chi"),
            weight_sugar=scalar("weight_sugar"),
            pucker_temperature=shared.pucker_temperature,
            bin_blend_sdev=shared.bin_blend_sdev,
        )
