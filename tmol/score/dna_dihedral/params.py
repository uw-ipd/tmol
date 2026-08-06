import attr
import torch

from tmol.database.scoring.dna_dihedral import DnaDihedralDatabase
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs

# torsions this term reads, by chemical-layer name; order fixes tensor layout
BACKBONE_TORSIONS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
SUGAR_TORSIONS = ("delta", "nu4", "nu0", "nu1")
CHI_TORSION = "chi1"

TORSION_NAMES = BACKBONE_TORSIONS + ("nu4", "nu0", "nu1", CHI_TORSION)
TORSION_IND = {name: i for i, name in enumerate(TORSION_NAMES)}

BASES = ("a", "c", "g", "t")
BASE_FOR_NAME3 = {"DA": 0, "DC": 1, "DG": 2, "DT": 3}
N_PUCKER = 10

# pucker states on the C3'-endo side; the rest are south
NORTH_PUCKERS = (0, 1, 2, 3, 4)

# chi is scored against a fixed syn mean inside this window
SYN_MEAN = 50.0
SYN_RANGE = (20.0, 100.0)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DnaDihedralParams(ValidateAttrs):
    # means in degrees; 3 bins for alpha/gamma, 2 for beta/epsilon/zeta
    backbone_means: Tensor[torch.float32][6, 3]
    backbone_n_bins: Tensor[torch.int32][6]
    backbone_sdev: Tensor[torch.float32][6]

    sugar_means: Tensor[torch.float32][N_PUCKER, 4]  # delta nu4 nu0 nu1
    chi_means: Tensor[torch.float32][4, N_PUCKER]  # per base

    # bin-population energies; each bin assignment is charged once
    well_pucker: Tensor[torch.float32][N_PUCKER]
    well_alpha_gamma: Tensor[torch.float32][3, 3]
    well_bibii_pucker: Tensor[torch.float32][2, 2]
    well_alphanext_bibii: Tensor[torch.float32][3, 2]
    well_chi_syn: Tensor[torch.float32][2, N_PUCKER, 4]
    is_north: Tensor[torch.bool][N_PUCKER]

    sdev_sugar: float
    sdev_chi: float
    weight_bb: float
    weight_chi: float
    weight_sugar: float
    pucker_temperature: float
    bin_blend_sdev: float

    @classmethod
    def from_database(cls, database: DnaDihedralDatabase, device: torch.device):
        g = database.global_parameters

        means = torch.zeros((6, 3), dtype=torch.float32, device=device)
        n_bins = torch.zeros((6,), dtype=torch.int32, device=device)
        for i, name in enumerate(BACKBONE_TORSIONS):
            if name not in database.backbone_means:  # delta lives in the sugar term
                continue
            vals = database.backbone_means[name]
            means[i, : len(vals)] = torch.tensor(vals, device=device)
            n_bins[i] = len(vals)

        sugar = torch.zeros((N_PUCKER, 4), dtype=torch.float32, device=device)
        for j, name in enumerate(SUGAR_TORSIONS):
            sugar[:, j] = torch.tensor(database.sugar_means[name]["all"], device=device)

        chi = torch.zeros((4, N_PUCKER), dtype=torch.float32, device=device)
        for b, base in enumerate(BASES):
            chi[b] = torch.tensor(database.sugar_means["chi"][base], device=device)

        w = database.well_energies
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        chi_syn = torch.zeros((2, N_PUCKER, 4), dtype=torch.float32, device=device)
        for si, state in enumerate(("anti", "syn")):
            for b, base in enumerate(BASES):
                chi_syn[si, :, b] = t(w.chi_syn_given_pucker[state][base])
        is_north = torch.zeros((N_PUCKER,), dtype=torch.bool, device=device)
        is_north[list(NORTH_PUCKERS)] = True

        return cls(
            well_pucker=t(w.pucker),
            well_alpha_gamma=t(w.alpha_gamma),
            well_bibii_pucker=t(w.bibii_given_pucker),
            well_alphanext_bibii=t(w.alphanext_given_bibii),
            well_chi_syn=chi_syn,
            is_north=is_north,
            backbone_means=means,
            backbone_n_bins=n_bins,
            backbone_sdev=torch.tensor(
                g.sdev_backbone, dtype=torch.float32, device=device
            ),
            sugar_means=sugar,
            chi_means=chi,
            sdev_sugar=g.sdev_sugar,
            sdev_chi=g.sdev_chi,
            weight_bb=g.weight_bb,
            weight_chi=g.weight_chi,
            weight_sugar=g.weight_sugar,
            pucker_temperature=g.pucker_temperature,
            bin_blend_sdev=g.bin_blend_sdev,
        )
