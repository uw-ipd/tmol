import torch
from math import pi

from tmol.types.torch import Tensor
from tmol.types.functional import convert_args, validate_args

Coords = Tensor(torch.float)[..., 3]
Params = Tensor(torch.float)[...]
PolyParams = Tensor(torch.double)[...]


@convert_args
def polyval(
        A: Tensor(torch.double)[..., :],
        Arange: Tensor(torch.double)[..., 2],
        Abound: Tensor(torch.double)[..., 2],
        x: Tensor(torch.double)[...],
) -> Tensor(torch.float)[:]:
    """evaluate polynomial function (Horner's rule)"""
    p = A[:, 0]
    for i in range(1, A.shape[-1]):
        p = p * x + A[:, i]

    selector_low = (x < Arange[:, 0])
    p[selector_low] = Abound[selector_low, 0]
    selector_high = (x > Arange[:, 1])
    p[selector_high] = Abound[selector_high, 1]

    return p


@validate_args
def sp2chi_energy(
        d: float,
        m: float,
        l: float,
        BAH: Params,
        chi: Params,
):
    """Evaluate the functional form for sp2 acceptors that looks at the
    angle "BAH" between the acceptor-base (B), the acceptor (A), and the
    hydrogen (A), as well as the dihedral (chi) defined by the
    acceptor-base-base (B0), B, A, and H.

    This functional form was developed to prefer hydrogens lying in the
    sp2 plane, and at a BAH angle of 120 degrees -- it also avoids numerical
    stability issues for the chi dihedral when the BAH angle is near 180
    degrees.

    See O'Meara et al., JCTC 2015 for greater detail.
    """

    H = 0.5 * (torch.cos(2 * chi) + 1)

    F = torch.empty_like(chi)
    F.fill_(m - 0.5)
    G = torch.empty_like(chi)
    G.fill_(m - 0.5)

    selector_upper = (BAH >= pi * 2.0 / 3.0)
    selector_mid = ~selector_upper & (BAH >= pi * (2.0 / 3.0 - l))

    F[selector_upper
      ] = d / 2 * torch.cos(3 * (pi - BAH[selector_upper])) + d / 2 - 0.5
    G[selector_upper] = d - 0.5

    outer_rise = torch.cos(pi - (pi * 2 / 3 - BAH[selector_mid]) / l)
    F[selector_mid] = m / 2 * outer_rise + m / 2 - 0.5
    G[selector_mid] = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5

    E = H * F + (1 - H) * G

    return E


# energy evaluation to an sp2 acceptor
@validate_args
def hbond_donor_sp2_score(
        # Input coordinates
        d: Coords,
        h: Coords,
        a: Coords,
        b: Coords,
        b0: Coords,

        # type pair parameters
        glob_accwt: Params,
        glob_donwt: Params,
        AHdist_coeffs: PolyParams,
        AHdist_ranges: PolyParams,
        AHdist_bounds: PolyParams,
        cosBAH_coeffs: PolyParams,
        cosBAH_ranges: PolyParams,
        cosBAH_bounds: PolyParams,
        cosAHD_coeffs: PolyParams,
        cosAHD_ranges: PolyParams,
        cosAHD_bounds: PolyParams,

        # Global score parameters
        hb_sp2_range_span: float,
        hb_sp2_BAH180_rise: float,
        hb_sp2_outer_width: float,
):
    """Accepts a set of donor (d), hydrogen (h), acceptor (a), acceptor-base (b), and
    acceptor-base-base (b0) coordinates, all of the same length, and where the
    acceptor is sp2 hybridized, and the corresponding set of parameters, and for each
    entry i in these tensors, evaluates the full potential.

    Note that this function does not first eliminate hydrogen/acceptor pairs
    based on their distance, but rather, computes the full potential for all pairs.
    """

    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = pi - torch.acos(xD)
    # in non-cos space

    BAvecn = (a - b)
    BAvecn = BAvecn / BAvecn.norm(dim=-1).unsqueeze(dim=-1)
    xH = (AHvecn * BAvecn).sum(dim=-1)
    BAH = pi - torch.acos(xH)

    BB0vecn = (b0 - b)
    BB0vecn = BB0vecn / BB0vecn.norm(dim=-1).unsqueeze(dim=-1)
    xchi = (BB0vecn * AHvecn).sum(dim=-1) - \
           ((BB0vecn * BAvecn).sum(dim=-1) * (BAvecn * AHvecn).sum(dim=-1))

    ychi = (torch.cross(BAvecn, AHvecn, dim=-1) * BB0vecn).sum(dim=-1)
    chi = -torch.atan2(ychi, xchi)

    Pd = polyval(AHdist_coeffs, AHdist_ranges, AHdist_bounds, D)
    PxH = polyval(cosBAH_coeffs, cosBAH_ranges, cosBAH_bounds, xH)
    PxD = polyval(cosAHD_coeffs, cosAHD_ranges, cosAHD_bounds, AHD)

    # sp2 chi part
    Pchi = sp2chi_energy(
        hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width, BAH, chi
    )

    energy = acc_don_scale * (Pd + PxD + PxH + Pchi)

    # fade (squish [-0.1,0.1] to [-0.1,0.0])
    high_energy_selector = (energy > 0.1)
    med_energy_selector = ~high_energy_selector & (energy > -0.1)

    energy[high_energy_selector] = 0.0
    energy[med_energy_selector] = (
        -0.025 + 0.5 * energy[med_energy_selector] -
        2.5 * energy[med_energy_selector] * energy[med_energy_selector]
    )

    return energy


# energy evaluation to an sp3 acceptor
@validate_args
def hbond_donor_sp3_score(
        # Input coordinates
        d: Coords,
        h: Coords,
        a: Coords,
        b: Coords,
        b0: Coords,

        # type pair parameters
        glob_accwt: Params,
        glob_donwt: Params,
        AHdist_coeffs: PolyParams,
        AHdist_ranges: PolyParams,
        AHdist_bounds: PolyParams,
        cosBAH_coeffs: PolyParams,
        cosBAH_ranges: PolyParams,
        cosBAH_bounds: PolyParams,
        cosAHD_coeffs: PolyParams,
        cosAHD_ranges: PolyParams,
        cosAHD_bounds: PolyParams,

        # Global score parameters
        hb_sp3_softmax_fade: float,
):
    """Accepts a set of donor (d), hydrogen (h), acceptor (a), acceptor-base (b), and
    acceptor-base-base (b0) coordinates, all of the same length, and where the
    acceptor is sp3 hybridized, and the corresponding set of parameters, and for each
    entry i in these tensors, evaluates the full potential.

    Note that this function does not first eliminate hydrogen/acceptor pairs
    based on their distance, but rather, computes the full potential for all pairs.
    """

    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = pi - torch.acos(xD)
    # in non-cos space

    BAvecn = (a - b)
    BAvecn = BAvecn / BAvecn.norm(dim=-1).unsqueeze(dim=-1)
    xH1 = (AHvecn * BAvecn).sum(dim=-1)

    B0Avecn = (a - b0)
    B0Avecn = B0Avecn / B0Avecn.norm(dim=-1).unsqueeze(dim=-1)
    xH2 = (AHvecn * B0Avecn).sum(dim=-1)

    Pd = polyval(AHdist_coeffs, AHdist_ranges, AHdist_bounds, D)
    PxD = polyval(cosAHD_coeffs, cosAHD_ranges, cosAHD_bounds, AHD)
    PxH1 = polyval(cosBAH_coeffs, cosBAH_ranges, cosBAH_bounds, xH1)
    PxH2 = polyval(cosBAH_coeffs, cosBAH_ranges, cosBAH_bounds, xH2)
    PxH = 1.0 / hb_sp3_softmax_fade * torch.log(
        torch.exp(PxH1 * hb_sp3_softmax_fade) +
        torch.exp(PxH2 * hb_sp3_softmax_fade)
    )

    energy = acc_don_scale * (Pd + PxD + PxH)

    # fade (squish [-0.1,0.1] to [-0.1,0.0])
    high_energy_selector = (energy > 0.1)
    med_energy_selector = ~high_energy_selector & (energy > -0.1)

    energy[high_energy_selector] = 0.0
    energy[med_energy_selector] = (
        -0.025 + 0.5 * energy[med_energy_selector] -
        2.5 * energy[med_energy_selector] * energy[med_energy_selector]
    )

    return energy


# energy evaluation to a ring acceptor
@validate_args
def hbond_donor_ring_score(
        # Input coordinates
        d: Coords,
        h: Coords,
        a: Coords,
        b: Coords,
        b0: Coords,

        # type pair parameters
        glob_accwt: Params,
        glob_donwt: Params,
        AHdist_coeffs: PolyParams,
        AHdist_ranges: PolyParams,
        AHdist_bounds: PolyParams,
        cosBAH_coeffs: PolyParams,
        cosBAH_ranges: PolyParams,
        cosBAH_bounds: PolyParams,
        cosAHD_coeffs: PolyParams,
        cosAHD_ranges: PolyParams,
        cosAHD_bounds: PolyParams,
):
    """Accepts a set of donor (d), hydrogen (h), acceptor (a), acceptor-base (b), and
    acceptor-base-base (b0) coordinates, all of the same length, and where the
    acceptor is ring hybridized, and the corresponding set of parameters, and for each
    entry i in these tensors, evaluates the full potential.

    Note that this function does not first eliminate hydrogen/acceptor pairs
    based on their distance, but rather, computes the full potential for all pairs.
    """
    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = pi - torch.acos(xD)
    # in non-cos space

    BAvecn = (a - 0.5 * (b + b0))
    BAvecn = BAvecn / BAvecn.norm(dim=-1).unsqueeze(dim=-1)
    xH = (AHvecn * BAvecn).sum(dim=-1)

    Pd = polyval(AHdist_coeffs, AHdist_ranges, AHdist_bounds, D)
    PxD = polyval(cosAHD_coeffs, cosAHD_ranges, cosAHD_bounds, AHD)
    PxH = polyval(cosBAH_coeffs, cosBAH_ranges, cosBAH_bounds, xH)

    energy = acc_don_scale * (Pd + PxD + PxH)

    # fade (squish [-0.1,0.1] to [-0.1,0.0])
    high_energy_selector = (energy > 0.1)
    med_energy_selector = ~high_energy_selector & (energy > -0.1)

    energy[high_energy_selector] = 0.0
    energy[med_energy_selector] = (
        -0.025 + 0.5 * energy[med_energy_selector] -
        2.5 * energy[med_energy_selector] * energy[med_energy_selector]
    )

    return energy
