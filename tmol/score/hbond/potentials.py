import torch
import numpy


# evaluate polynomial function (Horner's rule)
def polyval(A, Arange, Abound, x):
    p = A[:, 0]
    for i in range(1, A.shape[-1]):
        p = p * x + A[:, i]

    selector_low = (x < Arange[:, 0])
    p[selector_low] = Abound[selector_low, 0]
    selector_high = (x > Arange[:, 1])
    p[selector_high] = Abound[selector_high, 1]

    return p


# evaluate sp2chi
def sp2chi_energy(d, m, l, BAH, chi):
    H = 0.5 * (torch.cos(2 * chi) + 1)

    F = torch.empty_like(chi)
    F.fill_(m - 0.5)
    G = torch.empty_like(chi)
    G.fill_(m - 0.5)

    selector_upper = (BAH >= numpy.pi * 2.0 / 3.0)
    selector_mid = ~selector_upper & (BAH >= numpy.pi * (2.0 / 3.0 - l))

    F[selector_upper
      ] = d / 2 * torch.cos(3 * (numpy.pi - BAH[selector_upper])) + d / 2 - 0.5
    G[selector_upper] = d - 0.5

    outer_rise = torch.cos(
        numpy.pi - (numpy.pi * 2 / 3 - BAH[selector_mid]) / l
    )
    F[selector_mid] = m / 2 * outer_rise + m / 2 - 0.5
    G[selector_mid] = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5

    E = H * F + (1 - H) * G

    return E


# energy evaluation to an sp2 acceptor
def hbond_donor_sp2_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        b0,

        # type pair parameters
        glob_accwt,
        glob_donwt,
        AHdist_coeffs,
        AHdist_ranges,
        AHdist_bounds,
        cosBAH_coeffs,
        cosBAH_ranges,
        cosBAH_bounds,
        cosAHD_coeffs,
        cosAHD_ranges,
        cosAHD_bounds,

        # Global score parameters
        hb_sp2_range_span,
        hb_sp2_BAH180_rise,
        hb_sp2_outer_width,
        max_dis
):
    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = numpy.pi - torch.acos(xD)
    # in non-cos space

    BAvecn = (a - b)
    BAvecn = BAvecn / BAvecn.norm(dim=-1).unsqueeze(dim=-1)
    xH = (AHvecn * BAvecn).sum(dim=-1)
    BAH = numpy.pi - torch.acos(xH)

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
def hbond_donor_sp3_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        b0,

        # type pair parameters
        glob_accwt,
        glob_donwt,
        AHdist_coeffs,
        AHdist_ranges,
        AHdist_bounds,
        cosBAH_coeffs,
        cosBAH_ranges,
        cosBAH_bounds,
        cosAHD_coeffs,
        cosAHD_ranges,
        cosAHD_bounds,

        # Global score parameters
        hb_sp3_softmax_fade,
        max_dis
):
    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = numpy.pi - torch.acos(xD)
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
def hbond_donor_ring_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        bp,

        # type pair parameters
        glob_accwt,
        glob_donwt,
        AHdist_coeffs,
        AHdist_ranges,
        AHdist_bounds,
        cosBAH_coeffs,
        cosBAH_ranges,
        cosBAH_bounds,
        cosAHD_coeffs,
        cosAHD_ranges,
        cosAHD_bounds,

        # Global score parameters
        max_dis
):
    acc_don_scale = glob_accwt * glob_donwt

    # Using R3 nomenclature... xD = cos(180-AHD); xH = cos(180-BAH)
    D = (a - h).norm(dim=-1)

    AHvecn = (h - a)
    AHvecn = AHvecn / AHvecn.norm(dim=-1).unsqueeze(dim=-1)
    HDvecn = (d - h)
    HDvecn = HDvecn / HDvecn.norm(dim=-1).unsqueeze(dim=-1)
    xD = (AHvecn * HDvecn).sum(dim=-1)
    AHD = numpy.pi - torch.acos(xD)
    # in non-cos space

    BAvecn = (a - 0.5 * (b + bp))
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
