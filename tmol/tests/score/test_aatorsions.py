import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.torsions import AlphaAABackboneTorsionProvider


@reactive_attrs(auto_attribs=True)
class TCartTorsions(
    CartesianAtomicCoordinateProvider, AlphaAABackboneTorsionProvider, TorchDevice
):
    """Cart total."""

    pass


def test_create_torsion_provider(ubq_system):
    src = TCartTorsions.build_for(ubq_system)

    assert src

    gold_phi = torch.tensor(
        [
            numpy.nan,
            -91.0202,
            -131.0989,
            -115.9912,
            -118.0303,
            -95.2261,
            -99.5796,
            -73.4252,
            -101.3971,
            77.4447,
            -96.2711,
            -119.9447,
            -109.4737,
            -101.4413,
            -126.4132,
            -111.7952,
            -139.0196,
            -120.0428,
            -54.9367,
            -79.8384,
            -71.0060,
            -83.7038,
            -61.3275,
            -57.5667,
            -65.4883,
            -58.4233,
            -60.8123,
            -66.1219,
            -64.2164,
            -69.9996,
            -62.1262,
            -53.4121,
            -93.5800,
            -123.5788,
            81.2164,
            -79.7333,
            -57.0030,
            -57.1842,
            -68.2389,
            -95.7981,
            -84.8311,
            -121.2432,
            -103.5581,
            -122.0749,
            -144.2704,
            48.1713,
            61.7145,
            -115.0664,
            -85.7723,
            -79.5553,
            -101.8294,
            -48.1901,
            -82.9252,
            -85.4496,
            -104.5098,
            -61.2180,
            -63.8914,
            -55.5523,
            -91.0216,
            57.9396,
            -88.6669,
            -103.4373,
            -54.7883,
            66.8974,
            -71.1469,
            -119.1522,
            -103.1357,
            -105.5834,
            -106.9736,
            -108.1151,
            -96.0197,
            -117.6195,
            -83.7244,
            -97.6604,
            120.4148,
            174.1601,
        ]
    )
    numpy.testing.assert_allclose(
        (src.phi_tor * 180 / numpy.pi).detach().numpy().squeeze(), gold_phi, atol=1e-4
    )

    gold_psi = torch.tensor(
        [
            149.62877,
            138.2641,
            163.04655,
            140.22603,
            114.22436,
            127.541565,
            170.75417,
            -6.935815,
            14.92933,
            16.544413,
            138.07892,
            131.75684,
            141.98853,
            139.73755,
            154.00023,
            121.12877,
            170.7171,
            144.51154,
            -24.527037,
            -8.13681,
            148.4354,
            160.43427,
            -37.20999,
            -40.48768,
            -44.444294,
            -46.408836,
            -37.96477,
            -38.120052,
            -37.259007,
            -39.563976,
            -48.62783,
            -41.766552,
            -24.398628,
            -6.345671,
            5.3212533,
            124.86679,
            136.96703,
            -32.16529,
            -15.57162,
            -10.469159,
            129.69547,
            115.95642,
            130.24083,
            131.80696,
            129.63657,
            45.993965,
            21.61639,
            142.71301,
            130.28285,
            138.33296,
            139.9595,
            -42.210014,
            -8.888623,
            165.46437,
            164.59276,
            -36.20516,
            -29.622236,
            -39.302757,
            4.651424,
            45.355312,
            116.44079,
            169.54126,
            143.05714,
            19.147184,
            159.5173,
            126.67337,
            154.61172,
            135.67017,
            115.77554,
            139.93367,
            138.79993,
            98.81105,
            150.39395,
            93.879585,
            125.55793,
            numpy.nan,
        ]
    )

    gold_omega = torch.tensor(
        [
            178.30745,
            173.35893,
            179.56941,
            175.79433,
            -178.11598,
            179.94527,
            -177.98494,
            179.94191,
            179.79585,
            175.00797,
            176.16022,
            179.69595,
            178.15007,
            -179.26176,
            173.99286,
            -177.93565,
            174.7003,
            177.7299,
            -177.54395,
            -178.99536,
            175.73361,
            175.78523,
            176.99873,
            -179.64114,
            178.6179,
            -179.78072,
            176.85904,
            176.0259,
            179.91176,
            179.45953,
            179.13687,
            -175.65298,
            179.86525,
            176.50458,
            179.7132,
            178.30354,
            -179.28783,
            -177.44258,
            178.70926,
            -177.61922,
            172.09601,
            -177.50903,
            -177.55902,
            177.58876,
            177.38298,
            -179.52815,
            177.64848,
            174.56908,
            -175.91823,
            -177.70485,
            -175.83871,
            -177.23625,
            176.47205,
            177.11407,
            177.17581,
            -179.45142,
            179.75917,
            179.28006,
            179.91965,
            175.85329,
            -176.84166,
            -176.91852,
            179.4875,
            174.7964,
            175.93106,
            175.0767,
            177.97643,
            176.5758,
            175.07704,
            178.4371,
            177.24236,
            -179.31216,
            -179.6361,
            179.34215,
            179.2217,
            numpy.nan,
        ]
    )

    numpy.testing.assert_allclose(
        (src.psi_tor * 180 / numpy.pi).detach().numpy().squeeze(), gold_psi, atol=1e-4
    )

    numpy.testing.assert_allclose(
        (src.omega_tor * 180 / numpy.pi).detach().numpy().squeeze(),
        gold_omega,
        atol=1e-4,
    )

    # numpy.set_printoptions(threshold=numpy.nan)
    # print( "gold psi" )
    # print( (src.psi_tor * 180 / numpy.pi ).detach().numpy() )
    #
    # print( "gold omega" )
    # print( (src.omega_tor * 180 / numpy.pi ).detach().numpy() )


# def test_system_score_support_res_aas(ubq_system):
#     ubq_seq = [res.residue_type.name3 for res in ubq_system.residues]
#     src = TCartTorsions.build_for(ubq_system)
#     ind3 = AAIndex.canonical_laa_ind3()
#     numpy.testing.assert_array_equal(
#         src.res_aas.cpu().numpy().squeeze(), ind3.get_indexer(ubq_seq)
#     )