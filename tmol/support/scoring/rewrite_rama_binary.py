import numpy
import torch
import zarr
from pathlib import Path
import os
import sys

from tmol.numeric.bspline import BSplineInterpolation

# A conversion script from rosetta BB probability tables to
# tmol probability tables.  tmol stores tables which contain
# the weighted sum of p_aa_pp and rama energies.  This requires
# several preprocessing steps:
#  1) converting rama probs to E_rama
#  2) converting p_aa_pp _and_ p_aa probs to E_paapp
#  3) resampling p_aa_pp (gridded at 5,15,25...) to the
#     rama grid (0,10,...)
#  4) combining the weighted sum of both terms using the
#     weights in beta16 (rama 0.5, paapp 0.61)
# Additionally, the R3 ad hoc low-probability corrections
# are replaced by a simple M estimate, where exp(-20) is added
# to all table probabilities.  This values may be modulated
# through the variable 'eps' in this script.
#
# Note that this may lead to large changes in energies in
# low-probability reagions of ramachandran space
#
# Tables are then written as zarr ZipStores.  rama energies
# in tmol simply involve interpolating these tables.


def parse_paa(lines):
    prob_tables = {}
    for line in lines:
        cols = line.split()
        if cols[0][0] == "#":
            continue
        prob_tables[cols[0]] = float(cols[1])
    return prob_tables


def parse_lines_as_ndarrays(lines, aacol=0, phipsicol=1, probcol=3):
    prob_tables = {}
    for line in lines:
        cols = line.split()
        if cols[0][0] == "#":
            continue
        curaa = cols[aacol]
        if curaa not in prob_tables.keys():
            prob_tables[curaa] = numpy.zeros((36, 36), dtype=numpy.float)
        phi_ind, psi_ind = (
            int(float(x)) // 10 + 18 for x in cols[phipsicol : (phipsicol + 2)]
        )
        prob_tables[curaa][phi_ind, psi_ind] = float(cols[probcol])
    return prob_tables


def parse_all_tables(rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir):
    # load tables
    general = parse_lines_as_ndarrays(open(r3_rama_dir + "all.ramaProb").readlines())
    prepro = parse_lines_as_ndarrays(open(r3_rama_dir + "prepro.ramaProb").readlines())
    paapp = parse_lines_as_ndarrays(open(r3_paapp_dir + "a20.prop").readlines(), 2, 0)
    paa = parse_paa(open(r3_paa_dir + "P_AA").readlines())

    for aa, prob in paapp.items():
        # convert p_aa_pp to energies
        # (do not normalize, normalization is across aas)
        energies = -numpy.log(prob / paa[aa])

        # resample, shifting -5 degrees
        espline = BSplineInterpolation.from_coordinates(
            torch.tensor(energies, dtype=torch.float)
        )
        x = torch.arange(36, dtype=torch.float) - 0.5
        xx, yy = torch.meshgrid((x, x))
        Xs = torch.stack((xx, yy)).reshape(2, -1).transpose(0, 1).contiguous()
        energies = espline.interpolate(Xs).reshape(36, 36).numpy()

        paapp[aa] = energies

    for aa, prob in prepro.items():
        # convert rama to energies
        # NOTE: this normalization is not properly done in R3 for prepro
        prob /= numpy.sum(prob)
        entropy = numpy.sum(prob * numpy.log(prob))
        energies = -numpy.log(prob) + entropy

        # reweight, add in paapp
        prepro[aa] = rama_wt * energies + paapp_wt * paapp[aa]

    for aa, prob in general.items():
        # convert rama to energies
        prob /= numpy.sum(prob)
        entropy = numpy.sum(prob * numpy.log(prob))
        energies = -numpy.log(prob) + entropy

        # reweight, add in paapp
        general[aa] = rama_wt * energies + paapp_wt * paapp[aa]

    # numpy.savetxt("ala.csv", general['ALA'], delimiter=",")
    return (general, prepro)


def zarr_from_db(rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir, output_path):
    """
    Write the Ramachandran binary file after reading Rosetta3's
    rama and p_aa_pp, and combining the weighted sum
    """
    with zarr.ZipStore(output_path + "/rama.zip", mode="w") as store:
        general, prepro = parse_all_tables(
            rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir
        )

        # write tables
        zgroup = zarr.group(store=store)
        for aa, prob in general.items():
            group_aa = zgroup.create_group(aa)
            data_aa = group_aa.create_dataset("prob", data=prob)
            data_aa.attrs["bbstep"] = [numpy.pi / 18.0, numpy.pi / 18.0]
            data_aa.attrs["bbstart"] = [-numpy.pi, -numpy.pi]

        for aa, prob in prepro.items():
            group_aa = zgroup.create_group(aa + "_prepro")
            data_aa = group_aa.create_dataset("prob", data=prob)
            data_aa.attrs["bbstep"] = [numpy.pi / 18.0, numpy.pi / 18.0]
            data_aa.attrs["bbstart"] = [-numpy.pi, -numpy.pi]


if __name__ == "__main__":
    r3_rama_dir = (
        str(Path.home())
        + "/Rosetta/main/database/scoring/score_functions/rama/fd_beta_nov2016/"
    )
    r3_paapp_dir = (
        str(Path.home())
        + "/Rosetta/main/database/scoring/"
        + "score_functions/P_AA_pp/shapovalov/10deg/kappa131/"
    )
    r3_paa_dir = (
        str(Path.home()) + "/Rosetta/main/database/scoring/score_functions/P_AA_pp/"
    )
    output_path = (
        str(os.path.dirname(os.path.realpath(__file__)))
        + "/../../database/default/scoring/"
    )
    if len(sys.argv) > 1:
        r3_rama_dir = sys.argv[1]
    if len(sys.argv) > 2:
        r3_paapp_dir = sys.argv[2]
    if len(sys.argv) > 3:
        r3_paapp_dir = sys.argv[2]
    if len(sys.argv) > 4:
        output_path = sys.argv[3]

    zarr_from_db(0.5, r3_rama_dir, 0.61, r3_paapp_dir, r3_paa_dir, output_path)
