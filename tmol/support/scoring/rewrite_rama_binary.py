import numpy
import torch
from pathlib import Path
import os
import argparse
import yaml
import attr
import cattr


from tmol.numeric.bspline import BSplineInterpolation
from tmol.database.scoring.rama import RamaDatabase, RamaTables

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
            prob_tables[curaa] = numpy.zeros((36, 36), dtype=float)
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


def create_rama_database(rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir):
    general, prepro = parse_all_tables(
        rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir
    )

    rama_tables = []
    for aa, prob in general.items():
        rama_tables.append(
            RamaTables(
                table_id=aa,
                table=prob,
                bbstep=[numpy.pi / 18.0, numpy.pi / 18.0],
                bbstart=[-numpy.pi, -numpy.pi],
            )
        )

    for aa, prob in prepro.items():
        rama_tables.append(
            RamaTables(
                table_id=aa + "_prepro",
                table=prob,
                bbstep=[numpy.pi / 18.0, numpy.pi / 18.0],
                bbstart=[-numpy.pi, -numpy.pi],
            )
        )

    path_lookup = "tmol/database/default/scoring/rama.yaml"

    with open(path_lookup, "r") as infile_lookup:
        raw = yaml.safe_load(infile_lookup)
        rama_lookup = cattr.structure(
            raw["rama_lookup"], attr.fields(RamaDatabase).rama_lookup.type
        )

    input_uniq_id = hash((rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir))

    uniq_id = path_lookup + "," + str(input_uniq_id)
    return RamaDatabase(uniq_id, rama_lookup, rama_tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosetta_dir", default=os.path.join(Path.home(), "Rosetta/main/")
    )
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--r3_rama_dir",
        default=os.path.join(
            args.rosetta_dir, "database/scoring/score_functions/rama/fd_beta_nov2016/"
        ),
    )
    parser.add_argument(
        "--r3_paapp_dir",
        default=os.path.join(
            args.rosetta_dir,
            "database/scoring/score_functions/P_AA_pp/shapovalov/10deg/kappa131/",
        ),
    )
    parser.add_argument(
        "--r3_paa_dir",
        default=os.path.join(
            args.rosetta_dir, "database/scoring/score_functions/P_AA_pp/"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../database/default/scoring/rama.zip",
        ),
    )
    args = parser.parse_args()

    torch.save(
        create_rama_database(
            0.5, args.r3_rama_dir, 0.61, args.r3_paapp_dir, args.r3_paa_dir
        ),
        args.output_path,
    )
