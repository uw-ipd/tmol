import numpy
import zarr
from pathlib import Path
import os
import sys

# A conversion script from rosetta omega bbdep tables to
# tmol probability tables.  For each of five classes
# of AA (general, gly, val+ile, pro, and prepro) it computes
#
# Tables are then written as zarr ZipStores.


def parse_lines_as_ndarrays(lines, mucol=4, sigmacol=5):
    mu = numpy.zeros((36, 36), dtype=float)
    sigma = numpy.zeros((36, 36), dtype=float)
    for line in lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        x, y = int(cols[0]), int(cols[1])

        mu[x, y] = float(cols[mucol])
        sigma[x, y] = float(cols[sigmacol])
    return mu, sigma


def parse_all_tables(r3_bbdepomega_dir):
    # load tables
    tables = {}
    for tid in ["all", "gly", "prepro", "pro", "valile"]:
        tables[tid] = parse_lines_as_ndarrays(
            open(r3_bbdepomega_dir + "omega_ppdep." + tid + ".txt").readlines()
        )

    return tables


def zarr_from_db(r3_bbdepomega_dir, output_path):
    """
    Write the BBDep omega binary file
    """
    tables = parse_all_tables(r3_bbdepomega_dir)

    with zarr.ZipStore(output_path + "/omega_bbdep.zip", mode="w") as store:
        # write tables
        zgroup = zarr.group(store=store)

        for aa, (mu, sig) in tables.items():
            group_aa = zgroup.create_group(aa)
            mu_aa = group_aa.create_dataset("mu", data=mu)
            mu_aa.attrs["bbstep"] = [numpy.pi / 18.0, numpy.pi / 18.0]
            mu_aa.attrs["bbstart"] = [numpy.pi / 36.0, numpy.pi / 36.0]
            sigma_aa = group_aa.create_dataset("sigma", data=sig)
            sigma_aa.attrs["bbstep"] = [numpy.pi / 18.0, numpy.pi / 18.0]
            sigma_aa.attrs["bbstart"] = [numpy.pi / 36.0, numpy.pi / 36.0]


if __name__ == "__main__":
    r3_bbdepomega_dir = (
        str(Path.home()) + "/Rosetta/main/database/scoring/score_functions/omega/"
    )
    output_path = (
        str(os.path.dirname(os.path.realpath(__file__)))
        + "/../../database/default/scoring/"
    )
    if len(sys.argv) > 1:
        r3_bbdepomega_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    zarr_from_db(r3_bbdepomega_dir, output_path)
