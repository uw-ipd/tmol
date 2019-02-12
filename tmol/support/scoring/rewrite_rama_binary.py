import numpy
import zarr
from pathlib import Path
import os
import sys


def parse_lines_as_ndarrays(lines):
    curr_aa = None
    prob_tables = {}
    for line in lines:
        cols = line.split()
        if cols[0] != curr_aa:
            curr_aa = cols[0]
            prob_tables[curr_aa] = numpy.zeros((36, 36), dtype=numpy.float)
        phi_ind, psi_ind = (int(x) // 10 + 18 for x in cols[1:3])
        prob_tables[curr_aa][phi_ind, psi_ind] = float(cols[3])
    return prob_tables


def zarr_from_db(r3_rama_dir, output_path):
    """
    Write the Ramachandran binary file from the all.ramaProb and
    prepro.ramaProb files that should be found in the
    r3_rama_path directory.
    """

    with zarr.ZipStore(output_path + "/rama.zip", mode="w") as store:
        zgroup = zarr.group(store=store)
        general_case = parse_lines_as_ndarrays(
            open(r3_rama_dir + "all.ramaProb").readlines()
        )
        print(general_case)
        for aa, prob in general_case.items():
            group_aa = zgroup.create_group(aa)
            data_aa = group_aa.create_dataset("prob", data=prob)
            data_aa.attrs["bbstep"] = [10, 10]
            data_aa.attrs["bbstart"] = [-180, -180]
        prepro_case = parse_lines_as_ndarrays(
            open(r3_rama_dir + "prepro.ramaProb").readlines()
        )
        for aa, prob in prepro_case.items():
            group_aa = zgroup.create_group(aa + "_prepro")
            data_aa = group_aa.create_dataset("prob", data=prob)
            data_aa.attrs["bbstep"] = [10, 10]
            data_aa.attrs["bbstart"] = [-180, -180]


if __name__ == "__main__":
    r3_rama_dir = (
        str(Path.home())
        + "/Rosetta/main/database/scoring/score_functions/rama/fd_beta_nov2016/"
    )
    output_path = (
        str(os.path.dirname(os.path.realpath(__file__)))
        + "/../../database/default/scoring/"
    )
    print(r3_rama_dir)
    print(output_path)
    if len(sys.argv) > 1:
        r3_rama_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    zarr_from_db(r3_rama_dir, output_path)
