import numpy
from pathlib import Path
import os
import sys
import argparse
import torch
import yaml
import attr
import cattr

from tmol.database.scoring.omega_bbdep import (
    OmegaBBDepDatabase,
    OmegaBBDepTables,
)

# A conversion script from rosetta omega bbdep tables to
# tmol probability tables.  For each of five classes
# of AA (general, gly, val+ile, pro, and prepro) it computes


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


def create_omega_db(r3_bbdepomega_dir):
    """
    Write the BBDep omega binary file
    """
    tables = parse_all_tables(r3_bbdepomega_dir)

    bbdep_omega_tables = []
    # write tables
    for aa, (mu, sig) in tables.items():
        bbdep_omega_tables.append(
            OmegaBBDepTables(
                table_id=aa,
                mu=mu,
                sigma=sig,
                bbstep=[numpy.pi / 18.0, numpy.pi / 18.0],
                bbstart=[numpy.pi / 36.0, numpy.pi / 36.0],
            )
        )

    path_lookup = "tmol/database/default/scoring/omega_bbdep.yaml"

    with open(path_lookup, "r") as infile_lookup:
        raw = yaml.safe_load(infile_lookup)
        bbdep_omega_lookup = cattr.structure(
            raw["omega_bbdep_lookup"],
            attr.fields(OmegaBBDepDatabase).bbdep_omega_lookup.type,
        )

    input_uniq_id = hash(r3_bbdepomega_dir)

    uniq_id = path_lookup + "," + str(input_uniq_id)

    return OmegaBBDepDatabase(
        uniq_id=uniq_id,
        bbdep_omega_lookup=bbdep_omega_lookup,
        bbdep_omega_tables=bbdep_omega_tables,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosetta_dir", default=os.path.join(Path.home(), "Rosetta/main/")
    )
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--r3_bbdepomega_dir",
        default=os.path.join(
            args.rosetta_dir, "database/scoring/score_functions/omega/"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../database/default/scoring/omega_bbdep.zip",
        ),
    )
    args = parser.parse_args()

    torch.save(create_omega_db(args.r3_bbdepomega_dir), args.output)
