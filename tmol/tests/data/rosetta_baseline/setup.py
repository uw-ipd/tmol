import logging
import pickle

from tmol.support.scoring.rosetta import PoseScoreWrapper
from tmol.tests.data.pdb import data


def generate(target_pdbs, output_dir):
    for target in target_pdbs:
        logging.info(f"Scoring target: {target}")
        scores = PoseScoreWrapper.from_pdbstring(data[target])

        logging.info(f"Writing: {output_dir}/{target}.scores.pickle")
        with open(f"{output_dir}/{target}.scores.pickle", "wb") as of:
            pickle.dump(scores, of)
