import os
from .setup import generate

if __name__ == "__main__":

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs).03d %(name)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

    import tmol.support.rosetta.init  # noqa

    generate(target_pdbs=["1ubq"], output_dir=os.path.dirname(__file__))
