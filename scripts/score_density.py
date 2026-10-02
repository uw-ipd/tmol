#!/usr/bin/env python
"""Score one or more CIF models against a cryo-EM map."""

from __future__ import annotations

import argparse

import numpy
import torch

from tmol.io import atom_array_from_cif
from tmol.score.density import DensityCorrelation


_ELEMENT_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "K": 19,
    "CA": 20,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "ZN": 30,
}


def structure_tensors(path, model, device):
    atoms = atom_array_from_cif(path, model=model)
    elements = numpy.char.upper(atoms.element.astype(str))
    heavy = elements != "H"
    coords = torch.as_tensor(atoms.coord[heavy], dtype=torch.float32, device=device)
    atomic_numbers = torch.tensor(
        [_ELEMENT_Z.get(element, 6) for element in elements[heavy]],
        dtype=torch.int64,
        device=device,
    )
    return coords, atomic_numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("structures", nargs="+")
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--models", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    scorer = DensityCorrelation.from_mrc(
        args.map,
        args.resolution,
        device=args.device,
        atom_chunk_size=args.chunk_size,
    )
    for structure in args.structures:
        for model in range(1, args.models + 1):
            try:
                coords, atomic_numbers = structure_tensors(
                    structure, model, args.device
                )
            except (IndexError, ValueError):
                if model == 1:
                    raise
                break
            cc = scorer.correlation(coords, atomic_numbers)
            print(
                f"{structure}\tmodel={model}\tatoms={coords.shape[0]}"
                f"\tCC={cc.item():.6f}"
            )


if __name__ == "__main__":
    main()
