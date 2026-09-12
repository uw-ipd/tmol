"""Executable tutorial examples; these are application code, not tmol APIs."""

import torch

from tmol.io import (
    CanonicalForm,
    default_canonical_ordering,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)


def prepare_named_layout(sequence, chain_id, residue_names, atom_names):
    """Prepare a model's fixed sequence/layout once; bind coordinates in Torch.

    This example uses the default protein database. General chemistry callers
    should use the ordering and packed types from their PoseBuildContext.
    Padding is explicitly identified by chain_id == -1.
    """
    if sequence.ndim != 2 or chain_id.shape != sequence.shape:
        raise ValueError("sequence and chain_id must have shape [batch, residues]")
    if sequence.dtype not in (torch.int32, torch.int64):
        raise TypeError("sequence must contain integer token IDs")
    if chain_id.dtype not in (torch.int32, torch.int64):
        raise TypeError("chain_id must contain integer chain IDs")
    if sequence.device != chain_id.device or torch.any(chain_id < -1):
        raise ValueError(
            "chain IDs must share the sequence device and use -1 for padding"
        )
    present = chain_id >= 0
    if torch.any(present & ((sequence < 0) | (sequence >= len(residue_names)))):
        raise ValueError("An observed residue has an unknown token ID")
    sequence = sequence.clone().long()
    sequence[~present] = 0
    chain_id = chain_id.clone().to(torch.int32)
    ordering = default_canonical_ordering()
    packed_types = default_packed_block_types(sequence.device)
    restype_map, atom_map, real_atoms = ordering.create_src_2_tmol_mappings(
        residue_names, atom_names, sequence.device
    )
    restypes = restype_map[sequence].to(torch.int32)
    if torch.any(present & (restypes < 0)):
        raise ValueError("A residue needs a chemical definition before construction")
    restypes[~present] = -1
    real = real_atoms[sequence] & present.unsqueeze(-1)
    pose, residue, slot = torch.nonzero(real, as_tuple=True)
    target_atom = atom_map[sequence][real]
    shape = (*sequence.shape, real.shape[-1], 3)

    def build(coords, *, observed=None):
        if tuple(coords.shape) != shape or coords.dtype != torch.float32:
            raise ValueError(f"Expected float32 coordinates shaped {shape}")
        if coords.device != sequence.device:
            raise ValueError("Coordinates and prepared mapping must share a device")
        source = coords[pose, residue, slot]
        if observed is not None:
            if observed.shape != coords.shape[:-1] or observed.dtype != torch.bool:
                raise ValueError(
                    "observed must be a boolean mask over coordinate slots"
                )
            source = torch.where(
                observed[pose, residue, slot, None], source, float("nan")
            )
        valid = torch.isfinite(source).all(-1) | torch.isnan(source).all(-1)
        if not bool(valid.all()):
            raise ValueError(
                "Supply finite coordinate triplets or all-NaN missing atoms"
            )
        canonical = coords.new_full(
            (*sequence.shape, ordering.max_n_canonical_atoms, 3), float("nan")
        )
        canonical[pose, residue, target_atom] = source
        form = CanonicalForm(
            chain_id=chain_id,
            res_types=restypes,
            coords=canonical,
            chain_labels=None,
            res_labels=None,
            residue_insertion_codes=None,
            atom_occupancy=None,
            atom_b_factor=None,
            disulfides=None,
            res_not_connected=None,
        )
        return pose_stack_from_canonical_form(ordering, packed_types, *form)

    return build


def openfold_example(prediction):
    """Adapt this predictor's token IDs, final atom14 coordinates and chain mask."""
    from atomworks.ml.encoding_definitions import AF2_ATOM14_ENCODING

    # This token order is part of the predictor's contract, independent of shape.
    names = list(AF2_ATOM14_ENCODING.token_atoms)[:20]
    builder = prepare_named_layout(
        prediction["aatype"],
        prediction["chain_index"],
        names,
        AF2_ATOM14_ENCODING.token_atoms,
    )
    return builder(
        prediction["positions"][-1], observed=prediction["atom14_atom_exists"].bool()
    )


def rf2_example(prediction, residue_names, atom_names, *, hydrogens):
    """Use num2aa/aa2long from the RF2 version that produced this prediction.

    ``hydrogens='preserve'`` retains supplied nonterminal H coordinates;
    ``'rebuild'`` marks all supplied H as missing. The generic amide H at each
    chain's N terminus is always rebuilt as the appropriate terminal hydrogens.
    """
    if hydrogens not in ("preserve", "rebuild"):
        raise ValueError("Choose hydrogens='preserve' or 'rebuild'")
    sequence = prediction["seq"].unsqueeze(0)
    coords = prediction["xyz"].unsqueeze(0)
    lengths = prediction["chainlens"]
    if any(length <= 0 for length in lengths) or sum(lengths) != sequence.shape[1]:
        raise ValueError("Positive chain lengths must cover the entire sequence")
    chain_id = torch.repeat_interleave(
        torch.arange(len(lengths), device=coords.device),
        torch.tensor(lengths, device=coords.device),
    ).unsqueeze(0)
    names = {
        name: [atom.strip() if atom else "" for atom in row]
        for name, row in zip(residue_names, atom_names)
    }
    builder = prepare_named_layout(sequence, chain_id, residue_names, names)
    hydrogen_slots = torch.tensor(
        [
            [atom.lstrip("123").startswith("H") for atom in names[name]]
            for name in residue_names
        ],
        device=coords.device,
    )
    amide_h = torch.tensor(
        [[atom == "H" for atom in names[name]] for name in residue_names],
        device=coords.device,
    )
    nterm = torch.ones_like(chain_id, dtype=torch.bool)
    nterm[:, 1:] = chain_id[:, 1:] != chain_id[:, :-1]
    suppressed = amide_h[sequence] & nterm.unsqueeze(-1)
    if hydrogens == "rebuild":
        suppressed |= hydrogen_slots[sequence]
    return builder(coords, observed=~suppressed)
