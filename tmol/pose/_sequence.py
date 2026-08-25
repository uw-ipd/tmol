"""Parsing of sequence strings into block-type names.

Grammar, one string per pose:

    A          protein one-letter code
    a          DNA one-letter code
    X[NAME]    explicit block type, e.g. A[ALA:nterm], a[RA], X[ATP]
    X(SMILES)  ligand built through the ligand pipeline
    :          chain break
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import attr

# Terminal variants tried in order; the first one present in the residue type
# set wins.
_NTERM_VARIANTS = ("nterm", "na5prime")
_CTERM_VARIANTS = ("cterm", "na3prime")

# Backbone type a one-letter code refers to, by case.
_UPPER_BACKBONE = "alpha"
_LOWER_BACKBONE = "dna"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SeqToken:
    """One residue position in a sequence string."""

    letter: str
    name: Optional[str] = None
    smiles: Optional[str] = None


def tokenize_sequences(
    seqs: Union[str, Sequence[str]],
) -> Tuple[List[List[SeqToken]], List[List[int]]]:
    """Split sequence strings into per-pose tokens and chain lengths."""
    if isinstance(seqs, str):
        seqs = [seqs]
    tokens = []
    chain_lengths = []
    for seq in seqs:
        seq_tokens, seq_chains = _tokenize_one(seq)
        tokens.append(seq_tokens)
        chain_lengths.append(seq_chains)
    return tokens, chain_lengths


def smiles_in_tokens(tokens: List[List[SeqToken]]) -> List[str]:
    """Unique SMILES strings appearing in the tokenized sequences."""
    seen = {}
    for seq_tokens in tokens:
        for tok in seq_tokens:
            if tok.smiles is not None:
                seen[tok.smiles] = None
    return list(seen)


def resolve_block_type_names(
    tokens: List[List[SeqToken]],
    chain_lengths: List[List[int]],
    restype_set,
    ligand_names: Optional[Dict[str, str]] = None,
    termini: bool = True,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Turn tokens into block type names and chain lengths.

    Chains are split wherever a polymer bond cannot form, so ligands and waters
    end up in chains of their own.
    """
    rt_by_name = {rt.name: rt for rt in restype_set.residue_types}
    patched = _patched_name_index(restype_set)
    one_letter = _one_letter_index(restype_set)

    names = [
        [
            _resolve_token(tok, rt_by_name, one_letter, ligand_names)
            for tok in seq_tokens
        ]
        for seq_tokens in tokens
    ]
    chain_lengths = [
        _split_unbondable_chains(seq_names, seq_chains, rt_by_name)
        for seq_names, seq_chains in zip(names, chain_lengths)
    ]
    if termini:
        for seq_names, seq_chains in zip(names, chain_lengths):
            _apply_termini(seq_names, seq_chains, rt_by_name, patched)
    return names, chain_lengths


def _tokenize_one(seq: str) -> Tuple[List[SeqToken], List[int]]:
    tokens: List[SeqToken] = []
    chain_lengths: List[int] = []
    n_in_chain = 0
    i = 0
    while i < len(seq):
        c = seq[i]
        if c == ":":
            if n_in_chain == 0:
                raise ValueError(f"empty chain at position {i} of {seq!r}")
            chain_lengths.append(n_in_chain)
            n_in_chain = 0
            i += 1
            continue
        if not c.isalpha():
            raise ValueError(f"invalid character {c!r} at position {i} of {seq!r}")
        i += 1
        name = smiles = None
        if i < len(seq) and seq[i] in "[(":
            is_name = seq[i] == "["
            body, i = _read_delimited(seq, i)
            name, smiles = (body, None) if is_name else (None, body)
        tokens.append(SeqToken(letter=c, name=name, smiles=smiles))
        n_in_chain += 1
    if n_in_chain == 0:
        raise ValueError(f"empty chain at the end of {seq!r}")
    chain_lengths.append(n_in_chain)
    return tokens, chain_lengths


def _read_delimited(seq: str, start: int) -> Tuple[str, int]:
    """Body of the bracketed group opening at ``start``, and the index past it."""
    open_c = seq[start]
    close_c = "]" if open_c == "[" else ")"
    depth = 0
    for j in range(start, len(seq)):
        if seq[j] == open_c:
            depth += 1
        elif seq[j] == close_c:
            depth -= 1
            if depth == 0:
                body = seq[start + 1 : j]
                if not body:
                    raise ValueError(f"empty {open_c}{close_c} at position {start}")
                return body, j + 1
    raise ValueError(f"unbalanced {open_c!r} at position {start} of {seq!r}")


def _one_letter_index(restype_set) -> Dict[Tuple[str, str], str]:
    index: Dict[Tuple[str, str], str] = {}
    for rt in restype_set.residue_types:
        if rt.one_letter_code is None:
            continue
        key = (rt.properties.polymer.backbone_type, rt.one_letter_code)
        if key in index:
            raise ValueError(f"duplicate one-letter code for {key}: {rt.name}")
        index[key] = rt.name
    return index


def _patched_name_index(restype_set) -> Dict[Tuple[str, frozenset], str]:
    index = {}
    for rt in restype_set.residue_types:
        base, _, suffixes = rt.name.partition(":")
        if suffixes:
            index[(base, frozenset(suffixes.split(":")))] = rt.name
    return index


def _resolve_token(tok, rt_by_name, one_letter, ligand_names) -> str:
    if tok.smiles is not None:
        if ligand_names is None or tok.smiles not in ligand_names:
            raise ValueError(f"no residue type prepared for SMILES {tok.smiles!r}")
        return ligand_names[tok.smiles]
    if tok.name is not None:
        base = tok.name.split("--")[0]
        if base not in rt_by_name:
            raise ValueError(f"unknown block type {base!r}")
        return tok.name
    backbone = _UPPER_BACKBONE if tok.letter.isupper() else _LOWER_BACKBONE
    key = (backbone, tok.letter)
    if key not in one_letter:
        raise ValueError(f"unknown {backbone} one-letter code {tok.letter!r}")
    return one_letter[key]


def _split_unbondable_chains(seq_names, seq_chains, rt_by_name) -> List[int]:
    split = []
    start = 0
    for chain_len in seq_chains:
        run = 1
        for pos in range(start + 1, start + chain_len):
            prev = rt_by_name[seq_names[pos - 1].split("--")[0]]
            cur = rt_by_name[seq_names[pos].split("--")[0]]
            if prev.up_connection_ind == -1 or cur.down_connection_ind == -1:
                split.append(run)
                run = 0
            run += 1
        split.append(run)
        start += chain_len
    return split


def _apply_termini(seq_names, seq_chains, rt_by_name, patched):
    start = 0
    for chain_len in seq_chains:
        for pos in {start, start + chain_len - 1}:
            name = seq_names[pos]
            if ":" in name or "--" in name:
                continue
            rt = rt_by_name[name]
            wanted = []
            if pos == start and rt.down_connection_ind != -1:
                wanted.append(_NTERM_VARIANTS)
            if pos == start + chain_len - 1 and rt.up_connection_ind != -1:
                wanted.append(_CTERM_VARIANTS)
            for variants in _variant_combinations(wanted):
                patched_name = patched.get((name, frozenset(variants)))
                if patched_name is not None:
                    seq_names[pos] = patched_name
                    break
        start += chain_len


def _variant_combinations(wanted):
    """Candidate variant sets, most-preferred first."""
    if not wanted:
        return []
    combos = [()]
    for variants in wanted:
        combos = [combo + (v,) for combo in combos for v in variants]
    return combos
