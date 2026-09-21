import requests


def fetch_pdb(pdbid):
    """Download a PDB-format structure from the RCSB Protein Data Bank."""
    return requests.get(
        "https://files.rcsb.org/download/%s.pdb" % str.upper(pdbid)
    ).text
