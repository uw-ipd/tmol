import requests

def fetch_pdb(pdbid):
    return requests.get("https://files.rcsb.org/download/%s.pdb" % str.upper(pdbid)).text
