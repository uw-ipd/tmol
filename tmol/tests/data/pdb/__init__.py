import os

def get(pdbid):
    root = os.path.dirname(__file__)
    dfile = os.path.join(root, pdbid.upper() + ".pdb")

    with open(dfile, "r") as f:
        return f.read()
