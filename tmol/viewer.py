import functools
import torch

from IPython.display import display

import tmol.extern.py3dmol as py3dmol
from tmol.io.generic import to_pdb

class SystemViewer:

    def __init__(self, system):
        self.system = system
        self.view = py3dmol.view(1200, 600)

        self.system = system

        self.pdb = None

        self.update()
        self.view.zoomTo()
        self.update()

    def update(self):
        self.view.clear()

        self.pdb = to_pdb(self.system)

        self.view.addModel(self.pdb, "pdb")
        self.view.setStyle({"sphere" : {}})

        display(self.view.update())
