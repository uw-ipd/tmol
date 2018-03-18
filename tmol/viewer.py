import torch

from IPython.display import display

import tmol.extern.py3dmol as py3dmol

class SystemViewer:

    def __init__(self, target, score = None):
        self.target = target
        self.view = py3dmol.view(1200, 600)

        self.target = target
        self.score = score

        if self.score is not None:
            self.score.bond_graph = self.target.bond_graph
            self.score.coords = torch.autograd.Variable(torch.Tensor(self.target.coords), requires_grad=False)

        self.pdb = None

        self.update()
        self.view.zoomTo()
        self.update()

    def update(self):
        self.view.clear()

        if self.score is not None:
            self.score.coords = torch.autograd.Variable(torch.Tensor(self.target.coords), requires_grad=False)

        self.pdb = self.target.to_pdb(
            b = self.score.atom_scores.numpy() if self.score else None
        )
        self.view.addModel(self.pdb, "pdb")
        if self.score:
            self.view.setStyle({"sphere" : {"colorscheme" : {"prop":'b',"gradient": 'rwb',"min":1,"max":-1}}})
        else:
            self.view.setStyle({"sphere" : {}})

        display(self.view.update())
