"""Generic py3mol-based visualization."""

from IPython.display import display

import tmol.extern.py3dmol as py3dmol
from tmol.io.generic import to_pdb, to_cdjson


class SystemViewer:
    """Generic py3dmol-based jupyter display widget.

    A py3dmol-based jupyter viewing component, utilizing :py:mod:`tmol.io.generic` dispatch functions
    to render arbitrary model components.
    """

    transforms = {"cdjson": to_cdjson, "pdb": to_pdb}

    def __init__(self, system, style={"sphere": {}}, mode="cdjson"):
        self.system = system
        if isinstance(style, str):
            style = {style: {}}
        self.style = style
        self.mode = mode

        self.data = None

        self.view = py3dmol.view(1200, 600)

        self.update()
        self.view.zoomTo()
        self.update()

    def update(self):
        self.view.clear()

        self.data = self.transforms[self.mode](self.system)

        self.view.addModel(self.data, self.mode)
        self.view.setStyle(self.style)

        display(self.view.update())
