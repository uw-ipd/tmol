import torch
from tmol.kinematics.metadata import DOFTypes

# modules for cartesian and torsion-space optimization
#
# unlike typical NN training, the model is fixed and we want to optimize inputs
#   therefore, the coordinates are parameters
#   there are no model inputs
#
# potentially a dof mask could be added here (?)
#  - or we might want to keep that with dof creation


# cartesian space minimization
class CartesianEnergyNetwork(torch.nn.Module):
    def __init__(self, score_graph):
        super(CartesianEnergyNetwork, self).__init__()

        # scoring graph
        self.graph = score_graph

        # parameters
        self.dofs = torch.nn.Parameter(self.graph.coords)

    def forward(self):
        self.graph.coords = self.dofs
        self.graph.reset_coords()
        return self.graph.intra_score().total


# mask out relevant dofs to the minimizer
class DOFMaskingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fg, mask, bg):
        ctx.mask = mask
        ctx.fg = fg
        bg[mask] = fg
        return bg

    @staticmethod
    def backward(ctx, grad_output):
        grad = torch.zeros_like(ctx.fg)
        grad = grad_output[ctx.mask]
        return grad, None, None


# torsion space minimization
class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_graph):
        super(TorsionalEnergyNetwork, self).__init__()

        # scoring graph
        self.graph = score_graph

        # todo: make this a configurable parameter
        #   (for now it defaults to torsion minimization)
        dofmask = self.graph.dofmetadata[
            self.graph.dofmetadata.dof_type == DOFTypes.bond_torsion
        ]
        self.mask = (dofmask.node_idx, dofmask.dof_idx)

        # parameters
        self.dofs = torch.nn.Parameter(self.graph.dofs[self.mask])

    def forward(self):
        self.graph.dofs = DOFMaskingFunc.apply(self.dofs, self.mask, self.graph.dofs)
        self.graph.reset_coords()
        return self.graph.intra_score().total
