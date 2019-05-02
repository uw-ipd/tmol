import torch

# modules for cartesian and torsion-space optimization
#
# unlike typical NN training, the model is fixed and we want to optimize inputs
#   therefore, the coordinates are parameters
#   there are no model inputs
#
# potentially a dof mask could be added here (?)
#  - or we might want to keep that with dof creation


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


class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_graph):
        super(TorsionalEnergyNetwork, self).__init__()

        # scoring graph
        self.graph = score_graph

        # parameters
        self.dofs = torch.nn.Parameter(self.graph.dofs)

    def forward(self):
        self.graph.dofs = self.dofs
        self.graph.reset_coords()
        return self.graph.intra_score().total
