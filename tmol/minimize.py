import scipy.optimize

import properties
from tmol.properties.array import Array

import torch.autograd

from tmol.system import FixedNamedAtomSystem
from tmol.score import ScoreGraph

CoordArray = Array("minimizable view of system coords", dtype="f4", cast="unsafe")[:]

class SimpleMinimizer(properties.HasProperties):
    system = properties.Instance("target system", ScoreGraph)
    
    @CoordArray
    def coords(self):
        return self.system.coords.detach().numpy().reshape(-1)
    
    @coords.setter
    def coords(self, x):
        x = x.reshape(self.system.coords.shape)
        
        del self.system.coords
        self.system.coords = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
        
    def fun(self, x):
        self.coords = x
        
        return (
            self.system.total_score.detach().numpy(),
            torch.autograd.grad(self.system.total_score, self.system.coords)[0].numpy().reshape(-1)
        )
    
    def minimize(self, options=dict(maxiter=250), **kwargs):
        self.result = scipy.optimize.minimize(
            self.fun,
            self.coords,
            jac=True,
            options = options,
            **kwargs
        )
        
        return self
