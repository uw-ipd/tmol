import torch
import lbfgs_armijo


class SimpleLJScore:
    def __init__(self, r_m=1.0, epsilon=1.0):
        self.r_m = r_m
        self.epsilon = epsilon

    def __call__(self, coords):
        ind = torch.arange(coords.shape[0], requires_grad=False)

        ind_a = ind.view((-1, 1))
        ind_b = ind.view((1, -1))
        deltas = coords.view((-1, 1, 3)) - coords.view((1, -1, 3))

        dist = torch.norm(deltas, 2, -1)

        fd = (self.r_m / dist)
        fd2 = fd * fd
        fd6 = fd2 * fd2 * fd2
        fd12 = fd6 * fd6
        lj = self.epsilon * (fd12 - 3 * fd6)

        self.lj = torch.where(ind_a != ind_b, lj, torch.Tensor([0.0]))

        self.atom_scores = torch.sum(self.lj.detach(), dim=-1)
        self.total_score = torch.sum(self.lj)

        return self


if __name__ == '__main__':
    dtype = torch.float
    device = torch.device("cpu")

    Natoms = 100
    x = torch.randn(Natoms, 3, device=device, dtype=dtype, requires_grad=True)
    scorefunc = SimpleLJScore(r_m=1.0, epsilon=1.0)

    learning_rate = 1e-3

    optimizer = lbfgs_armijo.LBFGS_Armijo([x])

    def closure():
        optimizer.zero_grad()
        E = scorefunc(10 * x)
        E.total_score.backward()
        return E.total_score

    print(closure())
    optimizer.step(closure)
    print(closure())
