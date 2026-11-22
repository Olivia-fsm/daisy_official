"""The scorer of DAiSY: a running ridge regression from the frozen embedding phi(x) to the Gaussianized ranks of the
utility labels, S(x) = w^T [phi(x); 1], refitted in closed form every U steps with exponential forgetting (paper
Section 3.3)."""

import torch


def rank_transform(y):
    """Per-step label transform: Gaussianized ranks (robust to the drifting scale and the heavy tails of the labels)."""
    n = len(y)
    r = torch.empty(n, device=y.device)
    r[torch.argsort(y)] = torch.arange(n, device=y.device, dtype=torch.float32)
    return torch.special.ndtri((r + 0.5) / n)


class RunningRidge:
    """Ridge regression phi -> t over all labelled steps so far, with exponential forgetting (one factor `decay` per
    labelled step), in closed form. Regularization: lam x the mean eigenvalue of the (uncentered) Gram matrix of phi;
    the bias is not regularized."""

    def __init__(self, d, device, lam=0.01, decay=0.999):
        self.lam, self.decay, self.dev = lam, decay, device
        self.A = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
        self.b = torch.zeros(d + 1, dtype=torch.float64, device=device)
        self.w = None
        self.n = 0

    def add(self, phi, t):
        """One labelled step: phi (n, d), targets t (n,)."""
        X = torch.cat([phi.double(), torch.ones(len(phi), 1, dtype=torch.float64, device=phi.device)], 1)
        self.A.mul_(self.decay).add_(X.T @ X)
        self.b.mul_(self.decay).add_(X.T @ t.double())
        self.n += 1

    def solve(self):
        if self.n == 0:
            return
        d = self.A.shape[0] - 1
        reg = torch.zeros_like(self.A)
        reg[:d, :d] = torch.eye(d, dtype=torch.float64, device=self.A.device) * (self.lam * float(self.A[:d, :d].trace()) / d)
        self.w = torch.linalg.solve(self.A + reg, self.b).float()

    def score(self, phi):
        return phi @ self.w[:-1] + self.w[-1]

    def state_dict(self):
        return {"A": self.A.cpu(), "b": self.b.cpu(), "w": None if self.w is None else self.w.cpu(), "n": self.n}

    def load_state_dict(self, st):
        self.A, self.b, self.n = st["A"].to(self.dev), st["b"].to(self.dev), st["n"]
        self.w = None if st["w"] is None else st["w"].to(self.dev)
