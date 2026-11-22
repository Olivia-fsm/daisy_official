"""Per-step scorer metrics (the "scorer" records of metrics.jsonl): the documents of every labelled step are scored
BEFORE the scorer learns from them, so these are held-out measures of how well it ranks new documents.

  spearman, pearson   correlation of the scores with the labels y
  top25               precision of the top quarter
  mse_cal             1 - pearson(S, t)^2: MSE to the regression target t (the Gaussianized ranks) after the best
                      per-step rescaling a S + b, per unit variance of t (0 = perfect, 1 = no information)
  mse_raw             mean (S - t)^2 as is
"""

import torch

from daisy.scorers.ridge import rank_transform


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b)) if len(a) > 2 else float("nan")


def pearson(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def top_precision(s, y, frac=0.25):
    k = max(1, int(round(frac * len(y))))
    a, b = set(torch.topk(s, k).indices.tolist()), set(torch.topk(y, k).indices.tolist())
    return len(a & b) / k


def _ranks(y):
    r = torch.empty(len(y), device=y.device)
    r[torch.argsort(y)] = torch.arange(len(y), device=y.device, dtype=torch.float32)
    return r


def step_metrics(S, y, t=None):
    """All per-step metrics of the scores S on one step's documents with labels y (t: their Gaussianized ranks)."""
    S = S.float()
    t = rank_transform(y) if t is None else t
    return {"spearman": spearman(S, y), "pearson": pearson(S, y), "top25": top_precision(S, y),
            "mse_cal": 1.0 - pearson(S, t) ** 2, "mse_raw": float(((S - t) ** 2).mean())}
