"""Gumbel-top-k selection by tokens (MATES paper, Sec. 3.2; docs/mates_baseline.md "Selection").

Every candidate x gets the key s(x) / tau + G(x), G iid Gumbel(0, 1): sorting by the key draws documents without
replacement with probability proportional to exp(s / tau) at each draw (Plackett-Luce). Documents are taken in key
order until they hold the stage's token budget, the last one cut to fill it exactly, and the taken list is shuffled:
that is the stage's training order. s is the influence model's prediction of the z-scored oracle (higher = better).
The released code's variant (the k smallest of pred / tau + G on the predicted post-step loss, default tau 0.5) is not
used; the paper's form is.
"""

import numpy as np


def gumbel_top_k(scores, lengths, budget, tau, rng):
    """scores, lengths (n,); budget: tokens to select; rng: numpy Generator (noise first, then the shuffle).
    -> dict: gumbel (n,) the noise, selected (n,) bool, order: the selected candidates' indices in training order,
    ends: their token counts in that order (the capped length, less for the one cut document)."""
    scores = np.asarray(scores, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if int(lengths.sum()) < budget:
        raise ValueError(f"candidates hold {int(lengths.sum())} tokens, fewer than the budget {budget}")
    g = rng.gumbel(size=len(scores))
    ranked = np.argsort(-(scores / tau + g), kind="stable")
    cum = np.cumsum(lengths[ranked])
    k = int(np.searchsorted(cum, budget))               # first position where the taken tokens reach the budget
    take = ranked[: k + 1]
    ends = lengths[take].copy()
    ends[-1] = budget - (int(cum[k - 1]) if k > 0 else 0)   # the last one cut (or exactly full)
    selected = np.zeros(len(scores), dtype=bool)
    selected[take] = True
    perm = rng.permutation(len(take))
    return {"gumbel": g, "selected": selected, "order": take[perm], "ends": ends[perm]}
