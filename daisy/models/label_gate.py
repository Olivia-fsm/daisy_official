"""Label gates: the utility label of every trained document, read off the ordinary training backward pass (paper
Section 3.2), with no hooks and no side effects (works under torch.compile and DDP).

For a tracked Linear with per-token input a and output y = W a, `_LabelGate` is the identity on y in the forward pass.
In the backward pass it passes the output gradient g = dL/dy through unchanged and, from the same (a, g), returns as
the gradient of a zero "gate" input z of shape (B, T), one per reference matrix M:

    dL/dz_tok = g_tok^T M a_tok            (M = P_t ⊙ grad L_V restricted to the layer, daisy/labels/gates.py)

so z.grad holds the token's contribution to the label, and the document's label is the sum over its tokens and over
the tracked layers. The weight gradients are untouched: training is identical with and without the gates.

Used only when the model is called with `gates=...`. The `gates` dict (daisy/labels/gates.py: TargetLabels.make):
    {"z": {layer: [z_0, ...]}, "mats": {layer: [M_0, ...]}}        one gate per (layer, target)
"""

import torch


class _LabelGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, a, n, *tensors):
        # tensors: n x (z, M), the gates and their reference matrices
        ctx.n = n
        ctx.z_shapes = [tensors[2 * i].shape for i in range(n)]
        ctx.save_for_backward(a, *[tensors[2 * i + 1] for i in range(n)])
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        saved = ctx.saved_tensors
        a = saved[0]
        g2 = g.reshape(-1, g.shape[-1]).float()
        a2 = a.reshape(-1, a.shape[-1]).float()
        out = [g, None, None]
        for i in range(ctx.n):
            M = saved[1 + i]
            if M.shape[0] <= M.shape[1]:      # (tok, d_out) intermediate
                s = ((a2 @ M.T) * g2).sum(-1)
            else:                             # (tok, d_in) intermediate
                s = ((g2 @ M) * a2).sum(-1)
            out += [s.reshape(ctx.z_shapes[i]), None]
        return tuple(out)


def gated_linear(mod, x, gates):
    """mod(x), with the label gate when `gates` holds references for this layer (the layer must have `gate_name`)."""
    name = getattr(mod, "gate_name", None)
    if gates is None or name is None or name not in gates["mats"]:
        return mod(x)
    # Cast once here under autocast: the Linear then saves this same tensor for its weight gradient, and the gate
    # saves it too (one copy). autocast would have produced exactly this cast inside mod(x).
    dev = x.device.type
    if torch.is_autocast_enabled(dev) and x.dtype == torch.float32:
        x = x.to(torch.get_autocast_dtype(dev))
    y = mod(x)
    tensors = []
    for z, M in zip(gates["z"][name], gates["mats"][name]):
        tensors += [z, M]
    return _LabelGate.apply(y, x, len(gates["mats"][name]), *tensors)
