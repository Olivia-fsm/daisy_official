import torch

from daisy.models.base import LayerNorm
from daisy.models.llama import Llama, RMSNorm

# modules whose weights are not weight-decayed
NO_DECAY_MODULES = (torch.nn.LayerNorm, LayerNorm, RMSNorm, torch.nn.Embedding)


def get_model(args):
    """The language model of a run: a Llama-style decoder built from the run's configuration."""
    return Llama(args)
