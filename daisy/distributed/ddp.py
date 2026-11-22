import datetime
import math
import os
from contextlib import contextmanager

from torch.distributed import destroy_process_group, get_world_size, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from daisy.distributed.backend import DistributedBackend


class DataParallelDistributedBackend(DistributedBackend):
    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", -1))
        assert self.rank != -1, "the DDP backend needs RANK (start the run with torchrun)"
        assert "cuda" in args.device, "the DDP backend needs CUDA devices"
        # timeout of a collective (minutes): DAISY_NCCL_TIMEOUT_MIN lets a rank that is slow but recovering (a long
        # recompilation, an I/O stall on one node) finish instead of the whole job aborting
        timeout = datetime.timedelta(minutes=float(os.environ.get("DAISY_NCCL_TIMEOUT_MIN", "10")))
        init_process_group(backend=args.distributed_backend, timeout=timeout)
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def get_adjusted_args_for_process(self, args):
        """The global step holds batch_size x acc_steps rows: every process trains batch_size x acc_steps / world of
        them, in acc_steps / gcd(acc_steps, world) micro-batches."""
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(f"the effective batch size {effective_batch_size} is not divisible by the world size "
                             f"{world_size}")
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.local_rank
        return args

    def transform_model(self, model):
        return DDP(model, device_ids=[self.local_rank])

    @contextmanager
    def get_context_for_microstep_forward(self, model, microstep_idx, gradient_accumulation_steps):
        # gradients are synchronized on the last micro-batch of a step only
        model.require_backward_grad_sync = microstep_idx == gradient_accumulation_steps - 1
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        return model.module

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f"module.{parameter_name}"]

    def get_world_size(self):
        return get_world_size()

    def finalize(self):
        destroy_process_group()
