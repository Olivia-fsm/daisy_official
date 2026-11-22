from daisy.distributed import ddp, single

BACKENDS = {"nccl": ddp.DataParallelDistributedBackend, None: single.SingleNodeBackend}


def make_backend_from_args(args):
    return BACKENDS[args.distributed_backend](args)
