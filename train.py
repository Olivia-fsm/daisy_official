"""Train a language model with DAiSY, or with one of the baselines that share its data stream (uniform sampling,
similarity, DSIR, MATES). See README.md and scripts/train.sh; `python train.py --help` lists every flag.

    torchrun --standalone --nproc_per_node=4 train.py --distributed_backend nccl --experiment_name NAME \\
        --daisy_index_dir INDEX --daisy_pool POOL ... (scripts/train.sh)
"""

import inspect
import json
import random
from pathlib import Path

import numpy as np
import torch

from daisy import config, distributed
from daisy.models import get_model
from daisy.train.trainer import train_daisy


def build_optimizer(args, model, backend):
    """AdamW over the model's two parameter groups (with and without weight decay)."""
    group_specs = backend.get_raw_model(model).get_parameter_group_specs(config=args)
    by_name = {name: p for name, p in model.named_parameters()}
    for g in group_specs:
        g["params"] = [by_name[n] for name in g["params"] for n in backend.translate_model_parameter_name_for_node(name)]
    fused = "cuda" in args.device and "fused" in inspect.signature(torch.optim.AdamW).parameters
    opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            **({"fused": True} if fused else {}))
    return opt, group_specs


def build_scheduler(args, opt, group_specs):
    """Warmup, then cosine (or linear) decay over --scheduler_total_steps (default: --iterations)."""
    if args.scheduler == "none":
        return None
    total = args.scheduler_total_steps or args.iterations
    assert args.warmup_steps < total, "the warmup must be shorter than the schedule"
    # the initial lr is lr / div_factor, the final one lr / div_factor / final_div_factor
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt, max_lr=[g.get("lr", args.lr) for g in group_specs], total_steps=total,
        pct_start=args.warmup_steps / total, anneal_strategy=args.scheduler, cycle_momentum=False, div_factor=1e2,
        final_div_factor=args.final_div_factor)


def main(args):
    backend = distributed.make_backend_from_args(args)
    args = backend.get_adjusted_args_for_process(args)      # per-process batch size, device and seed
    args.world_size = backend.get_world_size()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    exp_dir = Path(args.results_base_folder) / args.experiment_name
    master = backend.is_master_process()
    if master and args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args), entity=args.wandb_entity)
        wandb.define_metric("iter")
        for prefix in ("train/*", "daisy/*", "lr", "tokens"):
            wandb.define_metric(prefix, step_metric="iter")
    if master:
        print(f"Experiment directory: {exp_dir}\nConfig:\n{vars(args)}\n")

    model = get_model(args).to(args.device)
    model = backend.transform_model(model)
    opt, group_specs = build_optimizer(args, model, backend)
    if master:
        print(f"number of parameters: {backend.get_raw_model(model).get_num_params() / 1e6:.2f}M")
    scheduler = build_scheduler(args, opt, group_specs)

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(f"{exp_dir} already has a checkpoint: pass --auto_resume true to continue it, or choose "
                             f"another --experiment_name")
        args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif master:
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train_daisy(model=model, opt=opt, scheduler=scheduler, exp_dir=exp_dir, distributed_backend=backend,
                        cfg=args)
    stats["args"] = vars(args)
    if master:
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(stats, f)
    backend.finalize()


if __name__ == "__main__":
    main(config.parse_args())
