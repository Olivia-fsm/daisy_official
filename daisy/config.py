"""Command-line configuration of a training run (train.py). The defaults are those of the paper's 124M runs wherever
the paper fixes a value; scripts/train.sh sets the model size, the schedule and the paths."""

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ("true", "1", "yes")


def none_or_str(value):
    return None if value == "None" else value


def build_parser():
    p = argparse.ArgumentParser(allow_abbrev=False)

    g = p.add_argument_group("run")
    g.add_argument("--experiment_name", required=True, type=str, help="the run's directory under --results_base_folder")
    g.add_argument("--results_base_folder", default="./exps", type=str)
    g.add_argument("--seed", default=0, type=int, help="model initialization; also added to --data_seed for the stream")
    g.add_argument("--data_seed", default=1337, type=int)
    g.add_argument("--device", default="cuda:0", type=str)
    g.add_argument("--distributed_backend", default=None, type=none_or_str, choices=[None, "nccl"],
                   help="nccl: DistributedDataParallel under torchrun; None: one process")
    g.add_argument("--dtype", default="bfloat16", type=str, choices=["float32", "float16", "bfloat16"],
                   help="autocast dtype on CUDA")
    g.add_argument("--compile", action="store_true", help="torch.compile the model (all runs of the paper)")
    g.add_argument("--log_interval", default=50, type=int)
    g.add_argument("--latest_ckpt_interval", default=0, type=int, help="steps between ckpts/latest (0 = off)")
    g.add_argument("--permanent_ckpt_interval", default=0, type=int, help="steps between ckpts/<step> (0 = off)")
    g.add_argument("--extra_ckpt_iters", default="", type=str,
                   help="comma-separated steps with a permanent checkpoint in addition to the interval")
    g.add_argument("--resume_from", default=None, type=str, help="a checkpoint directory (set by --auto_resume)")
    g.add_argument("--auto_resume", default=True, type=str2bool,
                   help="continue from EXP/ckpts/latest when it exists")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb_project", default="daisy", type=str)
    g.add_argument("--wandb_entity", default=None, type=none_or_str)

    g = p.add_argument_group("model (Llama-style decoder)")
    g.add_argument("--vocab_size", default=50304, type=int)
    g.add_argument("--sequence_length", default=2048, type=int)
    g.add_argument("--n_embd", default=768, type=int)
    g.add_argument("--n_head", default=12, type=int)
    g.add_argument("--n_layer", default=12, type=int)
    g.add_argument("--multiple_of", default=256, type=int, help="the SwiGLU hidden size is rounded up to a multiple")
    g.add_argument("--rmsnorm_eps", default=1e-5, type=float)
    g.add_argument("--init_std", default=0.02, type=float)
    g.add_argument("--dropout", default=0.0, type=float)
    g.add_argument("--bias", default=False, type=str2bool)
    g.add_argument("--untied_embeds", action="store_true")

    g = p.add_argument_group("optimization (AdamW)")
    g.add_argument("--batch_size", default=16, type=int,
                   help="rows per micro-batch; under DDP: per GPU when --acc_steps equals the number of GPUs")
    g.add_argument("--acc_steps", default=1, type=int,
                   help="micro-batches per step; under DDP it is divided by the number of processes")
    g.add_argument("--iterations", default=256000, type=int)
    g.add_argument("--scheduler", default="cos", choices=["cos", "linear", "none"])
    g.add_argument("--scheduler_total_steps", default=None, type=int,
                   help="horizon of the learning-rate schedule (default: --iterations); a shorter run follows the "
                        "first --iterations steps of it")
    g.add_argument("--warmup_steps", default=2000, type=int)
    g.add_argument("--final_div_factor", default=1, type=float)
    g.add_argument("--lr", default=3e-4, type=float)
    g.add_argument("--beta1", default=0.8, type=float)
    g.add_argument("--beta2", default=0.999, type=float)
    g.add_argument("--weight_decay", default=0.1, type=float)
    g.add_argument("--grad_clip", default=0.5, type=float)

    g = p.add_argument_group("data")
    g.add_argument("--daisy_index_dir", required=True, type=str,
                   help="the embedding index (daisy/index/build_index.py): the held-out web-text set and the hashes that "
                        "exclude it from training; also the training documents when --daisy_pool is not given")
    g.add_argument("--daisy_pool", default="", type=str,
                   help="the tokenized pool with embedding sidecars (daisy/tokenize) for the training stream; '' = train "
                        "on the index's own documents")
    g.add_argument("--daisy_pool_open_shards", default=32, type=int, help="pool shards a rank draws documents from at once")
    g.add_argument("--daisy_pool_reshuffle", action="store_true",
                   help="when a rank's share of the pool is used up, start another pass over it with new shard and "
                        "document orders (large window factors)")
    g.add_argument("--daisy_pool_preload", default=8, type=int,
                   help="shards prepared in the background before they open (their document order and embeddings)")
    g.add_argument("--daisy_pool_subset_tokens", default=0, type=int,
                   help="limited-pool runs: train on a fixed subset of the pool with this many tokens, in passes "
                        "(daisy/data/pool.py); 0 = the whole pool")
    g.add_argument("--daisy_pool_keep", default="", type=str,
                   help="directory with a per-document uint8 score of every pool shard (e.g. the FineWeb-Edu score of "
                        "daisy/targets/edu_scores.py); only documents with score >= --daisy_pool_keep_min are drawn")
    g.add_argument("--daisy_pool_keep_min", default=3, type=int)
    g.add_argument("--daisy_pool_exclude", default="", type=str,
                   help="directory with text_hash.npy / url_hash.npy of documents that are never trained on, in "
                        "addition to the index's held-out documents")
    g.add_argument("--daisy_phi_path", default="", type=str,
                   help="another location of the index's embedding file (e.g. a copy on a RAM disk)")
    g.add_argument("--daisy_max_doc_tokens", default=2048, type=int, help="documents are capped at this many tokens")
    g.add_argument("--daisy_prefetch", default=8, type=int, help="micro-batches planned and read ahead (0 = off)")
    g.add_argument("--daisy_io_threads", default=64, type=int, help="parallel read workers per rank")

    g = p.add_argument_group("selection")
    g.add_argument("--daisy_select_mult", default=0.0, type=float,
                   help="the window factor lambda: each micro-batch is selected from a window of lambda x its tokens; "
                        "0 = no selection (uniform sampling)")
    g.add_argument("--daisy_learn", action="store_true",
                   help="DAiSY: label the trained documents and fit the ridge scorer on them; with "
                        "--daisy_select_mult > 0 the ridge selects")
    g.add_argument("--daisy_select_static", default="", type=str,
                   help="the similarity baseline: a fixed scorer [w; b] (769,) or (K, 769) (maximum over K), written by "
                        "daisy/scorers/similarity.py")
    g.add_argument("--daisy_select_cosine", action="store_true",
                   help="score by w^T phi / |phi| + b (with --daisy_select_static)")
    g.add_argument("--daisy_select_dsir", default="", type=str,
                   help="the DSIR baseline: directory of precomputed importance weights (daisy/scorers/dsir_build.py)")
    g.add_argument("--daisy_scorer_start", default=2500, type=int,
                   help="t0: the step from which the scorer (ridge or fixed) selects; before it the micro-batches are "
                        "uniformly random subsets of their windows")

    g = p.add_argument_group("labels and scorer (--daisy_learn)")
    g.add_argument("--daisy_targets", default="arc_easy,arc_challenge,piqa,sciq,hellaswag", type=str,
                   help="the target tasks: directories under --daisy_target_dir with train.bin / train.len")
    g.add_argument("--daisy_target_dir", default="", type=str)
    g.add_argument("--daisy_target_tokens", default=32768, type=int, help="target tokens per task")
    g.add_argument("--daisy_label_start", default=2000, type=int, help="the first step with label gates")
    g.add_argument("--daisy_label_every", default=4, type=int, help="L: labels on every L-th step")
    g.add_argument("--daisy_no_task_labels", action="store_true",
                   help="keep only the combined target reference at a refresh (saves memory with many target tasks; "
                        "the selection is unchanged)")
    g.add_argument("--daisy_scorer_update_every", default=10, type=int, help="U: steps between ridge refits")
    g.add_argument("--daisy_ridge_reg", default=0.01, type=float,
                   help="ridge regularization, as a multiple of the mean eigenvalue of the Gram matrix")
    g.add_argument("--daisy_ridge_decay", default=0.999, type=float, help="gamma: forgetting factor per labelled step")
    g.add_argument("--daisy_metrics_every", default=200, type=int, help="steps between scorer-metric records")

    g = p.add_argument_group("evaluation during training")
    g.add_argument("--daisy_refresh", default=200, type=int,
                   help="R: steps between refreshes (target gradient, held-out losses)")
    g.add_argument("--daisy_eval_dir", default="", type=str,
                   help="directory of the evaluation tasks (val.bin / val.len): the held-out downstream loss, every R "
                        "steps; '' = off")
    g.add_argument("--daisy_eval_tasks", default="", type=str,
                   help="the evaluation tasks; '' = --daisy_targets (set it when the targets are not the benchmarks)")
    g.add_argument("--daisy_eval_tokens", default=65536, type=int, help="evaluation tokens per task")
    g.add_argument("--daisy_val_docs_per_topic", default=64, type=int, help="held-out web-text documents per topic")
    g.add_argument("--daisy_val_max_doc_tokens", default=1024, type=int)
    g.add_argument("--daisy_val_rows_per_batch", default=4, type=int)

    # the MATES baseline (daisy/mates/, docs/mates_baseline.md): uniform training over the pool whose data is re-selected
    # at every stage boundary; nothing below is read unless --mates is set
    g = p.add_argument_group("MATES baseline")
    g.add_argument("--mates", action="store_true",
                   help="MATES: stages of --mates_stage_steps; at each boundary one-step oracle probes, a BERT data "
                        "influence model, Gumbel-top-k selection of the next stage's documents (needs --daisy_pool)")
    g.add_argument("--mates_stage_steps", default=32000, type=int, help="U: steps per stage (stage 1 is uniform)")
    g.add_argument("--mates_first_boundary", default=0, type=int,
                   help="F: the first stage boundary (stage 1 = steps < F); later ones every U steps (F + U, F + 2U, "
                        "...); 0 = U")
    g.add_argument("--mates_probes_first", default=80000, type=int, help="oracle probes at the first boundary (all ranks)")
    g.add_argument("--mates_probes_later", default=20000, type=int, help="oracle probes at every later boundary (all ranks)")
    g.add_argument("--mates_ratio", default=0.2, type=float, help="selected share of the candidate tokens: candidates = stage token budget / ratio per rank")
    g.add_argument("--mates_tau", default=1.0, type=float, help="Gumbel-top-k temperature")
    g.add_argument("--mates_probe_shards", default=8, type=int,
                   help="the last N shards of every rank's pool shard list: reserved for probe documents, never "
                        "trained on")
    g.add_argument("--mates_probe_precision", default="bf16", choices=["bf16", "fp32"],
                   help="probe document step and reference forward: bf16 autocast (as training) or fp32 (TF32 off)")
    g.add_argument("--mates_ref_rows_per_batch", default=5, type=int, help="reference target rows per forward batch")
    g.add_argument("--mates_bert_dir", default="", type=str, help="local bert-base-uncased directory (weights, config, tokenizer.json or vocab.txt)")
    g.add_argument("--mates_gpt2_tokenizer", default="", type=str,
                   help="GPT-2 tokenizer.json for decoding documents ('' = the Hugging Face hub cache, daisy/tok.py)")
    g.add_argument("--mates_bert_epochs", default=5, type=int)
    g.add_argument("--mates_bert_batch", default=256, type=int, help="global batch of the influence-model fine-tuning")
    g.add_argument("--mates_bert_micro_batch", default=32, type=int, help="documents per rank and forward (accumulated)")
    g.add_argument("--mates_bert_lr", default=5e-5, type=float)
    g.add_argument("--mates_bert_wd", default=0.01, type=float)
    g.add_argument("--mates_bert_val_share", default=0.1, type=float, help="validation share of the probes")
    g.add_argument("--mates_bert_eval_every", default=20, type=int, help="fine-tuning steps between validations")
    g.add_argument("--mates_score_padded", action="store_true",
                   help="score with the layout of the MATES paper (every document padded to 2 x 512) instead of length "
                        "buckets")
    g.add_argument("--mates_score_bucket", default=64, type=int, help="length-bucket granularity of the scoring chunks")
    g.add_argument("--mates_score_tokens", default=65536, type=int, help="padded tokens per scoring batch")
    g.add_argument("--mates_tok_workers", default=-1, type=int, help="CPU threads per rank for read + decode + WordPiece (-1: the rank's share of the cores minus 2; 0: in the "
                        "calling thread)")
    g.add_argument("--mates_init_from", default="", type=str,
                   help="a permanent checkpoint directory of another run (the uniform run at the first boundary) to "
                        "start from when this run has no checkpoint of its own: model, optimizer, schedule, random "
                        "generators, stream position")
    g.add_argument("--mates_cost_offset", default=-1.0, type=float,
                   help="GPU-hours spent before this run (stage 1); < 0: with --mates_init_from, that run's steady "
                        "seconds per step x the steps x its GPUs, else 0")
    g.add_argument("--mates_budget", default=0.0, type=float, help="stop like a finished run when the GPU-hours (offset + training steps + boundaries) reach this (0 = off)")
    g.add_argument("--mates_boundary_reserve_min", default=0.0, type=float,
                   help="a chained job (DAISY_STOP_AT) with less than this many minutes left stops before a stage "
                        "boundary instead of starting it")
    return p


def parse_args(args=None):
    return build_parser().parse_args(args)
