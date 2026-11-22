# Instruction tuning

Instruction tuning (supervised fine-tuning, SFT) of pretrained checkpoints on the Tulu 3 mixture, for the experiment
"Do the gains survive instruction tuning?". Every pretraining run is tuned on the same conversations in the same
order, with the same optimizer and schedule, so the tuned models differ only in the checkpoint they start from.

| file | what it does |
|---|---|
| `data.py` | renders a conversation with the Tulu chat template as text (GPT-2 has no chat tokens), with the loss on the assistant turns only; packs whole conversations into document-masked rows |
| `prep_tulu.py` | downloads `allenai/tulu-3-sft-mixture`, holds out a seeded validation subset and tokenizes both splits |
| `sft_train.py` | the trainer (one process or DDP under `torchrun`): AdamW from a fresh optimizer state, warmup, a linear or warmup-stable-decay schedule, validation loss, a checkpoint per epoch, restart from a saved state |

Each file's docstring describes its formats and options in full. All commands run from the repository root; `DATAPATH`
and `EXPS` are the locations of `scripts/env.sh`.

1. **Data** (CPU; needs `datasets` and `transformers`): writes the tokens, the loss mask and the conversation offsets
   of the training and validation splits.

   ```bash
   python daisy/sft/prep_tulu.py --out DATAPATH/tulu3-sft [--val 2000] [--workers 64]
   ```

2. **Tuning** of one pretraining run `EXPS/RUN` (its `ckpts/latest`, or `--ckpt STEP`). The output directory has the
   layout of a pretraining run (`daisy/meta_train.json`, `ckpts/latest/main.pt`), plus `sft/metrics.jsonl` (training
   and validation loss) and `sft/finished.json`.

   ```bash
   torchrun --standalone --nproc_per_node=4 daisy/sft/sft_train.py --base EXPS/RUN --data DATAPATH/tulu3-sft \
       --out EXPS/RUN-sft --lr 1e-4 --compile
   ```

   This is the recipe of the paper: one epoch of 6,278 steps of 64 rows (4 GPUs x 16 rows), AdamW with a fresh state,
   a peak learning rate of 1e-4, linear warmup over 3% of the steps, linear decay to zero and clipping at 1.0.

   The five-epoch runs of the paper add `--epochs 5 --schedule wsd --save_epochs`: after the warmup the learning rate
   stays at its peak; before the end of each epoch a copy of the model is annealed linearly to zero over 0.2 epochs
   (`--cooldown_frac`) and saved as `ckpts/<step>`, and the run continues from the state before the copy, so the
   checkpoint of epoch `k` is a finished run of `k` epochs. `--save_every N` writes a state from which the same
   command continues.

3. **Evaluation**: the tuned run loads like a pretraining run.

   ```bash
   python daisy/eval/lm_eval_chat.py --run EXPS/RUN-sft --out EVALS/RUN-sft.json [--ckpt STEP]    # chat format
   python daisy/eval/lm_eval_run.py  --run EXPS/RUN-sft --out EVALS/RUN-sft.raw.json              # raw-text prompts
   ```
