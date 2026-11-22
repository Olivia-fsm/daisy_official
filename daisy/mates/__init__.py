"""The MATES baseline (Yu et al. 2024, arXiv 2406.06046) inside the DAiSY trainer (--mates).

Training runs in stages of U steps (--mates_stage_steps). Stage 1 is the uniform pool stream, up to the first
boundary F (--mates_first_boundary, default U). At every stage boundary t = F + k U (k >= 0), before step t
(stage.py: MatesController):
  1. probes      probes.py: n_t fresh documents from each rank's reserved pool shards; per document one AdamW step of
                 the run's optimizer from the current model + optimizer state (unwrapped model, clip, lr of step t),
                 the oracle o = L_r(before) - L_r(after) on the target tasks' train-split rows; state restored;
  2. fine-tune   influence.py: the data influence model (BERT over the document's first 1024 WordPiece tokens, two
                 512-token chunks, masked mean, linear head) regresses the z-scored oracle (DDP over the ranks);
  3. candidates  streams.py: every rank draws documents from its live pool stream until they hold 5x its stage token
                 budget; text.py turns their GPT-2 tokens into WordPiece ids on CPU workers, influence.py scores them;
  4. selection   select.py: Gumbel-top-k (tau) by tokens down to the stage budget, the last document cut, the list
                 shuffled; streams.py: StageListStream plays it for the next U steps.
The run stops at a GPU-hour budget that counts every MATES cost (stage.py: budget). The specification is
docs/mates_baseline.md; outputs go to EXP/daisy/mates/.
"""
