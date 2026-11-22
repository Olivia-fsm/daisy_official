"""DAiSY: online data selection by adaptive utility estimation for LLM pretraining (see README.md).

  models/        the Llama-style decoder, document-masked attention, the label gates
  data/          documents: the index store, the tokenized pool stream, the candidate-window selector, prefetching
  labels/        utility labels from the training backward pass, and the selection targets they are taken against
  scorers/       the running ridge (DAiSY) and the fixed scorers of the baselines (similarity, DSIR)
  train/         the training loop (trainer.py), the online learner (learner.py), checkpoints
  eval/          held-out losses during training and of saved checkpoints; zero-shot accuracy with lm-evaluation-harness
  index/, tokenize/   the embedding index and the tokenized pool (offline, once)
  targets/       the benchmark target and the web-text targets
  mates/         the MATES baseline
  sft/           instruction tuning of pretrained checkpoints
  multilingual/  the multilingual experiments (FineWeb2)
"""
