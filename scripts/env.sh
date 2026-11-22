# Locations of the data and of the experiment outputs, sourced by every script of this directory.
# Replace DATAPATH by a directory of your own (or export the variables before calling a script).

export DATAPATH=${DATAPATH:-DATAPATH}

# inputs you provide
export EMBEDDED_POOL=${EMBEDDED_POOL:-$DATAPATH/fineweb-embedded}   # FineWeb parquet files with XLM-RoBERTa embeddings (README, "Data")
export FINEWEB_EDU=${FINEWEB_EDU:-$DATAPATH/fineweb-edu-score-2}     # a local copy of FineWeb-Edu (score-2 version), for the FineWeb-Edu target
export BERT=${BERT:-$DATAPATH/models/bert-base-uncased}             # a local copy of bert-base-uncased, for the MATES baseline

# what the pipeline builds
export INDEX=${INDEX:-$DATAPATH/index}             # 1. the embedding index (held-out web-text set, hashes, topics)
export POOL=${POOL:-$DATAPATH/pool}                # 2. the tokenized pool with embedding sidecars
export BENCH=${BENCH:-$DATAPATH/benchmarks}        # 3. the benchmark target tasks (train and val splits)
export TARGETS=${TARGETS:-$DATAPATH/targets}       # 3. the web-text targets: fineweb_topics/, fineweb_edu/
export SCORERS=${SCORERS:-$DATAPATH/scorers}       # 3. the fixed scorers of the baselines: similarity/, dsir/
export EDU_SCORES=${EDU_SCORES:-$DATAPATH/edu_scores}   # 3. the FineWeb-Edu score of every pool document
export EXPS=${EXPS:-$DATAPATH/exps}                # 4. one directory per run
export EVALS=${EVALS:-$DATAPATH/evals}             # 5. evaluations of checkpoints

export HF_HOME=${HF_HOME:-$DATAPATH/hf_cache}      # Hugging Face cache (tokenizers, benchmark datasets, XLM-RoBERTa)
