# RLAA: Rational Localized Adversarial Anonymization

This repository provides a reference implementation of RLAA (Rational Localized Adversarial Anonymization).

RLAA is a fully-localized, training-free anonymization framework designed to avoid the “privacy paradox”, where users would otherwise need to send raw sensitive text to third-party APIs.

## Repository Structure (Up-to-date)

The repo is organized by dataset/task. Each subdirectory is self-contained with its own `data/`, `script/`, and `src/`:

```text
.
├── PersonalReddit/
│   ├── data/                 # Example jsonl files
│   ├── script/               # Runner scripts (RLAA / FgAA / SFT / eval / gen_data)
│   └── src/                  # Main code entrypoints
├── reddit-self-disclosure/
│   ├── data/                 # Important: no redistribution (see data/README.md)
│   ├── script/
│   └── src/
├── requirements.txt
└── README.md
```

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick Start (Run from a Subdirectory)

All commands below should be executed inside a task directory (e.g., `PersonalReddit/` or `reddit-self-disclosure/`).

### 1) Run RLAA (Main Method)

```bash
cd PersonalReddit

# Optional: override defaults via environment variables
export MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export INPUT_FILE="data/test.jsonl"
export OUTPUT_FILE="results/rlaa_output.jsonl"
export MAX_ITERATIONS=10

bash script/run_rlaa.sh
```

### 2) Run Baselines (FgAA)

```bash
cd PersonalReddit

# Naive migration baseline
bash script/run_fgaa_naive.sh

# SFT baseline (may trigger training/finetuning scripts)
bash script/run_fgaa_sft.sh
```

### 3) Evaluation (Usually Requires API_KEY)

The evaluation script typically uses hosted API models (e.g., `deepseek-chat`) as the judge/adversary, so you need to export `API_KEY` first:

```bash
cd PersonalReddit

export API_KEY="your_api_key_here"

# You may override: INPUT_FILE / OUTPUT_FILE / JUDGE_MODEL / ADVERSARY_MODEL / WORKERS / LIMIT
bash script/eval.sh
```

## Data Notes

- `PersonalReddit/data/`: example jsonl files are included for quick reproduction.
- `reddit-self-disclosure/data/`: redistribution is restricted by the dataset authors. Please read `data/README.md` in that directory.

## Entry Points (Per Subdirectory)

Key scripts live under each task directory:

- `src/run_rlaa.py`: RLAA inference entry
- `src/eval.py`: evaluation entry (often API-based)
- `script/run_rlaa.sh` / `script/eval.sh`: recommended reproducible runner scripts (support env-var overrides)