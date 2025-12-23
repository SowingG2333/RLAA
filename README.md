# RLAA: Rational Localized Adversarial Anonymization

This repository contains the official PyTorch implementation for the paper **"Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization"**.

## 🚀 Overview

**RLAA** is a fully localized, training-free framework designed to resolve the "Privacy Paradox" in text anonymization—where users must disclose sensitive data to untrusted third-party APIs for protection.

Unlike naive adversarial strategies that suffer from severe **utility collapse** (over-editing) when migrated to local small models (LSMs), RLAA introduces a novel **Attacker-Arbitrator-Anonymizer (A-A-A)** architecture.

### Key Features
* **Rationality Gatekeeper**: An Arbitrator module that validates attacker inferences, filtering out "ghost leaks" (hallucinations) and negligible privacy gains.
* **Economic Efficiency**: Prevents the "diminishing returns" problem by enforcing a rational stopping criterion, modeled via Marginal Rate of Substitution (MRS).
* **Fully Localized**: Optimized for consumer-grade GPUs (e.g., RTX 3090/4090/5090) using quantized 8B/7B models (Llama-3, Qwen-2.5), eliminating API-based privacy risks.

## 📂 Project Structure

```text
.
├── data/               # Dataset files
├── script/             # Shell scripts for running experiments
├── src/                # Source code (Python)
├── requirements.txt    # Dependencies
└── README.md

```

## 📦 Requirements

* python >= 3.8
* pytorch >= 2.0.0
* transformers >= 4.30.0
* Install dependencies:
```bash
pip install -r requirements.txt

```

## 📊 Data Preparation

This repository currently includes the **PersonalReddit** dataset for immediate reproduction of our results.

### Note on `reddit-self-disclosure` Dataset

The **reddit-self-disclosure** dataset used in the paper is not included in this repository due to licensing restrictions which prohibit direct redistribution without author permission.

> **Update Plan:** We are preparing the data processing scripts and related code to allow users to construct the dataset from the source. These will be released in a future update.

## ⚡ Quick Start

### 1. Environment Setup

Set up your API key (only required for using DeepSeek as the *Evaluator* or for *Data Generation*; the core RLAA framework runs locally).

```bash
export API_KEY="your_api_key_here"

```

### 2. Run Baselines (FgAA)

Reproduce the baseline results (Naive Migration and SFT variants):

```bash
# Naive Mode (Shared Model, creates utility collapse)
bash script/run_fgaa_naive.sh

# SFT Mode (Separate Attacker/Anonymizer)
bash script/run_fgaa_sft.sh

```

### 3. Run RLAA (Main Method)

Run the proposed Rational Agent Framework:

```bash
bash script/run_rlaa.sh

```

### 4. Evaluation

Evaluate the utility (readability, meaning preservation) and privacy risk (attacker success rate) of the generated outputs:

```bash
bash script/eval.sh

```

## 🧪 Experiments & Training

If you wish to train the Attacker or Anonymizer components from scratch (e.g., for the SFT baseline comparison), use the provided training script:

```bash
bash script/sft.sh

```

*Note: RLAA itself is a training-free framework and does not require this step for inference.*