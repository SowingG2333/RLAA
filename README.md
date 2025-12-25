# RLAA: Rational Localized Adversarial Anonymization

RLAA is a fully localized, training-free anonymization framework designed to resolve the "privacy paradox" by eliminating the need to send raw sensitive text to third-party APIs.

## 🌟 Core Architecture: A-A-A

Unlike traditional greedy adversarial strategies that often lead to utility collapse on local small-scale models (LSMs), RLAA introduces an **Attacker-Arbitrator-Anonymizer** architecture:

![RLAA Framework](assets/RLAA.jpg)

* **Attacker**: Acts as a sensory module to identify potential identity leaks and provide reasoning chains.
* **Arbitrator**: Functions as a rationality gatekeeper, validating attacker inferences to filter out ghost leaks.
* **Anonymizer**: Executes precise and minimal modifications based on the validated feedback to preserve semantic integrity.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt

```

## 🚀 Quick Start

### 1. Run RLAA Inference

Navigate to a task directory (e.g., `PersonalReddit`) and execute the runner script:

```bash
cd PersonalReddit
export MODEL_PATH="path/to/llama-3-8b-instruct" # Specify your local model path
bash script/run_rlaa.sh

```

### 2. Evaluation

Evaluation often requires an external strong model (e.g., DeepSeek-V3) as a judge or adversary:

```bash
export API_KEY="your_api_key_here"
bash script/eval.sh

```

## 📂 Repository Structure

The repository is organized by task. Each subdirectory contains its own `data/`, `script/`, and `src/` folders:

* **PersonalReddit/**: Multi-attribute synthetic dataset, runner scripts, and core source code for the PersonalReddit task.
* **reddit-self-disclosure/**: Real-world health disclosure dataset and scripts for the health-issue inference task.
* **assets/**: Project diagrams and documentation.