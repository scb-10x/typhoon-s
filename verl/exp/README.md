# Section 3 Experiments: InK-GRPO for Sovereign Capability

Training scripts for **InK-GRPO** (Injected Knowledge GRPO) on sovereign capability tasks.

## Overview

- **`rl_sft/`** — Standard InK-GRPO (non-agentic, single-turn reasoning)
- **`agent/`** — Agentic InK-GRPO (multi-turn with RAG tools)
- **`data_processing/`** — Dataset preprocessing scripts

## Quick Start

### 1. Prepare Data

```bash
cd data_processing

# Process NitiBench dataset
python augment_nitibench.py \
    --input_path <path-to-raw-nitibench> \
    --output_dir <output-directory>

# Process MIRAGE-Bench dataset
python augment_mirage.py \
    --input_path <path-to-mirage> \
    --output_dir <output-directory>
```

### 2. Standard InK-GRPO Training (Section 3, RQ5-6)

Single-turn reasoning with InK-GRPO on NitiBench:

```bash
cd rl_sft/nitibench

# Edit train_nitibench_rl_pretrain.sh to set:
# - ROOT_DIR (your data directory)
# - LLM_JUDGE_API_KEY (for reward evaluation)
# - Other hyperparameters as needed

bash train_nitibench_rl_pretrain.sh
```

### 3. Agentic InK-GRPO Training (Section 3, RQ7)

Multi-turn with RAG tools (search + read):

```bash
cd agent/nitibench

# Edit train_nitibench_agent_pretrain.sh to set:
# - ROOT_DIR (your data directory)
# - MODEL_PATH, experiment parameters

bash train_nitibench_agent_pretrain.sh
```

The script automatically:
1. Starts NitiBench RAG server (with FAISS index)
2. Runs agentic InK-GRPO training
3. Cleans up server on exit

## Hardware Requirements

- **Standard InK-GRPO:** 4×H100 (80GB), ~1 day training
- **Agentic InK-GRPO:** 4×H100 (80GB), ~1 day training
